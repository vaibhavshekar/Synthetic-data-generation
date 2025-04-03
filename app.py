import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import time
import io

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CRBM(nn.Module):
    def __init__(self, n_visible, n_hidden, n_conditions):
        super(CRBM, self).__init__()
        
        # Xavier initialization for stability
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * np.sqrt(2.0/(n_visible + n_hidden)))
        self.b = nn.Parameter(torch.zeros(n_visible))
        self.a = nn.Parameter(torch.zeros(n_hidden))
        self.U = nn.Parameter(torch.randn(n_conditions, n_hidden) * np.sqrt(2.0/(n_conditions + n_hidden)))
        
        self.batch_norm = nn.BatchNorm1d(n_hidden)  # Adding batch normalization

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_conditions = n_conditions
    
    def free_energy(self, v, c):
        """Compute the free energy with KL regularization"""
        vW = torch.matmul(v, self.W)
        cU = torch.matmul(c, self.U)
        hidden_activations = self.batch_norm(self.a + vW + cU)  # Apply batch norm
        hidden_term = F.softplus(hidden_activations).sum(dim=1)

        # KL divergence term to prevent mode collapse
        kl_term = 0.01 * torch.sum(self.W ** 2)
        
        return -torch.matmul(v, self.b) - hidden_term + kl_term
    
    def sample_h_given_v_c(self, v, c, temperature=1.0):
        """Sample hidden units with temperature scaling"""
        v, c = v.to(device), c.to(device)
        activation = (torch.matmul(v, self.W) + torch.matmul(c, self.U) + self.a) / temperature
        p_h = torch.sigmoid(activation)
        return p_h, torch.bernoulli(p_h)
    
    def sample_v_given_h(self, h):
        """Sample visible units given hidden units with clamping"""
        h = h.to(device)
        activation = torch.matmul(h, self.W.t()) + self.b
        p_v = torch.sigmoid(torch.clamp(activation, -80, 80))
        return p_v, torch.bernoulli(p_v)
    
    def gibbs_sampling(self, v, c, k=50):
        """Perform k-step Gibbs sampling"""
        v, c = v.to(device), c.to(device)
        v_k = v
        for _ in range(k):
            p_h, h = self.sample_h_given_v_c(v_k, c)
            p_v, v_k = self.sample_v_given_h(h)
        return p_v, v_k, p_h, h
    
    def contrastive_divergence(self, v_pos, c, k=1):
        """Contrastive Divergence with k steps"""
        v_pos, c = v_pos.to(device), c.to(device)
        p_h_pos, h_pos = self.sample_h_given_v_c(v_pos, c)
        p_v_neg, v_neg, p_h_neg, h_neg = self.gibbs_sampling(v_pos, c, k)
        return p_h_pos, h_pos, p_v_neg, v_neg, p_h_neg, h_neg
    
    def forward(self, v, c):
        """Forward pass"""
        p_h, h = self.sample_h_given_v_c(v, c)
        return p_h, h

def load_credit_card_data():
    try:
        data = pd.read_csv('creditcard.csv')
    except FileNotFoundError:
        st.error("Could not find creditcard.csv. Please upload the file.")
        return None, None, None, None, None
    
    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # One-hot encode the target
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.values.reshape(-1, 1))
    
    return X_scaled, y, y_onehot, scaler, encoder

def load_glass_data():
    try:
        data = pd.read_csv('glass.csv')
    except FileNotFoundError:
        st.error("Could not find glass.csv. Please upload the file.")
        return None, None, None, None, None
    
    # Separate features and target
    X = data.drop('Type', axis=1)
    y = data['Type']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # One-hot encode the target
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.values.reshape(-1, 1))
    
    return X_scaled, y, y_onehot, scaler, encoder

def train_crbm(model, X_train, y_onehot_train, batch_size=64, num_epochs=20, lr=0.01, k=1):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_onehot_train_tensor = torch.FloatTensor(y_onehot_train).to(device)

    dataset = TensorDataset(X_train_tensor, y_onehot_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_history = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (v, c) in enumerate(dataloader):
            v, c = v.to(device), c.to(device)
            
            p_h_pos, h_pos, p_v_neg, v_neg, p_h_neg, h_neg = model.contrastive_divergence(v, c, k)
            loss = torch.mean(model.free_energy(v, c) - model.free_energy(v_neg, c))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_epoch_loss)
        
        progress = (epoch + 1) / num_epochs
        progress_bar.progress(progress)
        status_text.text(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}')
        scheduler.step()
    
    return loss_history

def generate_synthetic_samples(model, condition_vector, n_samples=100, gibbs_steps=50):
    v = torch.rand(n_samples, model.n_visible)
    c = condition_vector.repeat(n_samples, 1)
    
    for _ in range(gibbs_steps):
        p_h, h = model.sample_h_given_v_c(v, c, temperature=0.8)
        p_v, v = model.sample_v_given_h(h)
    
    return v.cpu().detach().numpy()

def generate_realistic_time_distribution(original_df, synthetic_df):
    """Match time distribution using KDE (for credit card data only)"""
    if 'Time' in original_df.columns:
        fraud_times = original_df[original_df['Class'] == 1]['Time'].values
        if len(fraud_times) > 1:  # Need at least 2 samples for KDE
            kde = gaussian_kde(fraud_times)
            synthetic_times = kde.resample(len(synthetic_df)).flatten()
            synthetic_df['Time'] = np.clip(synthetic_times, original_df['Time'].min(), original_df['Time'].max())
    return synthetic_df

def visualize_results(original_df, balanced_df, feature_columns, target_col, n_samples=1000):
    """
    Visualize the results of the synthetic data generation.
    """
    st.subheader("Visualization Results")
    
    # 1. Class Distribution Before and After
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    original_counts = original_df[target_col].value_counts().sort_index()
    balanced_counts = balanced_df[target_col].value_counts().sort_index()

    class_dist_df = pd.DataFrame({
        'Original': original_counts,
        'Balanced': balanced_counts
    })

    class_dist_df.plot(kind='bar', ax=ax1)
    plt.title('Class Distribution Before and After Balancing')
    plt.ylabel('Count')
    plt.xlabel('Class')
    plt.xticks(rotation=0)
    st.pyplot(fig1)
    
    # 2. KDE Comparison of Top Features
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    top_features = feature_columns[:3]
    class_labels = sorted(original_df[target_col].unique())

    for feature in top_features:
        for label in class_labels:
            sns.kdeplot(original_df[original_df[target_col] == label][feature], 
                        label=f'Orig Class {label}', ax=ax2)
            sns.kdeplot(balanced_df[balanced_df[target_col] == label][feature], 
                        label=f'Synth Class {label}', linestyle='--', ax=ax2)
        break  # show only first feature for clarity
    plt.title('Feature KDE: Real vs Synthetic per Class')
    plt.xlabel('Feature Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig2)
    
    # 3. t-SNE Visualization
    st.write("Generating t-SNE visualization (this may take a while for large datasets)...")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    
    sample_orig = original_df.sample(min(n_samples, len(original_df)), random_state=42)
    sample_synth = balanced_df[len(original_df):].sample(min(n_samples, len(balanced_df) - len(original_df)), random_state=42)

    viz_data = pd.concat([sample_orig[feature_columns], sample_synth[feature_columns]])
    labels = pd.concat([sample_orig[target_col], sample_synth[target_col]])

    # Handle PCA dimension safely
    n_components = min(10, len(feature_columns), len(viz_data))
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(viz_data)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(pca_data)

    for label in sorted(labels.unique()):
        idx = labels == label
        ax3.scatter(tsne_results[idx, 0], tsne_results[idx, 1], label=f'Class {label}', alpha=0.6)
    plt.title('t-SNE: Real & Synthetic Class Distributions')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.legend()
    st.pyplot(fig3)
    
    # 4. Correlation Matrix Difference
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    orig_corr = original_df[feature_columns].corr()
    bal_corr = balanced_df[feature_columns].corr()
    corr_diff = bal_corr - orig_corr

    sns.heatmap(corr_diff, cmap='coolwarm', center=0, annot=False, fmt=".2f",
                cbar_kws={'label': 'Correlation Difference'}, ax=ax4)
    plt.title('Feature Correlation Change\n(Balanced - Original)')
    st.pyplot(fig4)

def main():
    st.title("Conditional Restricted Boltzmann Machine for Data Generation")
    st.write("This app generates synthetic samples to balance imbalanced datasets using a CRBM.")
    
    # Dataset selection
    dataset_option = st.sidebar.selectbox("Select Dataset", ["Credit Card Fraud", "Glass Identification"])
    
    # Advanced options expander
    with st.sidebar.expander("Advanced Options"):
        n_hidden = st.slider("Number of hidden units", 32, 256, 64, 32)
        num_epochs = st.slider("Number of epochs", 10, 100, 20, 5)
        learning_rate = st.slider("Learning rate", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")
        batch_size = st.slider("Batch size", 32, 256, 64, 32)
        gibbs_steps = st.slider("Gibbs sampling steps", 10, 100, 50, 10)
    
    # File upload option
    uploaded_file = st.sidebar.file_uploader("Or upload your own CSV file", type=["csv"])
    
    if st.sidebar.button("Run CRBM Training and Generation"):
        with st.spinner("Loading data and training model..."):
            start_time = time.time()
            
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.session_state.original_data = data
                    
                    # Assume last column is target
                    target_col = data.columns[-1]
                    X = data.drop(target_col, axis=1)
                    y = data[target_col]
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    encoder = OneHotEncoder(sparse_output=False)
                    y_onehot = encoder.fit_transform(y.values.reshape(-1, 1))
                    
                    X_train, _, y_train, _, y_onehot_train, _ = train_test_split(
                        X_scaled, y, y_onehot, test_size=0.2, stratify=y)
                    
                    feature_columns = X.columns.tolist()
                    target_column = target_col
                    
                except Exception as e:
                    st.error(f"Error processing uploaded file: {str(e)}")
                    return
            else:
                if dataset_option == "Credit Card Fraud":
                    X_scaled, y, y_onehot, scaler, encoder = load_credit_card_data()
                    if X_scaled is None:
                        return
                    feature_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
                    target_column = 'Class'
                else:  # Glass Identification
                    X_scaled, y, y_onehot, scaler, encoder = load_glass_data()
                    if X_scaled is None:
                        return
                    feature_columns = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']
                    target_column = 'Type'
                
                X_train, _, y_train, _, y_onehot_train, _ = train_test_split(
                    X_scaled, y, y_onehot, test_size=0.2, stratify=y)
            
            # Initialize and train CRBM
            crbm = CRBM(X_train.shape[1], n_hidden, y_onehot_train.shape[1])
            loss_history = train_crbm(
                crbm, X_train, y_onehot_train, 
                batch_size=batch_size, 
                num_epochs=num_epochs, 
                lr=learning_rate, 
                k=1
            )
            
            # Generate synthetic samples for minority classes to balance the dataset
            class_counts = pd.Series(y_train).value_counts()
            max_count = class_counts.max()
            synthetic_samples = []
            
            for class_label in np.unique(y_train):
                # Calculate how many samples to generate for this class
                n_samples = max_count - class_counts[class_label]
                if n_samples <= 0:
                    continue
                
                condition = torch.FloatTensor(encoder.transform([[class_label]]))
                class_samples = generate_synthetic_samples(
                    crbm, condition, 
                    n_samples=n_samples, 
                    gibbs_steps=gibbs_steps
                )
                synthetic_df = pd.DataFrame(scaler.inverse_transform(class_samples), columns=feature_columns)
                synthetic_df[target_column] = class_label
                synthetic_samples.append(synthetic_df)
            
            # Combine all synthetic samples
            synthetic_data = pd.concat(synthetic_samples, ignore_index=True)
            
            # For credit card data, adjust time distribution
            if dataset_option == "Credit Card Fraud":
                synthetic_data = generate_realistic_time_distribution(
                    st.session_state.original_data if 'original_data' in st.session_state else pd.DataFrame(X_scaled, columns=feature_columns),
                    synthetic_data
                )
            
            # Create balanced dataset
            original_df = st.session_state.original_data if 'original_data' in st.session_state else pd.DataFrame(X_scaled, columns=feature_columns)
            original_df[target_column] = y
            balanced_df = pd.concat([original_df, synthetic_data], ignore_index=True)
            
            # Save to session state for visualization
            st.session_state.balanced_df = balanced_df
            st.session_state.original_df = original_df
            st.session_state.feature_columns = feature_columns
            st.session_state.target_column = target_column
            
            # Plot training loss
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(loss_history, label='Training Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss Over Epochs')
            ax.legend()
            st.pyplot(fig)
            
            st.success(f"Completed in {time.time() - start_time:.2f} seconds")
    
    # Show results if available
    if 'balanced_df' in st.session_state:
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Original Class Distribution:")
            st.write(st.session_state.original_df[st.session_state.target_column].value_counts())
        
        with col2:
            st.write("Balanced Class Distribution:")
            st.write(st.session_state.balanced_df[st.session_state.target_column].value_counts())
        
        # Show visualizations
        visualize_results(
            st.session_state.original_df,
            st.session_state.balanced_df,
            st.session_state.feature_columns,
            st.session_state.target_column
        )
        
        # Download button for balanced dataset
        csv = st.session_state.balanced_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Balanced Dataset",
            data=csv,
            file_name='balanced_dataset.csv',
            mime='text/csv'
        )

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()