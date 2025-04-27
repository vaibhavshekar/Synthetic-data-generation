# Conditional Restricted Boltzmann Machines for Class-Conditioned Data Generation

## Overview
This repository implements a novel Conditional Restricted Boltzmann Machine (CRBM) framework for addressing class imbalance in machine learning datasets. The CRBM incorporates class-specific conditioning vectors into the energy function, enabling targeted data generation for minority classes while preserving intricate feature relationships.

## Features
- **Class-Conditioned Generation**: Produces high-quality synthetic samples for minority classes
- **Energy-Based Architecture**: Offers superior training stability compared to GANs
- **Effective with Limited Data**: Performs well even with scarce minority class samples
- **Interpretable Weight Structures**: Provides transparency into class-specific feature contributions

## Implementation Details
- PyTorch implementation with batch normalization and kernel density estimation (KDE)
- Contrastive Divergence (CD-1) training algorithm
- Gradient clipping and temperature scaling for training stability
- Gibbs sampling with KDE refinement for continuous features

## Results
The CRBM demonstrates significant improvements in classification performance on imbalanced datasets:

### Credit Card Fraud Detection Dataset (0.17% minority class)
- F1-score improvement: 8.7% over baseline
- Jensen-Shannon divergence: Mean 0.37
- Preserved feature correlations with statistical significance (p < 0.01)

### Glass Identification Dataset (multiclass imbalance)
- F1-score improvement: 14.3% over baseline
- Jensen-Shannon divergence: Mean 0.316
- Maintained distributional alignment validated through t-SNE visualizations

### Installation
```bash
pip install -r requirements.txt

