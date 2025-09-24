# GlacierHack 2025: Step-by-Step Guide

This document provides comprehensive instructions for developing a glacier segmentation model for the GlacierHack 2025 competition. Each step is detailed to ensure you can successfully complete the project from initial data exploration to final submission.

## Table of Contents
1. [Environment Setup](#1-environment-setup)
2. [Data Exploration](#2-data-exploration)
3. [Data Preparation](#3-data-preparation)
4. [Model Development](#4-model-development)
5. [Training Infrastructure](#5-training-infrastructure)
6. [Model Training](#6-model-training)
7. [Model Evaluation](#7-model-evaluation)
8. [Solution Preparation](#8-solution-preparation)
9. [Testing](#9-testing)
10. [Troubleshooting](#10-troubleshooting)

## 1. Environment Setup

### Local Environment
```bash
# Create virtual environment
python -m venv glacier_env
source glacier_env/bin/activate  # On Windows: glacier_env\Scripts\activate

# Install required packages
pip install torch torchvision numpy pillow scikit-learn tifffile matplotlib opencv-python tqdm joblib
```

### Google Colab Setup
```python
# Install required packages
!pip install torch torchvision numpy pillow scikit-learn tifffile matplotlib opencv-python tqdm joblib

# Mount Google Drive for saving models
from google.colab import drive
drive.mount('/content/drive')
```

## 2. Data Exploration

### Run the exploratory analysis script
```bash
# Run the exploratory analysis script
python exploratory_analysis.py
```

This script will:
- Check the dataset structure
- Analyze a sample image from each band
- Calculate class distribution
- Analyze band statistics

Pay special attention to:
- Class imbalance (glacier vs non-glacier pixels)
- Band statistics for normalization
- Image dimensions and types

## 3. Data Preparation

### Understanding the data utility module
The `data_utils.py` module provides:
- Dataset classes for loading and preprocessing images
- Functions for normalizing bands
- Dataloader creation for training

Key components:
- `GlacierDataset`: Custom dataset class for loading tiles
- `normalize_band()`: Normalize bands with mean and std
- `create_dataloaders()`: Create train and validation dataloaders

## 4. Model Development

### Available model architectures
Three model architectures are provided in `models.py`:

1. **PixelANN**: Simple pixel-wise neural network
   - Fast training and inference
   - Lower accuracy but robust baseline
   - Great for initial experimentation

2. **UNet**: Image segmentation architecture
   - Encoder-decoder with skip connections
   - Better captures spatial context
   - Good balance of accuracy and efficiency

3. **DeepLabV3+**: Advanced segmentation architecture
   - ASPP (Atrous Spatial Pyramid Pooling)
   - Better context understanding
   - Potentially highest accuracy but more complex

## 5. Training Infrastructure

### Training utilities
The `train_utils.py` module provides:
- Loss functions (BCE, Dice, Focal, Combined)
- Training and validation loops
- Model saving and early stopping
- Metrics calculation (MCC, F1, Precision, Recall)

## 6. Model Training

### Training procedure

#### 1. Start with the simplest model (PixelANN)
```bash
# Train a PixelANN model with default parameters
python train_model.py --model_type=pixelann --data_dir=./Train --batch_size=4096 --epochs=50
```

#### 2. Try different loss functions
```bash
# Train with focal loss for handling class imbalance
python train_model.py --model_type=pixelann --loss=focal --data_dir=./Train
```

#### 3. Train advanced architectures
```bash
# Train a UNet model
python train_model.py --model_type=unet --data_dir=./Train --batch_size=16 --epochs=30

# Train a DeepLabV3+ model
python train_model.py --model_type=deeplabv3plus --data_dir=./Train --batch_size=8 --epochs=30
```

## 7. Model Evaluation

### Evaluating trained models
```bash
# Evaluate the best PixelANN model
python evaluate_model.py --model_type=pixelann --model_path=./models/best_model.pth --data_dir=./Train

# Evaluate a UNet model
python evaluate_model.py --model_type=unet --model_path=./models/best_model.pth --data_dir=./Train
```

Focus on:
- Matthews Correlation Coefficient (primary metric)
- Confusion matrix analysis
- Precision-recall balance

## 8. Solution Preparation

### Preparing the submission file
1. Make sure your final trained model is saved as `model.pth`
2. Ensure `solution.py` contains the correct model definition and inference code
3. Test your solution to verify it works as expected

## 9. Testing

### Testing your solution
```bash
# Test your solution on the training data
python test_solution.py --data_dir=./Train --output_dir=./test_results
```

This will:
- Run your solution on the training data
- Compare predictions to ground truth
- Calculate metrics
- Generate visualizations

## 10. Troubleshooting

### Common Issues and Solutions

#### Memory Issues During Training

**Symptoms:**
- CUDA out of memory errors
- System freezes during training

**Solutions:**
1. Reduce batch size
   ```bash
   python train_model.py --model_type=pixelann --batch_size=1024
   ```

2. Use CPU instead of GPU if GPU memory is limited
   ```bash
   python train_model.py --model_type=pixelann --device=cpu
   ```

3. Reduce model complexity for UNet/DeepLabV3+
   - Try simpler architectures
   - Reduce number of filters

#### Poor Model Performance

**Symptoms:**
- Low MCC on validation set
- Model predicts all pixels as one class

**Solutions:**
1. Check class imbalance with exploratory analysis

2. Try different loss functions for imbalanced data
   ```bash
   python train_model.py --model_type=pixelann --loss=focal
   ```

3. Adjust threshold for binary classification
   - The default threshold is 0.5
   - Consider adjusting based on precision-recall tradeoff

#### File Path Issues

**Symptoms:**
- FileNotFoundError when running scripts
- Issues finding the model.pth file

**Solutions:**
1. Check relative vs absolute paths
2. Ensure model.pth is in the correct location
3. Verify directory structure matches expectations

---

Good luck with your GlacierHack 2025 project! Follow these steps systematically to develop a strong solution.