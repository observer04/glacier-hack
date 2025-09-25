# Kaggle Notebook Setup for Glacier Hack

## Cell 1: Initial Setup and Data Download
```python
import os
import subprocess
import torch
import numpy as np
from pathlib import Path

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Set working directory
os.chdir('/kaggle/working')
print(f"Current directory: {os.getcwd()}")
```

## Cell 2: Clone Repository and Download Data
```python
# Clone your repository
!git clone https://github.com/observer04/glacier-hack.git
os.chdir('/kaggle/working/glacier-hack')

# Download and extract training data
!wget https://www.glacier-hack.in/train.zip
!unzip -q train.zip -d ./
!mv ./Train/Train/* ./Train/
!rmdir ./Train/Train

# Verify data structure
print("✅ Data structure:")
!ls -la Train/ | head -10
print(f"Total training files: {len(os.listdir('Train/'))}")
```

## Cell 3: Install Dependencies
```python
# Install required packages
!pip install tqdm scikit-learn matplotlib pillow tifffile

# Import and verify installation
import tqdm
import sklearn
import matplotlib.pyplot as plt
from PIL import Image
import tifffile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

print("✅ All dependencies installed successfully!")
```

## Cell 4: Quick Data Verification
```python
# Verify we can load the training modules
import sys
sys.path.append('/kaggle/working/glacier-hack')

try:
    from data_utils import GlacierDataset, compute_global_stats
    from models import UNet
    from train_utils import TverskyLoss
    print("✅ All modules imported successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")

# Quick data check
import glob
train_files = glob.glob('Train/*.tif')
print(f"Found {len(train_files)} training files")
print("Sample files:", train_files[:3])
```

## Cell 5: Start Training (UNet + Tversky - Recommended)
```python
# Start the optimized training
!python train_model.py \
    --model_type unet \
    --loss_type tversky \
    --batch_size 2 \
    --epochs 80 \
    --lr 0.001 \
    --save_dir /kaggle/working/models \
    --use_amp \
    --use_swa \
    --threshold_sweep \
    --scheduler plateau \
    --normalize_type global \
    --data_dir Train \
    --patience 15 \
    --gradient_accumulation_steps 4
```

## Alternative Training Commands

### High Performance (if GPU memory allows)
```python
# If you have enough GPU memory, try batch_size 4
!python train_model.py \
    --model_type unet \
    --loss_type tversky \
    --batch_size 4 \
    --epochs 60 \
    --lr 0.002 \
    --save_dir /kaggle/working/models \
    --use_amp \
    --use_swa \
    --threshold_sweep \
    --scheduler cosine \
    --normalize_type global \
    --data_dir Train \
    --patience 10
```

### Memory-Constrained Training
```python
# If you encounter memory issues
!python train_model.py \
    --model_type unet \
    --loss_type tversky \
    --batch_size 1 \
    --epochs 100 \
    --lr 0.0005 \
    --save_dir /kaggle/working/models \
    --use_amp \
    --use_swa \
    --threshold_sweep \
    --scheduler plateau \
    --normalize_type global \
    --data_dir Train \
    --patience 20 \
    --gradient_accumulation_steps 8
```

## Cell 6: Monitor Training Progress
```python
# Monitor training (run this in a separate cell while training)
import time
import matplotlib.pyplot as plt

def monitor_training():
    """Monitor training progress by reading logs"""
    log_files = glob.glob('/kaggle/working/models/*/training.log')
    if log_files:
        latest_log = max(log_files, key=os.path.getctime)
        print(f"Monitoring: {latest_log}")
        !tail -20 {latest_log}
    else:
        print("No training logs found yet...")

# Call this function periodically
monitor_training()
```

## Cell 7: Save Results to Kaggle Output
```python
# After training completes, save results
import shutil

# Create output directory
os.makedirs('/kaggle/working/final_output', exist_ok=True)

# Find the best model
model_dirs = glob.glob('/kaggle/working/models/*')
if model_dirs:
    latest_model_dir = max(model_dirs, key=os.path.getctime)
    print(f"Latest model directory: {latest_model_dir}")
    
    # Copy best model and solution.py
    best_model = glob.glob(f'{latest_model_dir}/best_model.pth')
    if best_model:
        shutil.copy(best_model[0], '/kaggle/working/final_output/model.pth')
        shutil.copy('solution.py', '/kaggle/working/final_output/')
        print("✅ Model and solution copied to final_output/")
    
    # Copy training logs and plots
    for file_pattern in ['*.log', '*.png', '*.txt']:
        files = glob.glob(f'{latest_model_dir}/{file_pattern}')
        for file in files:
            shutil.copy(file, '/kaggle/working/final_output/')
    
    print("Files in final_output:")
    !ls -la /kaggle/working/final_output/
```

## Expected Results
- **Training Time**: 2-3 hours for 80 epochs
- **Expected MCC**: 70-75% (breaking through the 60% plateau)
- **Memory Usage**: ~6-8GB GPU memory with batch_size=2
- **Files Generated**: model.pth, solution.py, training logs, plots

## Troubleshooting

### GPU Memory Issues
```python
# If you get CUDA out of memory:
# 1. Reduce batch_size to 1
# 2. Increase gradient_accumulation_steps to 8
# 3. Use smaller model if needed
```

### Slow Training
```python
# If training is too slow:
# 1. Increase batch_size if memory allows
# 2. Reduce epochs to 60
# 3. Use cosine scheduler instead of plateau
```

### Monitor GPU Usage
```python
# Check GPU utilization during training
!nvidia-smi
```