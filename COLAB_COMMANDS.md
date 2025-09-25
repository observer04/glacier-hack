# Optimized Training Commands for GlacierHack 2025
# Copy these commands to your Colab notebook for maximum performance

## 1. SINGLE MODEL TRAINING (Quick Start - 75+ MCC Target)

### EfficientUNet + Tversky Loss (Recommended for imbalanced data)
```bash
python train_model.py \
  --data_dir "/content/Train" \
  --model_type efficientunet \
  --loss tversky \
  --tversky_alpha 0.7 \
  --tversky_beta 0.3 \
  --learning_rate 1e-3 \
  --weight_decay 1e-4 \
  --batch_size 4 \
  --epochs 80 \
  --optimizer adam \
  --scheduler plateau \
  --amp \
  --grad_clip 1.0 \
  --global_stats \
  --threshold_sweep \
  --model_save_path "./models/efficientunet_tversky"
```

### EfficientUNet + Boundary Loss (For edge-aware segmentation)
```bash
python train_model.py \
  --data_dir "/content/Train" \
  --model_type efficientunet \
  --loss boundary \
  --learning_rate 8e-4 \
  --weight_decay 2e-4 \
  --batch_size 4 \
  --epochs 80 \
  --optimizer adam \
  --scheduler cosine \
  --amp \
  --grad_clip 1.0 \
  --global_stats \
  --threshold_sweep \
  --model_save_path "./models/efficientunet_boundary"
```

### UNet + Adaptive Loss (Multi-loss learning)
```bash
python train_model.py \
  --data_dir "/content/Train" \
  --model_type unet \
  --loss adaptive \
  --learning_rate 5e-4 \
  --weight_decay 1e-4 \
  --batch_size 6 \
  --epochs 80 \
  --optimizer adam \
  --scheduler plateau \
  --amp \
  --grad_clip 1.0 \
  --global_stats \
  --threshold_sweep \
  --model_save_path "./models/unet_adaptive"
```

## 2. MULTI-SCALE TRAINING (Advanced - 78+ MCC Target)

### Multi-scale EfficientUNet (Best for varied scales)
```bash
python train_multiscale.py \
  --data_dir "/content/Train" \
  --model_type efficientunet \
  --loss tversky \
  --tversky_alpha 0.7 \
  --tversky_beta 0.3 \
  --learning_rate 1e-3 \
  --weight_decay 1e-4 \
  --batch_size 3 \
  --epochs 100 \
  --optimizer adam \
  --scheduler plateau \
  --amp \
  --grad_clip 1.0 \
  --early_stopping_patience 15 \
  --model_save_path "./models/multiscale_efficientunet"
```

## 3. ENSEMBLE TRAINING (Maximum Performance - 80+ MCC Target)

### Full ensemble with 3 models
```bash
python train_ensemble.py \
  --data_dir "/content/Train" \
  --batch_size 4 \
  --num_models 3 \
  --val_split 0.2 \
  --num_workers 2 \
  --model_save_path "./models/ensemble_full"
```

### Quick ensemble with 2 models (faster)
```bash
python train_ensemble.py \
  --data_dir "/content/Train" \
  --batch_size 4 \
  --num_models 2 \
  --val_split 0.2 \
  --num_workers 2 \
  --model_save_path "./models/ensemble_quick"
```

## 4. EVALUATION COMMANDS

### Test Time Augmentation evaluation
```bash
python evaluate_model.py \
  --data_dir "/content/Train" \
  --model_path "./models/efficientunet_tversky/efficientunet_best.pth" \
  --model_type efficientunet \
  --tta \
  --batch_size 4
```

### Threshold sweep for optimal cutoff
```bash
python evaluate_model.py \
  --data_dir "/content/Train" \
  --model_path "./models/efficientunet_tversky/efficientunet_best.pth" \
  --model_type efficientunet \
  --threshold_sweep \
  --batch_size 4
```

## 5. COLAB SETUP (Run first in your Colab)

```python
# Install dependencies
!pip install tqdm scikit-learn matplotlib pillow

# Git pull latest changes
!cd /content && git pull origin main

# Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Set data path
import os
if os.path.exists('/content/Train'):
    print("Training data found!")
else:
    print("Please upload your Train folder to /content/Train")
```

## 6. PERFORMANCE OPTIMIZATION TIPS

### Memory Management
- Use `batch_size=3-4` for most models
- Use `batch_size=6` only for UNet (smaller model)
- Enable `--amp` for faster training and less memory usage

### Training Speed
- Use `num_workers=2` in Colab (more can cause issues)
- Enable gradient clipping with `--grad_clip 1.0`
- Use global normalization with `--global_stats`

### Best Practices
1. **Start with EfficientUNet + Tversky**: Best balance of performance and speed
2. **Use threshold sweeping**: Find optimal prediction threshold
3. **Enable AMP**: Faster training with mixed precision
4. **Monitor validation MCC**: Target 0.75+ for single models
5. **Try ensemble if time permits**: Can boost MCC by 2-3%

## 7. EXPECTED PERFORMANCE

| Method | Expected MCC | Training Time | Memory Usage |
|--------|--------------|---------------|--------------|
| EfficientUNet + Tversky | 0.75-0.78 | ~2-3 hours | ~6GB |
| Multi-scale EfficientUNet | 0.78-0.80 | ~3-4 hours | ~8GB |
| 3-Model Ensemble | 0.80-0.82 | ~6-8 hours | ~12GB |

## 8. TROUBLESHOOTING

### Out of Memory (OOM)
- Reduce batch_size to 2 or 3
- Use fewer num_workers (try 1)
- Restart Colab runtime

### Slow Training
- Ensure GPU is enabled in Colab
- Use --amp flag for mixed precision
- Reduce num_workers if data loading is slow

### Low Performance
- Check data path is correct
- Ensure global_stats is enabled
- Try different loss functions (tversky, boundary, adaptive)
- Use threshold sweeping to find optimal cutoff

## 9. SUBMISSION PREPARATION

After training, your best model will be saved. Use the solution.py script for inference:

```python
# Ensure solution.py uses your best model
# It will automatically detect and load the best available model
python solution.py  # This generates predictions for submission
```

The solution.py script includes TTA and will use the best model from your training runs.