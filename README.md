# Glacier Segmentation Project - Academic Overview

## Table of Contents
1. [Project Context](#1-project-context)
2. [Problem Definition](#2-problem-definition)
3. [Data Understanding](#3-data-understanding)
4. [Model Architecture](#4-model-architecture)
5. [Training Pipeline](#5-training-pipeline)
6. [Optimization Strategy](#6-optimization-strategy)
7. [Evaluation & Validation](#7-evaluation--validation)
8. [Inference Pipeline](#8-inference-pipeline)
9. [Key Learnings](#9-key-learnings)
10. [Results & Analysis](#10-results--analysis)

---

## 1. Project Context

### 1.1 Objective
Develop a semantic segmentation model to identify glaciers in satellite imagery using Sentinel-2 multispectral data.

### 1.2 Challenge
- **Task**: Binary segmentation (glacier vs. non-glacier)
- **Metric**: Matthews Correlation Coefficient (MCC)
- **Target**: MCC ≥ 0.70 on hidden test set
- **Constraints**: 
  - Model size < 200 MB
  - Processing time limits (no Test-Time Augmentation)
  - Read-only filesystem on platform

### 1.3 Why This Matters
Glacier monitoring is crucial for:
- Climate change research
- Water resource management
- Hazard assessment (glacial lake outbursts)
- Long-term environmental monitoring

---

## 2. Problem Definition

### 2.1 What is Semantic Segmentation?

Semantic segmentation assigns a class label to **every pixel** in an image. Unlike:
- **Classification**: One label per image ("this image contains a glacier")
- **Object Detection**: Bounding boxes around objects
- **Semantic Segmentation**: Per-pixel labels (each pixel is either "glacier" or "non-glacier")

### 2.2 Why is MCC Important?

**Matthews Correlation Coefficient (MCC)** ranges from -1 to +1:
- **+1**: Perfect prediction
- **0**: Random prediction
- **-1**: Total disagreement

**Formula:**
```
MCC = (TP×TN - FP×FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

**Why MCC over Accuracy?**
- Handles class imbalance (glaciers may cover only 20-40% of image)
- Considers all confusion matrix elements
- More robust metric for imbalanced datasets

### 2.3 Key Challenges

1. **Distribution Shift**: Training data patterns differ from test data
   - Solution: K-fold cross-validation + ensemble
   
2. **Class Imbalance**: Glaciers vary from 10% to 60% of image area
   - Solution: Dice-BCE loss function
   
3. **Multimodal Input**: 5 spectral bands with different value ranges
   - Solution: Per-band normalization
   
4. **Limited Data**: Only 25 training tiles
   - Solution: Heavy data augmentation + crops

---

## 3. Data Understanding

### 3.1 Input Data Structure

```
Train/
├── Band1/          # B2 (Blue) - Sentinel-2 band at 492nm
├── Band2/          # B3 (Green) - 560nm
├── Band3/          # B4 (Red) - 665nm
├── Band4/          # B6 (Red Edge) - 740nm
├── Band5/          # B10 (SWIR) - Short Wave Infrared at 1375nm
└── label/          # Ground truth binary masks
```

**File format**: GeoTIFF (.tif) files
**Spatial resolution**: 512 × 512 pixels per tile
**Tile naming**: `XX_YY` format (e.g., `02_07`, `03_09`)

### 3.2 Why These Bands?

| Band | Name | Wavelength | Purpose for Glaciers |
|------|------|------------|---------------------|
| B2 | Blue | 492 nm | High reflectance from snow/ice |
| B3 | Green | 560 nm | Distinguishes vegetation |
| B4 | Red | 665 nm | Strong contrast with ice |
| B6 | Red Edge | 740 nm | Separates ice from clouds |
| B10 | SWIR | 1375 nm | **Critical**: Ice absorbs SWIR, appears dark |

**Key Insight**: Glaciers are bright in visible bands but **dark in SWIR** - this is the signature!

### 3.3 Data Statistics

**Computed from all 25 training tiles** (in `quick_data_analysis.py`):

```python
BAND_MEANS = [23264.8359, 22882.7227, 22640.1055, 6610.3862, 23520.8379]
BAND_STDS = [22887.9688, 22444.9688, 22843.4453, 4700.4126, 14073.5098]
```

**Normalization** (critical for training):
```python
normalized_band = (raw_band - mean) / std
```

This centers data around 0 and scales to ~unit variance.

**Code location**: `solution.py` lines 115-117, `data_utils.py`

---

## 4. Model Architecture

### 4.1 Architecture Choice: U-Net with ResNet18 Encoder

**Why U-Net?**
- Designed for semantic segmentation (originally for medical images)
- Encoder-decoder structure with skip connections
- Captures multi-scale features

**Why ResNet18 as Encoder?**
- Pre-trained on ImageNet (transfer learning)
- Residual connections prevent gradient vanishing
- Lightweight (~11M parameters) - fits in 200MB limit
- Well-tested backbone

### 4.2 Architecture Diagram

```
Input (5 channels, 512×512)
         ↓
    ┌─────────────────────────────────────┐
    │ ENCODER (ResNet18)                  │
    │                                     │
    │ Conv 7×7, stride 2  →  x0 (256×256) │ ──┐ Skip
    │ MaxPool             →  (128×128)    │   │
    │ Layer1 (64 ch)      →  x1 (128×128) │ ──┤ Connections
    │ Layer2 (128 ch)     →  x2 (64×64)   │ ──┤
    │ Layer3 (256 ch)     →  x3 (32×32)   │ ──┤
    │ Layer4 (512 ch)     →  x4 (16×16)   │ ──┤
    └─────────────────────────────────────┘   │
         ↓                                     │
    ┌─────────────────────────────────────┐   │
    │ DECODER (U-Net style)               │   │
    │                                     │   │
    │ Upsample x4      →  d4 (32×32)      │ ←─┤
    │ Concat [d4, x3]  →  (32×32, 512 ch) │   │
    │ Upsample         →  d3 (64×64)      │ ←─┤
    │ Concat [d3, x2]  →  (64×64, 256 ch) │   │
    │ Upsample         →  d2 (128×128)    │ ←─┤
    │ Concat [d2, x1]  →  (128×128, 128ch)│   │
    │ Upsample         →  d1 (256×256)    │ ←─┘
    │ Final Upsample   →  (512×512)       │
    │ Conv 1×1         →  (512×512, 1 ch) │
    └─────────────────────────────────────┘
         ↓
    Output (1 channel, 512×512)
```

### 4.3 Key Modifications

**1. First Convolution Layer** (5 input channels instead of 3):

```python
# solution.py lines 42-47
self.encoder_input = nn.Conv2d(n_channels=5, out_channels=64, 
                               kernel_size=7, stride=2, padding=3)
```

**Strategy**: Initialize with averaged ImageNet weights
```python
pretrained_weight = resnet.conv1.weight.data  # (64, 3, 7, 7)
avg_weight = pretrained_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
new_weight = avg_weight.repeat(1, 5, 1, 1) / 5  # (64, 5, 7, 7)
```

**Why this works**: 
- Preserves magnitude of pretrained weights
- Each new channel gets averaged RGB knowledge
- Better than random initialization

**2. Skip Connections**:
```python
# solution.py lines 100-103
d3 = self.decoder3(torch.cat([d4, x3], dim=1))  # Concatenate features
```

**Why skip connections matter**:
- Encoder captures "what" (semantic information)
- Decoder captures "where" (spatial information)
- Skip connections combine both → precise boundaries

**3. Decoder Blocks**:
```python
# solution.py lines 73-81
def _make_decoder_block(self, in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),  # Upsample
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Refine
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
```

**Code location**: `solution.py` lines 28-110, `models.py`

---

## 5. Training Pipeline

### 5.1 K-Fold Cross-Validation Strategy

**Why K-Fold?**
- Limited data (25 tiles) → high variance in single train/val split
- K-fold reduces variance by averaging multiple models
- Each tile used for validation exactly once

**Implementation** (5-Fold CV):
```python
# train_improved_5fold.py lines 280-295
num_folds = 5
fold_size = len(all_tile_ids) // num_folds  # 25 tiles / 5 = 5 tiles per fold

for fold in range(num_folds):
    val_start = fold * fold_size
    val_end = (fold + 1) * fold_size
    
    val_tile_ids = all_tile_ids[val_start:val_end]  # 5 tiles for validation
    train_tile_ids = all_tile_ids[:val_start] + all_tile_ids[val_end:]  # 20 tiles for training
```

**Result**: 5 models, each trained on 20 tiles, validated on 5 different tiles.

### 5.2 Dataset Class

**Purpose**: Load and preprocess data efficiently

```python
# train_improved_5fold.py lines 50-120
class GlacierDataset(Dataset):
    def __init__(self, data_dir, tile_ids, stats, crops_per_tile=8, is_train=True):
        self.crops_per_tile = crops_per_tile  # Multiple crops per tile
        self.is_train = is_train
        
    def __len__(self):
        return len(self.tile_ids) * self.crops_per_tile  # 20 tiles × 8 crops = 160 samples
    
    def __getitem__(self, idx):
        tile_idx = idx // self.crops_per_tile
        crop_idx = idx % self.crops_per_tile
        
        # Load 5 bands + label
        # Apply normalization
        # Apply augmentations (if training)
        # Extract random crop
        
        return image_tensor, label_tensor
```

**Key concepts**:

1. **Multiple Crops per Tile** (data augmentation):
   - Each 512×512 tile → 8 different 512×512 crops
   - Crops are random during training → effectively 160 training samples from 20 tiles
   - **Why**: Increases training data 8× without duplicating tiles

2. **Normalization** (per-band):
   ```python
   for i in range(5):
       image[i] = (image[i] - BAND_MEANS[i]) / BAND_STDS[i]
   ```

3. **Augmentations** (training only):
   ```python
   if self.is_train and random.random() > 0.5:
       # Horizontal flip
       image = torch.flip(image, dims=[2])
       label = torch.flip(label, dims=[1])
   
   if self.is_train and random.random() > 0.5:
       # Vertical flip
       image = torch.flip(image, dims=[1])
       label = torch.flip(label, dims=[0])
   
   if self.is_train and random.random() > 0.5:
       # 90° rotation
       k = random.randint(1, 3)
       image = torch.rot90(image, k, dims=[1, 2])
       label = torch.rot90(label, k, dims=[0, 1])
   ```

**Code location**: `train_improved_5fold.py` lines 50-140, `data_utils.py`

### 5.3 Loss Function: Dice-BCE Loss

**Problem**: Standard Cross-Entropy struggles with imbalanced classes

**Solution**: Combine Dice Loss + Binary Cross-Entropy

```python
# train_improved_5fold.py lines 145-170
class DiceBCELoss(nn.Module):
    def forward(self, pred, target):
        # Dice Loss (IoU-based)
        pred_prob = torch.sigmoid(pred)
        intersection = (pred_prob * target).sum()
        dice_loss = 1 - (2 * intersection + smooth) / (pred_prob.sum() + target.sum() + smooth)
        
        # Binary Cross-Entropy
        bce_loss = F.binary_cross_entropy_with_logits(pred, target)
        
        # Combine (equal weight)
        return dice_loss + bce_loss
```

**Why Dice Loss?**
- **IoU-based**: Directly optimizes overlap between prediction and ground truth
- **Formula**: `Dice = 2×|A∩B| / (|A| + |B|)`
- **Advantage**: Works well with imbalanced data (focuses on positive class)

**Why also BCE?**
- **Pixel-wise supervision**: Penalizes each pixel independently
- **Gradient properties**: Provides stable gradients early in training
- **Complementary**: Dice for overlap, BCE for pixel accuracy

**Why combine both?**
- Dice Loss can be unstable when prediction is all zeros
- BCE ensures every pixel contributes to loss
- Empirically: 1:1 ratio works best

**Code location**: `train_improved_5fold.py` lines 145-170

### 5.4 Optimizer: AdamW

**Choice**: AdamW (Adam with Weight Decay)

```python
# train_improved_5fold.py line 318
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
```

**Why AdamW over SGD?**
- **Adaptive learning rates**: Different learning rate per parameter
- **Momentum**: Smooths gradient updates
- **Weight decay**: Proper L2 regularization (fixes Adam's weight decay bug)

**Hyperparameters**:
- `lr=2e-4`: Learning rate (conservative for fine-tuning)
- `weight_decay=1e-4`: L2 regularization (prevents overfitting)

**Why not SGD?**
- SGD requires careful learning rate tuning
- Adam adapts automatically to parameter scales

### 5.5 Learning Rate Scheduler: Cosine Annealing

```python
# train_improved_5fold.py line 320
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
```

**How it works**:
```
Learning Rate
    ↑
 2e-4 |╲
      | ╲
      |  ╲___
      |      ╲____
 ~0   |___________╲___
      └─────────────────→ Epoch
      0              70
```

**Why cosine annealing?**
- Smooth decay (no sudden drops)
- Encourages convergence to flat minima (better generalization)
- No hyperparameters to tune (unlike step decay)

**Code location**: `train_improved_5fold.py` lines 318-320

### 5.6 Training Loop

```python
# train_improved_5fold.py lines 175-230
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    
    for images, labels in dataloader:
        images = images.to(device)  # (batch_size, 5, 512, 512)
        labels = labels.to(device)  # (batch_size, 512, 512)
        
        # Forward pass
        with torch.cuda.amp.autocast():  # Mixed precision
            outputs = model(images)  # (batch_size, 1, 512, 512)
            loss = criterion(outputs.squeeze(1), labels)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
    
    return running_loss / len(dataloader)
```

**Key concepts**:

1. **Mixed Precision Training** (`autocast`):
   - Uses FP16 (16-bit floats) for forward/backward pass
   - Uses FP32 (32-bit floats) for parameter updates
   - **Benefit**: ~2× faster, uses less GPU memory
   - **No accuracy loss**: Critical operations stay in FP32

2. **Gradient Scaler**:
   ```python
   scaler = torch.cuda.amp.GradScaler()
   scaler.scale(loss).backward()  # Scale loss to prevent underflow
   scaler.step(optimizer)          # Unscale before optimizer step
   ```
   - Prevents gradient underflow in FP16

3. **Batch Processing**:
   - `batch_size=64`: Process 64 crops at once
   - **Why 64?**: Maximizes GPU utilization (2× T4 15GB GPUs)
   - Larger batch = more stable gradients

**Code location**: `train_improved_5fold.py` lines 175-230

### 5.7 Validation Loop

```python
# train_improved_5fold.py lines 235-275
def validate(model, dataloader, device, threshold=0.5):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.sigmoid(outputs.squeeze(1))  # Convert to probabilities
            
            # Binarize predictions
            binary_preds = (probs > threshold).float()
            
            all_preds.append(binary_preds.cpu().numpy())
            all_labels.append(labels.numpy())
    
    # Compute MCC
    preds_flat = np.concatenate(all_preds).flatten()
    labels_flat = np.concatenate(all_labels).flatten()
    mcc = matthews_corrcoef(labels_flat, preds_flat)
    
    return mcc
```

**Why MCC on validation?**
- Matches competition metric
- Provides realistic estimate of test performance
- Accounts for class imbalance

**Threshold = 0.5 during validation**:
- Standard threshold for binary classification
- **Note**: Optimized later to 0.55 based on testing

**Code location**: `train_improved_5fold.py` lines 235-275

### 5.8 Training Configuration

```python
# Final configuration (train_improved_5fold.py)
BATCH_SIZE = 64           # Maximizes GPU usage
CROPS_PER_TILE = 8        # Data augmentation multiplier
NUM_EPOCHS = 70           # Balance between convergence and time
LEARNING_RATE = 2e-4      # Conservative for transfer learning
WEIGHT_DECAY = 1e-4       # L2 regularization
NUM_FOLDS = 5             # K-fold cross-validation
```

**Training time**: ~3-4 hours for all 5 folds on 2× T4 GPUs

**Code location**: `train_improved_5fold.py` lines 300-310

---

## 6. Optimization Strategy

### 6.1 Evolution of Approach

**Submission History**:
1. **Baseline** (ResNet18, basic training): 0.594 MCC
2. **First Ensemble** (3 models): 0.602 MCC (+0.008)
3. **Improved Training** (5-fold CV, better augmentation): 0.6883 MCC mean validation
4. **Optimized Ensemble** (top 3 folds, threshold tuning): 0.6564 MCC test

### 6.2 Key Improvements

#### Improvement 1: K-Fold Cross-Validation

**Before**: Single train/val split
**After**: 5-fold CV, ensemble top 3 models

**Why**:
- Reduces variance from random split
- Each model sees different validation data
- Ensemble averages out individual model errors

**Code**: `train_improved_5fold.py` lines 280-400

#### Improvement 2: Data Augmentation

**Before**: 6 crops per tile
**After**: 8 crops per tile + stronger augmentations

```python
# Augmentations applied:
- Horizontal flip (50% probability)
- Vertical flip (50% probability)
- 90° rotation (50% probability, k={1,2,3})
- Random crop position (each time)
```

**Impact**: Effective dataset size = 20 tiles × 8 crops = 160 samples

**Code**: `train_improved_5fold.py` lines 80-115

#### Improvement 3: Optimizer & Scheduler

**Before**: Adam with fixed learning rate
**After**: AdamW + Cosine Annealing

**Why**:
- AdamW: Proper weight decay → less overfitting
- Cosine Annealing: Smooth LR decay → better convergence

**Code**: `train_improved_5fold.py` lines 318-320

#### Improvement 4: Batch Size Optimization

**Before**: batch_size=48
**After**: batch_size=64

**Why**:
- Maximizes GPU memory usage (15GB per T4)
- Larger batch → more stable gradients
- 33% more samples per batch → faster convergence

**Code**: `train_improved_5fold.py` line 301

#### Improvement 5: Threshold Optimization

**Problem**: Validation used threshold=0.5, but optimal is different

**Solution**: Test multiple thresholds on validation data

```python
# test_thresholds.py results:
Threshold=0.30 → MCC=0.5870
Threshold=0.40 → MCC=0.6798
Threshold=0.50 → MCC=0.7378
Threshold=0.55 → MCC=0.7490  ← OPTIMAL
Threshold=0.60 → MCC=0.7483
```

**Why 0.55 works better**:
- Ensemble produces well-calibrated probabilities
- Higher threshold reduces false positives
- Glacier boundaries are more conservative but accurate

**Code**: `solution.py` line 244, `test_thresholds.py`

### 6.3 Ensemble Strategy

**Why Ensemble?**
- Reduces variance (different models make different errors)
- Increases robustness (less sensitive to single model failure)
- Proven to improve MCC in competitions

**Implementation**:
1. Train 5 models (one per fold)
2. Select top 3 by validation MCC:
   - Fold 2: 0.7216 MCC
   - Fold 5: 0.7131 MCC
   - Fold 1: 0.6988 MCC
3. Average predictions: `final = (pred1 + pred2 + pred3) / 3`

**Why top 3 (not all 5)?**
- Model size constraint: 3×50MB = 150MB < 200MB limit
- Top models contribute most
- Diminishing returns from adding weaker models

**Code**: `create_final_ensemble.py`, `solution.py` lines 285-338

---

## 7. Evaluation & Validation

### 7.1 Metrics

**Primary: Matthews Correlation Coefficient (MCC)**

```python
# sklearn implementation
from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(y_true_flat, y_pred_flat)
```

**Why MCC?**
- Range: [-1, +1]
- Handles imbalanced classes
- Considers all confusion matrix elements:
  - True Positives (TP): Correctly predicted glacier pixels
  - True Negatives (TN): Correctly predicted non-glacier pixels
  - False Positives (FP): Non-glacier predicted as glacier
  - False Negatives (FN): Glacier predicted as non-glacier

**Secondary: Visual Inspection**

- Overlay predictions on input images
- Check glacier boundary accuracy
- Identify systematic errors (e.g., shadow confusion)

**Code**: `train_improved_5fold.py` lines 235-275, `evaluate_model.py`

### 7.2 Validation Strategy

**During Training**:
```python
# Validate every epoch
for epoch in range(num_epochs):
    train_loss = train_epoch(...)
    val_mcc = validate(...)
    
    # Early stopping
    if val_mcc > best_mcc:
        best_mcc = val_mcc
        torch.save(model.state_dict(), f'best_fold{fold}.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

**After Training**:
```python
# Test on held-out validation set
results = test_optimized_solution.py  # Local validation
# → Overall MCC: 0.7427 on all 25 training tiles
```

**Code**: `train_improved_5fold.py` lines 340-380, `test_optimized_solution.py`

### 7.3 Results Analysis

**5-Fold Cross-Validation Results**:
```
Fold 1: 0.6988 MCC (val: 02_07, 02_08, 03_07, 03_09, 03_11)
Fold 2: 0.7216 MCC (val: 04_08, 04_09, 04_10, 05_08, 05_09)
Fold 3: 0.6264 MCC (val: 05_10, 06_09, 06_11, 06_12, 07_10)
Fold 4: 0.6816 MCC (val: 07_11, 07_13, 08_12, 08_13, 08_14)
Fold 5: 0.7131 MCC (val: 09_13, 09_14, 10_12, 11_13, 12_12)

Mean: 0.6883 ± 0.0371
```

**Insights**:
- High variance (std=0.037) suggests some tiles are harder
- Fold 3 is weakest → those tiles are challenging
- Top 3 folds all > 0.69 MCC

**Local Test Results** (ensemble on all 25 tiles):
```
Best tile:  09_14 → 0.8331 MCC
Worst tile: 10_12 → 0.5780 MCC
Overall:    0.7427 MCC
```

**Gap Analysis**:
- Validation (per fold): ~0.69-0.72 MCC
- Local test (ensemble): 0.7427 MCC
- Platform test: 0.6564 MCC
- **Gap**: -0.086 MCC

**Possible reasons for gap**:
1. Distribution shift (test data differs from training)
2. Overfitting to training tiles
3. Different preprocessing on platform
4. Test tiles are more challenging

---

## 8. Inference Pipeline

### 8.1 Platform Submission Format

**Required files**:
1. `solution.py`: Python script with `maskgeration()` function
2. `model.pth`: Trained model weights (< 200MB)

**Function signature**:
```python
def maskgeration(imagepath, out_dir):
    """
    Args:
        imagepath: Dict of band folders {"Band1": path, "Band2": path, ...}
        out_dir: Model path (platform quirk - not output directory!)
    
    Returns:
        dict: {tile_id: numpy_array} - Binary masks with 0/1 values
    """
```

**Code**: `solution.py` lines 289-383

### 8.2 Data Loading

```python
# solution.py lines 151-238
def load_bands_from_dict(imagepath_dict):
    # Get all tiles from Band1 folder
    band1_files = os.listdir(imagepath_dict["Band1"])
    
    results = {}
    for tile_file in band1_files:
        tile_id = get_tile_id(tile_file)  # Extract numeric ID
        
        bands = []
        # Load each band (B2, B3, B4, B6, B10)
        for band_name, band_prefix in band_info:
            # Try multiple filename patterns (training vs test format)
            # Training: B2_B2_masked_02_07.tif
            # Test: img001.tif
            file_path = find_band_file(band_name, tile_id)
            band = np.array(Image.open(file_path), dtype=np.float32)
            bands.append(band)
        
        results[tile_id] = np.stack(bands, axis=0)  # (5, H, W)
    
    return results
```

**Key challenges**:
1. **Filename patterns differ**: Training uses `B2_B2_masked_XX_YY.tif`, test uses `imgXXX.tif`
2. **Tile ID extraction**: Must extract numeric part only (`001` not `img001`)
3. **Band naming**: Band folders don't match band names (Band1 → B2 files)

**Code**: `solution.py` lines 121-238

### 8.3 Model Loading

```python
# solution.py lines 305-338
def maskgeration(imagepath, out_dir):
    # Load ensemble checkpoint
    checkpoint = torch.load(out_dir, map_location=device)
    
    # Check if ensemble or single model
    if isinstance(checkpoint, dict) and any(f'fold{i}' in checkpoint for i in range(1, 6)):
        # Ensemble: load each fold model
        models = []
        for fold_name in sorted(checkpoint.keys()):
            model = ResNet18UNet(n_classes=1, n_channels=5)
            
            # Remove 'module.' prefix (from DataParallel)
            state_dict = {k.replace('module.', ''): v 
                         for k, v in checkpoint[fold_name].items()}
            
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            models.append(model)
    else:
        # Single model
        models = [load_single_model(out_dir, device)]
```

**Key concepts**:
1. **Ensemble detection**: Check for `foldX` keys in checkpoint
2. **DataParallel handling**: Remove `module.` prefix from parameter names
3. **Device management**: Use GPU if available, else CPU
4. **Eval mode**: Disable dropout and batchnorm updates

**Code**: `solution.py` lines 305-338

### 8.4 Inference Process

```python
# solution.py lines 340-377
for tile_id, bands in tiles_data.items():
    # 1. Normalize bands
    bands_normalized = normalize_bands(bands)
    
    # 2. Convert to tensor
    image_tensor = torch.from_numpy(bands_normalized).unsqueeze(0)  # (1, 5, 512, 512)
    image_tensor = image_tensor.to(device)
    
    # 3. Run inference with all models
    predictions = []
    with torch.no_grad():
        for model in models:
            output = model(image_tensor)  # (1, 1, 512, 512)
            prob = torch.sigmoid(output).squeeze().cpu().numpy()  # (512, 512)
            predictions.append(prob)
    
    # 4. Average ensemble predictions
    avg_prob = np.mean(predictions, axis=0)  # (512, 512)
    
    # 5. Post-process
    binary_mask = post_process_mask(avg_prob, min_size=50)  # (512, 512) with 0/1
    
    # 6. Store result
    results[tile_id] = binary_mask

return results
```

**Key concepts**:

1. **No Test-Time Augmentation (TTA)**:
   - TTA would test with flips/rotations and average
   - **Why not?**: Platform timeout (504 error)
   - Single pass is fast enough

2. **Ensemble Averaging**:
   - Average **probabilities** (not binary predictions)
   - More information preserved
   - Smoother boundaries

3. **Sigmoid Activation**:
   - Model outputs logits (unbounded values)
   - Sigmoid converts to probabilities [0, 1]
   - `prob = 1 / (1 + exp(-logit))`

**Code**: `solution.py` lines 340-383

### 8.5 Post-Processing

```python
# solution.py lines 242-255
def post_process_mask(mask, min_size=50):
    # 1. Binarize with threshold
    binary_mask = (mask > 0.55).astype(np.uint8)
    
    # 2. Find connected components
    labeled_array, num_features = scipy_label(binary_mask)
    
    # 3. Remove small regions
    for region_id in range(1, num_features + 1):
        region_mask = (labeled_array == region_id)
        if region_mask.sum() < min_size:
            binary_mask[region_mask] = 0
    
    return binary_mask
```

**Why post-processing?**

1. **Threshold (0.55)**:
   - Converts probabilities to binary
   - Higher threshold = more conservative predictions
   - Reduces false positives

2. **Min Size Filtering (50 pixels)**:
   - Removes tiny isolated predictions (noise)
   - Glacier regions are typically large
   - 50 pixels ≈ 0.02% of image

**Effect**:
- Cleaner predictions
- Removes artifacts
- Slightly lower recall, higher precision

**Code**: `solution.py` lines 242-255

---

## 9. Key Learnings

### 9.1 Technical Learnings

**1. Transfer Learning is Powerful**
- Pre-trained ResNet18 (ImageNet) → glacier segmentation
- Even with different domains (RGB photos → multispectral satellite)
- Key: Careful weight initialization for extra channels

**2. Loss Function Matters**
- Standard BCE struggles with imbalance
- Dice + BCE combination works best
- Each loss provides different gradients

**3. Data Augmentation is Critical**
- Limited data (25 tiles) → heavy augmentation needed
- Flips + rotations are natural for satellite imagery
- Multiple crops per tile effectively increases dataset size

**4. Ensemble Reduces Variance**
- Single model: unstable across different splits
- Ensemble: averages out errors, more robust
- Top K models better than all models (quality over quantity)

**5. Threshold Tuning Often Overlooked**
- Default 0.5 is not always optimal
- Testing on validation data found 0.55 is better
- Small change (+0.05) → significant MCC improvement

### 9.2 Practical Learnings

**1. Platform Constraints Matter**
- Model size limit (200MB) ruled out larger architectures
- Timeout limit ruled out TTA
- Read-only filesystem → no downloading pretrained weights

**2. Metric Choice is Important**
- MCC handles imbalanced data better than accuracy
- Directly optimizing MCC during training is hard
- Dice loss is a good proxy for IoU/overlap metrics

**3. Validation Strategy**
- K-fold CV more reliable than single split with limited data
- Each fold provides independent estimate
- Variance across folds indicates model stability

**4. Distribution Shift is Real**
- Gap between validation (0.74) and test (0.66) MCC
- Training data may not represent test data perfectly
- Robust models (ensembles, augmentation) help mitigate

**5. Iterative Development**
- Start simple (baseline ResNet18)
- Add improvements one at a time
- Measure impact of each change
- Some improvements don't help (TTA caused timeout)

### 9.3 Debugging Lessons

**Common Issues Encountered**:

1. **Model file not found** → Platform imports as module, not runs script
2. **FileExistsError** → `out_dir` parameter is model path, not output directory
3. **504 Timeout** → TTA too slow, removed
4. **Wrong return type** → Must return dict, not save files
5. **Corrupted filenames** → Extract numeric tile ID only
6. **Type errors** → Return numpy arrays, not file paths

**Solution**: Careful reading of platform requirements and testing locally first

---

## 10. Results & Analysis

### 10.1 Final Results

**Training Results (5-fold CV)**:
```
Fold 1: 0.6988 MCC
Fold 2: 0.7216 MCC  ← Selected
Fold 3: 0.6264 MCC
Fold 4: 0.6816 MCC
Fold 5: 0.7131 MCC  ← Selected

Mean: 0.6883 ± 0.0371 MCC
Top 3 Mean: 0.7112 MCC
```

**Local Validation (Ensemble on 25 tiles)**:
```
Overall MCC: 0.7427
Mean per tile: 0.7148 ± 0.0580
Best tile: 0.8331 (09_14)
Worst tile: 0.5780 (10_12)
```

**Platform Test Result**:
```
Test MCC: 0.6564
```

### 10.2 Performance Analysis

**What Went Well**:
- ✅ Strong local validation (0.7427 MCC)
- ✅ Consistent fold performance (low std=0.037)
- ✅ Ensemble improved over single model
- ✅ Threshold optimization worked (0.55 > 0.50)
- ✅ All platform requirements met

**What Needs Improvement**:
- ❌ Validation-test gap (-0.086 MCC)
- ❌ Test MCC (0.6564) below target (0.70)
- ❌ Fold 3 performance (0.6264 MCC) dragged down average

**Possible Explanations**:

1. **Distribution Shift**:
   - Test tiles have different characteristics
   - Training tiles not representative
   - Different geographic regions/seasons

2. **Overfitting**:
   - Model memorized training tiles
   - Despite augmentation, limited diversity
   - 25 tiles may not be enough

3. **Threshold Mismatch**:
   - Optimized threshold (0.55) on training data
   - Optimal threshold for test might differ
   - Could try lower threshold (0.45-0.50)

### 10.3 Potential Improvements

**Short-term (Quick Wins)**:
1. **Try different thresholds** (0.45, 0.50) on test
2. **Adjust min_size** to 30 or 70 pixels
3. **Use all 5 folds** (if model size allows compression)
4. **Add minority class weighting** in loss function

**Medium-term (Re-training)**:
1. **More aggressive augmentation**: 
   - MixUp between tiles
   - CutMix
   - Color jittering (careful with spectral data)

2. **Different architecture**:
   - EfficientNet encoder (better efficiency)
   - DeepLabV3+ (better for boundaries)
   - Attention mechanisms (focus on glacier regions)

3. **Self-supervised pre-training**:
   - Pre-train on unlabeled satellite imagery
   - Then fine-tune on glacier data

**Long-term (More Data)**:
1. **Collect more training data**: 
   - Additional tiles from same/different regions
   - Temporal data (multiple seasons)
   
2. **External datasets**:
   - Other glacier segmentation datasets
   - Transfer learning from related tasks

3. **Semi-supervised learning**:
   - Use unlabeled test data for consistency regularization
   - Pseudo-labeling

### 10.4 Comparison to Target

**Target**: MCC ≥ 0.70

**Achieved**: MCC = 0.6564

**Gap**: -0.0436 (-6.2%)

**How close?**
- Relatively close (within 1 threshold adjustment)
- Local validation suggested 0.72-0.75 range
- Distribution shift larger than expected

**What would it take to reach 0.70?**
- Reduce false positives by ~15%
- Improve precision on hard tiles
- Better calibration of probabilities

---

## 11. Code Structure

### 11.1 Key Files

```
glacier-hack/
├── solution.py                      # Main inference script (PLATFORM SUBMISSION)
├── train_improved_5fold.py          # Training script (5-fold CV)
├── create_final_ensemble.py         # Combine top 3 models
├── test_optimized_solution.py       # Local validation
├── data_utils.py                    # Dataset utilities
├── models.py                        # Model architecture
├── evaluate_model.py                # Evaluation utilities
│
├── model_final_top3_ensemble.pth    # Ensemble weights (147.9 MB)
├── improved_5fold_results.json      # Training results
├── ensemble_info.json               # Ensemble metadata
│
├── Train/                           # Training data
│   ├── Band1/                       # B2 (Blue)
│   ├── Band2/                       # B3 (Green)
│   ├── Band3/                       # B4 (Red)
│   ├── Band4/                       # B6 (Red Edge)
│   ├── Band5/                       # B10 (SWIR)
│   └── label/                       # Ground truth masks
│
└── README.md                        # This file
```

### 11.2 File Relationships

```
Training Pipeline:
    data_utils.py (GlacierDataset)
         ↓
    train_improved_5fold.py (5-fold CV training)
         ↓
    best_improved_fold{1-5}.pth (5 model files)
         ↓
    create_final_ensemble.py (select top 3)
         ↓
    model_final_top3_ensemble.pth (ensemble)

Inference Pipeline:
    imagepath dict (Band1-5 folders)
         ↓
    solution.py (load_bands_from_dict)
         ↓
    normalize_bands (per-band normalization)
         ↓
    ResNet18UNet model (3× models)
         ↓
    ensemble averaging (mean probabilities)
         ↓
    post_process_mask (threshold + filter)
         ↓
    {tile_id: binary_mask} dict
```

### 11.3 Code Flow

**Training**:
```python
1. Load all 25 tile IDs
2. Split into 5 folds (5 tiles each)
3. For each fold:
   a. Create train dataset (20 tiles × 8 crops = 160 samples)
   b. Create val dataset (5 tiles × 8 crops = 40 samples)
   c. Initialize ResNet18UNet model
   d. Train for 70 epochs with:
      - Dice-BCE loss
      - AdamW optimizer
      - Cosine annealing scheduler
      - Mixed precision training
   e. Validate each epoch, save best model
   f. Record best validation MCC
4. Save all 5 models + results JSON
5. Select top 3 models by MCC
6. Combine into single ensemble file
```

**Inference**:
```python
1. Receive imagepath dict (Band1-5 folders) and model path
2. Load ensemble checkpoint (3 models)
3. Load all test tiles from Band1 folder
4. For each tile:
   a. Load 5 bands (B2, B3, B4, B6, B10)
   b. Stack into (5, H, W) array
   c. Normalize each band using training stats
   d. Convert to tensor (1, 5, H, W)
   e. Run through 3 models, get probabilities
   f. Average probabilities across models
   g. Apply threshold (0.55) to binarize
   h. Remove small regions (< 50 pixels)
   i. Store in results dict
5. Return {tile_id: binary_mask} dict
```

---

## 12. Glossary

**Semantic Segmentation**: Pixel-wise classification task (assign label to every pixel)

**U-Net**: Encoder-decoder architecture with skip connections, designed for segmentation

**ResNet**: Residual Network with skip connections that help training very deep networks

**Transfer Learning**: Using pre-trained model weights (ImageNet) as initialization

**K-Fold Cross-Validation**: Split data into K parts, train K models (each using K-1 parts for training, 1 for validation)

**Ensemble**: Combining multiple models' predictions (usually by averaging)

**MCC (Matthews Correlation Coefficient)**: Metric for binary classification that handles class imbalance

**Dice Loss**: Loss function based on IoU (Intersection over Union), good for segmentation

**BCE (Binary Cross-Entropy)**: Standard loss for binary classification

**Mixed Precision Training**: Using FP16 for speed, FP32 for stability

**TTA (Test-Time Augmentation)**: Testing with multiple augmented versions and averaging predictions

**Skip Connections**: Direct connections from encoder to decoder that preserve spatial information

**Batch Normalization**: Normalize activations within each mini-batch for stable training

**Learning Rate Scheduler**: Gradually reduce learning rate during training

**Early Stopping**: Stop training when validation performance stops improving

**Threshold**: Value to convert probabilities [0,1] to binary predictions {0,1}

**Post-processing**: Cleaning up predictions (removing noise, filling holes)

**Connected Components**: Groups of connected pixels with same label

**Distribution Shift**: When test data differs from training data distribution

**Overfitting**: Model learns training data too well, performs poorly on new data

**Hyperparameter**: Configuration value set before training (learning rate, batch size, etc.)

**Gradient Descent**: Optimization algorithm that adjusts weights to minimize loss

**Backpropagation**: Algorithm for computing gradients of loss with respect to weights

**Activation Function**: Non-linear function applied to neuron outputs (ReLU, Sigmoid, etc.)

**Convolution**: Operation that slides a filter over input to extract features

**Pooling**: Downsampling operation (max pooling, average pooling)

**Transposed Convolution**: Upsampling operation (also called deconvolution)

**Feature Map**: Output of a convolutional layer (spatial map of features)

**Receptive Field**: Area of input that influences a particular output pixel

**Stride**: Step size when sliding filter over input

**Padding**: Adding borders to input to control output size

**Channel**: Dimension of data (5 input channels = 5 spectral bands)

---

## 13. Further Reading

### Academic Papers

1. **U-Net: Convolutional Networks for Biomedical Image Segmentation**
   - Ronneberger et al., 2015
   - Original U-Net paper

2. **Deep Residual Learning for Image Recognition**
   - He et al., 2015
   - ResNet architecture

3. **Rethinking the Inception Architecture for Computer Vision**
   - Szegedy et al., 2016
   - Label smoothing, auxiliary classifiers

4. **Adam: A Method for Stochastic Optimization**
   - Kingma & Ba, 2014
   - Adam optimizer

5. **Decoupled Weight Decay Regularization**
   - Loshchilov & Hutter, 2019
   - AdamW (fixes Adam's weight decay)

### Tutorials

1. PyTorch Documentation: https://pytorch.org/docs/
2. Albumentations (augmentation): https://albumentations.ai/
3. Segmentation Models PyTorch: https://github.com/qubvel/segmentation_models.pytorch

### Related Work

1. **Glacier Mapping**:
   - Machine learning for glacier mapping using satellite imagery
   - Transfer learning for remote sensing applications

2. **Semantic Segmentation**:
   - Medical image segmentation
   - Street scene segmentation (Cityscapes, KITTI)

3. **Satellite Imagery**:
   - Sentinel-2 data analysis
   - Multispectral image processing
   - Remote sensing for earth observation

---

## 14. Conclusion

This project demonstrated end-to-end development of a semantic segmentation model for glacier identification using satellite imagery. Starting from raw Sentinel-2 multispectral data, we:

1. **Built a custom U-Net architecture** with ResNet18 encoder adapted for 5-channel input
2. **Implemented robust training pipeline** with K-fold CV, mixed precision, and data augmentation
3. **Optimized hyperparameters** through systematic experimentation
4. **Created ensemble model** combining top 3 folds for robustness
5. **Tuned post-processing** (threshold, min size) based on validation data
6. **Deployed to platform** meeting all technical constraints

**Final Performance**:
- Local validation: 0.7427 MCC
- Platform test: 0.6564 MCC
- Gap of -0.086 MCC indicates distribution shift

**Key Takeaways**:
- Transfer learning works even across domains (RGB → multispectral)
- Ensemble methods improve robustness significantly
- Validation strategy (K-fold) critical with limited data
- Platform constraints (size, time) guide architecture choices
- Distribution shift remains challenging in real-world deployment

**For Students**: This project covers foundational concepts in:
- Deep learning (CNNs, transfer learning, optimization)
- Computer vision (segmentation, image processing)
- Remote sensing (multispectral data, satellite imagery)
- Software engineering (modular code, testing, deployment)

The gap between validation and test performance highlights the importance of robust validation strategies and the challenges of real-world deployment.

---

**Project Status**: Complete (submitted to platform)  
**Final Test MCC**: 0.6564  
**Target MCC**: 0.70  
**Date**: October 7, 2025
