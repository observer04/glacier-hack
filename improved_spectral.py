# ==================================================================================
# IMPROVED_SPECTRAL.PY - Optimized Glacier Segmentation with Spectral Engineering
#
# This script improves on baseline by adding:
# - Spectral indices: NDSI, NDWI, NDVI (8 input channels total)
# - Focal Loss + Dice Loss (better for class imbalance)
# - GPU optimization: batch_size=16, num_workers=4
# - Heavier augmentation (brightness, contrast for satellite data)
# - Same ResNet18 U-Net architecture
# - 5-fold Cross-Validation
#
# BASELINE: 0.585 MCC
# TARGET: 0.63-0.67 MCC
# TRAINING TIME: ~45 minutes on Kaggle 2xT4
# ==================================================================================

# pip install torch torchvision numpy pillow scikit-learn tifffile tqdm

import os
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from torch.amp import autocast, GradScaler
    USE_NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    USE_NEW_AMP = False
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision.transforms import v2 as T
from torchvision.models import resnet18, ResNet18_Weights
from scipy.ndimage import label as scipy_label

# ==================================================================================
# 1. MODEL ARCHITECTURE - ResNet18 U-Net (8 input channels)
# ==================================================================================

class ResNet18UNetSpectral(nn.Module):
    """U-Net with ResNet18 encoder, modified for 8 input channels (5 bands + 3 indices)"""
    
    def __init__(self, n_classes=1, n_channels=8):
        super().__init__()
        
        # Load pretrained ResNet18
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify first conv layer for 8 input channels
        self.encoder_input = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize weights: average pretrained RGB weights for all channels
        with torch.no_grad():
            pretrained_weight = resnet.conv1.weight.data
            avg_weight = pretrained_weight.mean(dim=1, keepdim=True)
            self.encoder_input.weight.data = avg_weight.repeat(1, n_channels, 1, 1) / n_channels
        
        # Encoder layers
        self.encoder_bn = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool
        
        self.encoder1 = resnet.layer1  # 64 channels
        self.encoder2 = resnet.layer2  # 128 channels
        self.encoder3 = resnet.layer3  # 256 channels
        self.encoder4 = resnet.layer4  # 512 channels
        
        # Decoder (upsampling path)
        self.decoder4 = self._make_decoder_block(512, 256)
        self.decoder3 = self._make_decoder_block(256 + 256, 128)
        self.decoder2 = self._make_decoder_block(128 + 128, 64)
        self.decoder1 = self._make_decoder_block(64 + 64, 64)
        
        # Final upsampling to original resolution
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Output head
        self.output = nn.Conv2d(32, n_classes, kernel_size=1)
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        x = self.encoder_input(x)
        x = self.encoder_bn(x)
        x = self.encoder_relu(x)
        x0 = x
        
        x = self.encoder_maxpool(x)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        
        # Decoder with skip connections
        d4 = self.decoder4(x4)
        d3 = self.decoder3(torch.cat([d4, x3], dim=1))
        d2 = self.decoder2(torch.cat([d3, x2], dim=1))
        d1 = self.decoder1(torch.cat([d2, x1], dim=1))
        
        # Final upsampling
        out = self.final_upsample(d1)
        out = self.output(out)
        
        return out

# ==================================================================================
# 2. SPECTRAL INDEX COMPUTATION
# ==================================================================================

def compute_spectral_indices(bands):
    """
    Compute spectral indices from 5 Sentinel-2 bands.
    
    Args:
        bands: numpy array (5, H, W) - [Blue, Green, Red, SWIR, TIR]
    
    Returns:
        indices: numpy array (3, H, W) - [NDSI, NDWI, NDVI]
    """
    blue = bands[0]   # Band1 - B2
    green = bands[1]  # Band2 - B3
    red = bands[2]    # Band3 - B4
    swir = bands[3]   # Band4 - B6
    tir = bands[4]    # Band5 - B10
    
    eps = 1e-8  # Avoid division by zero
    
    # NDSI = (Green - SWIR) / (Green + SWIR)
    # Standard glacier/snow detection index
    ndsi = (green - swir) / (green + swir + eps)
    ndsi = np.clip(ndsi, -1, 1)
    
    # NDWI = (Green - Red) / (Green + Red)
    # Water detection index (helps separate meltwater)
    ndwi = (green - red) / (green + red + eps)
    ndwi = np.clip(ndwi, -1, 1)
    
    # NDVI = (Red - SWIR) / (Red + SWIR)
    # Vegetation index (helps exclude vegetated areas)
    ndvi = (red - swir) / (red + swir + eps)
    ndvi = np.clip(ndvi, -1, 1)
    
    indices = np.stack([ndsi, ndwi, ndvi], axis=0).astype(np.float32)
    return indices

# ==================================================================================
# 3. DATASET & DATA LOADING
# ==================================================================================

def get_tile_id(filename):
    """Extract tile ID from filename (e.g., '02_07')"""
    match = re.search(r'(\d{2}_\d{2})', filename)
    return match.group(1) if match else None

class GlacierSpectralDataset(Dataset):
    """Dataset for glacier segmentation with spectral indices"""
    
    def __init__(self, data_dir, tile_ids, stats=None, augment=False):
        self.data_dir = data_dir
        self.tile_ids = tile_ids
        self.stats = stats
        self.augment = augment
        
        # Band prefixes (Sentinel-2 bands)
        self.band_info = [
            ("Band1", "B2"),   # Blue
            ("Band2", "B3"),   # Green
            ("Band3", "B4"),   # Red
            ("Band4", "B6"),   # SWIR
            ("Band5", "B10")   # TIR
        ]
        
        # Augmentation transforms
        if augment:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomApply([T.RandomRotation(degrees=(-90, 90))], p=0.5),
                # Brightness/contrast for satellite imagery
                T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.3),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.tile_ids)
    
    def _load_band(self, band_folder, band_prefix, tile_id):
        """Load a single band image"""
        possible_names = [
            f"{band_prefix}_{band_prefix}_masked_{tile_id}.tif",
            f"stacked_{tile_id}.tif"
        ]
        
        for name in possible_names:
            path = os.path.join(self.data_dir, band_folder, name)
            if os.path.exists(path):
                img = np.array(Image.open(path), dtype=np.float32)
                if img.ndim == 3:
                    img = img[:, :, 0]
                return img
        
        raise FileNotFoundError(f"Band {band_folder} not found for tile {tile_id}")
    
    def _load_label(self, tile_id):
        """Load label mask"""
        possible_names = [
            f"Y_output_resized_{tile_id}.tif",
            f"Y{tile_id}.tif"
        ]
        
        for name in possible_names:
            path = os.path.join(self.data_dir, "label", name)
            if os.path.exists(path):
                label = np.array(Image.open(path))
                if label.ndim == 3:
                    label = label[:, :, 0]
                return (label > 127).astype(np.float32)
        
        raise FileNotFoundError(f"Label not found for tile {tile_id}")
    
    def __getitem__(self, idx):
        tile_id = self.tile_ids[idx]
        
        # Load all 5 bands
        bands = []
        for band_folder, band_prefix in self.band_info:
            band = self._load_band(band_folder, band_prefix, tile_id)
            bands.append(band)
        
        # Stack bands: (5, H, W)
        bands = np.stack(bands, axis=0).astype(np.float32)
        
        # Normalize bands using global statistics
        if self.stats:
            for i in range(5):
                mean = self.stats['band_means'][i]
                std = self.stats['band_stds'][i]
                if std > 0:
                    bands[i] = (bands[i] - mean) / std
        
        # Compute spectral indices (3, H, W)
        indices = compute_spectral_indices(bands)
        
        # Concatenate bands + indices: (8, H, W)
        image = np.concatenate([bands, indices], axis=0)
        
        # Load label
        label = self._load_label(tile_id)
        
        # Convert to tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label).unsqueeze(0)
        
        # Augmentation
        if self.transform:
            combined = torch.cat([image, label], dim=0)
            combined = self.transform(combined)
            image = combined[:8]
            label = combined[8:9]
        
        return image, label, tile_id

def compute_global_stats(data_dir, tile_ids):
    """Compute mean and std for each band"""
    print("Computing global statistics...")
    
    band_info = [
        ("Band1", "B2"), ("Band2", "B3"), ("Band3", "B4"),
        ("Band4", "B6"), ("Band5", "B10")
    ]
    
    sums = [0.0] * 5
    sums_sq = [0.0] * 5
    counts = [0] * 5
    
    for tile_id in tqdm(tile_ids, desc="Stats"):
        for i, (band_folder, band_prefix) in enumerate(band_info):
            possible_names = [
                f"{band_prefix}_{band_prefix}_masked_{tile_id}.tif",
                f"stacked_{tile_id}.tif"
            ]
            
            path = None
            for name in possible_names:
                test_path = os.path.join(data_dir, band_folder, name)
                if os.path.exists(test_path):
                    path = test_path
                    break
            
            if path:
                band = np.array(Image.open(path), dtype=np.float32)
                if band.ndim == 3:
                    band = band[:, :, 0]
                
                sums[i] += band.sum()
                sums_sq[i] += (band ** 2).sum()
                counts[i] += band.size
    
    means = [sums[i] / counts[i] for i in range(5)]
    stds = [np.sqrt(sums_sq[i] / counts[i] - means[i] ** 2) for i in range(5)]
    
    stats = {'band_means': means, 'band_stds': stds}
    print(f"Global stats computed: {stats}")
    return stats

# ==================================================================================
# 4. LOSS FUNCTION - Focal + Dice
# ==================================================================================

class FocalDiceLoss(nn.Module):
    """Combined Focal Loss and Dice Loss for class imbalance"""
    
    def __init__(self, focal_weight=0.5, dice_weight=0.5, focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
    
    def forward(self, pred, target):
        # Focal Loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pred_sigmoid = torch.sigmoid(pred)
        
        # Focal weight: (1 - pt)^gamma
        pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
        focal_weight = (1 - pt) ** self.focal_gamma
        
        # Alpha weighting for class imbalance
        alpha_t = torch.where(target == 1, self.focal_alpha, 1 - self.focal_alpha)
        
        focal_loss = (alpha_t * focal_weight * bce_loss).mean()
        
        # Dice Loss
        smooth = 1.0
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        dice_loss = 1.0 - dice_score
        
        # Combined loss
        total_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        return total_loss

# ==================================================================================
# 5. TRAINING & VALIDATION
# ==================================================================================

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for images, labels, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if USE_NEW_AMP:
            with autocast('cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            with autocast(dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, device, min_pred_size=100):
    """Validate and compute MCC"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(loader, desc="Validating", leave=False):
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            if USE_NEW_AMP:
                with autocast('cuda', dtype=torch.float16):
                    outputs = model(images)
            else:
                with autocast(dtype=torch.float16):
                    outputs = model(images)
            
            preds = torch.sigmoid(outputs) > 0.5
            preds = preds.cpu().numpy()
            
            # Post-processing: Remove small connected components
            for i in range(preds.shape[0]):
                pred_mask = preds[i, 0]
                labeled_array, num_features = scipy_label(pred_mask)
                
                for region_id in range(1, num_features + 1):
                    region_mask = labeled_array == region_id
                    if region_mask.sum() < min_pred_size:
                        pred_mask[region_mask] = 0
                
                preds[i, 0] = pred_mask
            
            all_preds.append(preds.flatten())
            all_labels.append(labels.flatten())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    mcc = matthews_corrcoef(all_labels, all_preds)
    return mcc

# ==================================================================================
# 6. MAIN TRAINING LOOP
# ==================================================================================

def main():
    # Configuration
    CONFIG = {
        'data_dir': '/kaggle/working/Train',
        'work_dir': '/kaggle/working',
        'batch_size': 16,  # Increased from 8 - better GPU utilization
        'epochs': 35,
        'n_folds': 5,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'early_stop_patience': 10,
        'min_pred_size': 100,
        'num_workers': 4  # Increased from 2 - faster data loading
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Config: batch_size={CONFIG['batch_size']}, num_workers={CONFIG['num_workers']}")
    
    # Get all tile IDs
    label_dir = os.path.join(CONFIG['data_dir'], 'label')
    all_files = [f for f in os.listdir(label_dir) if f.endswith('.tif')]
    all_tile_ids = sorted([get_tile_id(f) for f in all_files if get_tile_id(f)])
    
    print(f"\nTotal tiles: {len(all_tile_ids)}")
    
    # Compute global statistics
    stats_path = os.path.join(CONFIG['work_dir'], 'stats_spectral.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        print(f"Loaded stats from {stats_path}")
    else:
        stats = compute_global_stats(CONFIG['data_dir'], all_tile_ids)
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
    
    # K-Fold Cross-Validation
    kf = KFold(n_splits=CONFIG['n_folds'], shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_tile_ids)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1}/{CONFIG['n_folds']}")
        print(f"{'='*70}")
        
        train_ids = [all_tile_ids[i] for i in train_idx]
        val_ids = [all_tile_ids[i] for i in val_idx]
        
        # Create datasets
        train_dataset = GlacierSpectralDataset(CONFIG['data_dir'], train_ids, stats, augment=True)
        val_dataset = GlacierSpectralDataset(CONFIG['data_dir'], val_ids, stats, augment=False)
        
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                                 shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=1,
                               shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=True)
        
        # Model, optimizer, loss
        model = ResNet18UNetSpectral(n_classes=1, n_channels=8).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'],
                                     weight_decay=CONFIG['weight_decay'])
        criterion = FocalDiceLoss(focal_weight=0.5, dice_weight=0.5, focal_gamma=2.0, focal_alpha=0.25)
        
        if USE_NEW_AMP:
            scaler = GradScaler('cuda')
        else:
            scaler = GradScaler()
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=4
        )
        
        # Training loop
        best_mcc = -1.0
        patience_counter = 0
        
        for epoch in range(CONFIG['epochs']):
            print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
            
            # Validate
            val_mcc = validate(model, val_loader, device, CONFIG['min_pred_size'])
            
            print(f"Train Loss: {train_loss:.4f} | Val MCC: {val_mcc:.4f}")
            
            # Learning rate scheduling
            scheduler.step(val_mcc)
            
            # Save best model
            if val_mcc > best_mcc:
                best_mcc = val_mcc
                patience_counter = 0
                model_path = os.path.join(CONFIG['work_dir'], f'best_spectral_fold{fold+1}.pth')
                torch.save(model.state_dict(), model_path)
                print(f"✓ Saved best model (MCC: {best_mcc:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= CONFIG['early_stop_patience']:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        fold_results.append(best_mcc)
        print(f"\nFold {fold + 1} Best MCC: {best_mcc:.4f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SPECTRAL MODEL RESULTS SUMMARY")
    print(f"{'='*70}")
    for i, mcc in enumerate(fold_results):
        print(f"Fold {i + 1}: MCC = {mcc:.4f}")
    print(f"\nMean MCC: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print(f"{'='*70}")
    
    # Save results
    results = {
        'fold_results': fold_results,
        'mean_mcc': float(np.mean(fold_results)),
        'std_mcc': float(np.std(fold_results))
    }
    
    with open(os.path.join(CONFIG['work_dir'], 'spectral_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Training complete! Upload best model for submission.")

if __name__ == '__main__':
    main()
