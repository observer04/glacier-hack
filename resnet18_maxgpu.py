# ==================================================================================
# RESNET18_MAXGPU.PY - MAXIMIZE GPU Utilization & Speed
#
# Strategy: Max GPU usage + eliminate I/O bottlenecks
# - LARGE batch size (48 per GPU = 96 total)
# - Optimized crop size (384) for speed
# - 8 data loading workers with prefetching
# - persistent_workers to avoid respawning
# - non_blocking GPU transfers
# - 6 crops per tile = 6x training data
# - MixUp augmentation
#
# TARGET: 0.63-0.68 MCC with FAST training
# GPU USAGE: 12-14GB per GPU (80-90% utilization)
# ==================================================================================

import os
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
try:
    from torch.amp.grad_scaler import GradScaler
except (ImportError, AttributeError):
    from torch.cuda.amp import GradScaler
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
from PIL import Image
from tqdm import tqdm
import numpy as np
from torchvision.transforms import v2 as T
from torchvision.models import resnet18, ResNet18_Weights
from scipy.ndimage import label as scipy_label

# ==================================================================================
# 1. MODEL ARCHITECTURE - ResNet18 U-Net
# ==================================================================================

class ResNet18UNet(nn.Module):
    """U-Net with ResNet18 encoder, modified for 5 input channels"""
    
    def __init__(self, n_classes=1, n_channels=5):
        super().__init__()
        
        # Load pretrained ResNet18
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify first conv layer for 5 input channels
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
        
        # Decoder
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
# 2. MIXUP AUGMENTATION
# ==================================================================================

def mixup_data(x, y, alpha=0.2):
    """Apply MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    
    return mixed_x, mixed_y

# ==================================================================================
# 3. DATASET WITH MULTI-SCALE CROPS
# ==================================================================================

def get_tile_id(filename):
    """Extract tile ID from filename (e.g., '02_07')"""
    match = re.search(r'(\d{2}_\d{2})', filename)
    return match.group(1) if match else None

class GlacierDataset(Dataset):
    """Dataset with fixed-size random crops"""
    
    def __init__(self, data_dir, tile_ids, stats=None, augment=False, crop_size=512, crops_per_tile=8):
        self.data_dir = data_dir
        self.tile_ids = tile_ids
        self.stats = stats
        self.augment = augment
        self.crop_size = crop_size  # Fixed size for batching
        self.crops_per_tile = crops_per_tile if augment else 1
        
        # Band prefixes (Sentinel-2 bands)
        self.band_info = [
            ("Band1", "B2"),   # Blue
            ("Band2", "B3"),   # Green
            ("Band3", "B4"),   # Red
            ("Band4", "B6"),   # SWIR
            ("Band5", "B10")   # TIR
        ]
        
        # Geometric augmentation
        if augment:
            self.transform = T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomApply([T.RandomRotation(degrees=(-90, 90))], p=0.5),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.tile_ids) * self.crops_per_tile
    
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
    
    def _random_crop(self, image, label, crop_size):
        """Extract random crop from full tile"""
        _, h, w = image.shape
        
        if h <= crop_size or w <= crop_size:
            # If tile smaller than crop, pad it
            pad_h = max(0, crop_size - h)
            pad_w = max(0, crop_size - w)
            image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
            label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = image.shape[1], image.shape[2]
        
        # Random crop
        top = np.random.randint(0, h - crop_size + 1)
        left = np.random.randint(0, w - crop_size + 1)
        
        image_crop = image[:, top:top+crop_size, left:left+crop_size]
        label_crop = label[top:top+crop_size, left:left+crop_size]
        
        return image_crop, label_crop
    
    def __getitem__(self, idx):
        # Map idx to tile (multiple crops per tile)
        tile_idx = idx // self.crops_per_tile
        tile_id = self.tile_ids[tile_idx]
        
        # Load all 5 bands
        bands = []
        for band_folder, band_prefix in self.band_info:
            band = self._load_band(band_folder, band_prefix, tile_id)
            bands.append(band)
        
        # Stack bands: (5, H, W)
        image = np.stack(bands, axis=0).astype(np.float32)
        
        # Normalize bands using global statistics
        if self.stats:
            # Handle both old and new stats format
            if 'band_means' in self.stats:
                means = self.stats['band_means']
                stds = self.stats['band_stds']
            elif 'means' in self.stats:
                means = self.stats['means']
                stds = self.stats['stds']
            else:
                # Legacy format: direct list
                means = self.stats.get('mean', [0]*5)
                stds = self.stats.get('std', [1]*5)
            
            for i in range(5):
                mean = means[i]
                std = stds[i]
                if std > 0:
                    image[i] = (image[i] - mean) / std
        
        # Load label
        label = self._load_label(tile_id)
        
        # Random crop (fixed size for batching)
        if self.augment:
            image, label = self._random_crop(image, label, self.crop_size)
        else:
            # Validation: use same crop size
            image, label = self._random_crop(image, label, self.crop_size)
        
        # Convert to tensors
        image = torch.from_numpy(image)
        label = torch.from_numpy(label).unsqueeze(0)
        
        # Geometric augmentation
        if self.transform:
            combined = torch.cat([image, label], dim=0)
            combined = self.transform(combined)
            image = combined[:5]
            label = combined[5:6]
        
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
    print(f"Global stats computed")
    return stats

# ==================================================================================
# 4. LOSS FUNCTION - Dice + BCE
# ==================================================================================

class DiceBCELoss(nn.Module):
    """Combined Dice Loss and BCE"""
    
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
    
    def forward(self, pred, target):
        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target)
        
        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice_score
        
        # Combined loss
        total_loss = self.dice_weight * dice_loss + self.bce_weight * bce_loss
        return total_loss

# ==================================================================================
# 5. TRAINING WITH GRADIENT ACCUMULATION
# ==================================================================================

def train_epoch(model, loader, criterion, optimizer, device, scaler, use_mixup=True, mixup_alpha=0.2):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for images, labels, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        
        # Apply MixUp
        if use_mixup and np.random.random() < 0.5:
            images, labels = mixup_data(images, labels, mixup_alpha)
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            images = images.to(device, non_blocking=True)
            labels = labels.cpu().numpy()
            
            with autocast():
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
# 6. MAIN TRAINING LOOP - MAX GPU UTILIZATION
# ==================================================================================

def main():
    # Configuration - MAXIMIZE GPU MEMORY & SPEED
    CONFIG = {
        'data_dir': '/kaggle/working/Train',
        'work_dir': '/kaggle/working',
        'batch_size': 48,  # INCREASED: 24→48 per GPU (use more memory)
        'epochs': 50,
        'n_folds': 5,
        'lr': 1e-4,
        'weight_decay': 1e-4,
        'early_stop_patience': 15,
        'min_pred_size': 100,
        'num_workers': 8,  # INCREASED: 4→8 workers (faster data loading)
        'prefetch_factor': 4,  # Prefetch 4 batches per worker
        'crop_size': 384,  # REDUCED: 512→384 (faster, fits more in batch)
        'crops_per_tile': 6,  # REDUCED: 8→6 (balance speed vs data)
        'mixup_alpha': 0.2,
        'accum_steps': 1  # REMOVED: gradient accumulation (not needed with large batch)
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"ResNet18 U-Net - MAX GPU MODE")
    print(f"Crop size: {CONFIG['crop_size']} (optimized for speed)")
    print(f"Crops per tile: {CONFIG['crops_per_tile']} (6x data increase)")
    print(f"Batch size: {CONFIG['batch_size']} per GPU (2x GPUs = {CONFIG['batch_size'] * 2} total)")
    print(f"Workers: {CONFIG['num_workers']} with prefetch_factor={CONFIG.get('prefetch_factor', 2)}")
    
    # Check for multiple GPUs
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"✓ Using {n_gpus} GPUs with DataParallel")
    
    # Get all tile IDs
    label_dir = os.path.join(CONFIG['data_dir'], 'label')
    all_files = [f for f in os.listdir(label_dir) if f.endswith('.tif')]
    all_tile_ids = sorted([tid for f in all_files if (tid := get_tile_id(f))])
    
    print(f"Total tiles: {len(all_tile_ids)}")
    print(f"Effective training tiles per fold: {len(all_tile_ids) * 0.8 * CONFIG['crops_per_tile']:.0f}")
    
    # Compute global statistics
    stats_path = os.path.join(CONFIG['work_dir'], 'stats_baseline.json')
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        print(f"Loaded stats from cache")
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
        
        # Create datasets with fixed-size crops
        train_dataset = GlacierDataset(
            CONFIG['data_dir'], train_ids, stats, 
            augment=True, crop_size=CONFIG['crop_size'], 
            crops_per_tile=CONFIG['crops_per_tile']
        )
        val_dataset = GlacierDataset(
            CONFIG['data_dir'], val_ids, stats, 
            augment=False, crop_size=CONFIG['crop_size'], 
            crops_per_tile=1
        )
        
        print(f"Training samples: {len(train_dataset)} ({len(train_ids)} tiles × {CONFIG['crops_per_tile']} crops)")
        print(f"Validation samples: {len(val_dataset)} ({len(val_ids)} tiles)")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG['batch_size'],
            shuffle=True, 
            num_workers=CONFIG['num_workers'], 
            pin_memory=True,
            prefetch_factor=CONFIG.get('prefetch_factor', 2),
            persistent_workers=True  # Keep workers alive between epochs
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=1,
            shuffle=False, 
            num_workers=CONFIG['num_workers'], 
            pin_memory=True,
            prefetch_factor=CONFIG.get('prefetch_factor', 2),
            persistent_workers=True
        )
        
        # Model, optimizer, loss
        model = ResNet18UNet(n_classes=1, n_channels=5).to(device)
        
        # Enable DataParallel for multiple GPUs
        if n_gpus > 1:
            model = nn.DataParallel(model)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'],
                                     weight_decay=CONFIG['weight_decay'])
        criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
        scaler = GradScaler()
        
        # Cosine annealing LR scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG['epochs'], eta_min=1e-6
        )
        
        # Training loop
        best_mcc = -1.0
        patience_counter = 0
        
        for epoch in range(CONFIG['epochs']):
            print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
            
            # Train with MixUp
            train_loss = train_epoch(
                model, train_loader, criterion, optimizer, device, scaler,
                use_mixup=True, mixup_alpha=CONFIG['mixup_alpha']
            )
            
            # Validate
            val_mcc = validate(model, val_loader, device, CONFIG['min_pred_size'])
            
            # Learning rate step
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Train Loss: {train_loss:.4f} | Val MCC: {val_mcc:.4f} | LR: {current_lr:.6f}")
            
            # Save best model
            if val_mcc > best_mcc:
                best_mcc = val_mcc
                patience_counter = 0
                model_path = os.path.join(CONFIG['work_dir'], f'best_maxgpu_fold{fold+1}.pth')
                
                # Save model state dict (unwrap DataParallel if needed)
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), model_path)
                else:
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
    print("MAX GPU MODEL RESULTS")
    print(f"{'='*70}")
    print(f"Baseline MCC: 0.5853")
    print(f"---")
    for i, mcc in enumerate(fold_results):
        improvement = mcc - 0.5853
        print(f"Fold {i + 1}: MCC = {mcc:.4f} (Δ = {improvement:+.4f})")
    print(f"\nMean MCC: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    improvement = np.mean(fold_results) - 0.5853
    print(f"Overall Improvement: {improvement:+.4f}")
    print(f"{'='*70}")
    
    # Save results
    results = {
        'fold_results': fold_results,
        'mean_mcc': float(np.mean(fold_results)),
        'std_mcc': float(np.std(fold_results)),
        'baseline_mcc': 0.5853,
        'improvement': float(improvement),
        'config': {
            'crops_per_tile': CONFIG['crops_per_tile'],
            'crop_size': CONFIG['crop_size'],
            'batch_size': CONFIG['batch_size'],
            'total_batch': CONFIG['batch_size'] * 2,
            'mixup_alpha': CONFIG['mixup_alpha'],
            'num_workers': CONFIG['num_workers']
        }
    }
    
    with open(os.path.join(CONFIG['work_dir'], 'maxgpu_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✓ Training complete! Check maxgpu_results.json for details.")

if __name__ == '__main__':
    main()
