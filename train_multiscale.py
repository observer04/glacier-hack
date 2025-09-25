#!/usr/bin/env python3
"""
Multi-scale training and inference for glacier segmentation.
This script implements advanced techniques to break through performance plateaus.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm

# Import our modules
from data_utils import create_segmentation_dataloaders, compute_global_stats
from models import UNet, DeepLabV3Plus, EfficientUNet
from train_utils import (train_model, CombinedLoss, FocalLoss, DiceLoss, 
                        WeightedBCELoss, TverskyLoss, BoundaryLoss, AdaptiveLoss,
                        collect_validation_probs, compute_threshold_metrics)

def set_seed(seed: int):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class MultiScaleDataset(torch.utils.data.Dataset):
    """Dataset that provides multiple scales of the same image."""
    
    def __init__(self, base_dataset, scales=[0.75, 1.0, 1.25]):
        self.base_dataset = base_dataset
        self.scales = scales
        
    def __len__(self):
        return len(self.base_dataset) * len(self.scales)
    
    def __getitem__(self, idx):
        base_idx = idx // len(self.scales)
        scale_idx = idx % len(self.scales)
        scale = self.scales[scale_idx]
        
        # Get original image and mask
        img, mask = self.base_dataset[base_idx]
        
        if scale != 1.0:
            # Resize image and mask
            _, h, w = img.shape
            new_h, new_w = int(h * scale), int(w * scale)
            
            img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), 
                              mode='bilinear', align_corners=False).squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w),
                               mode='nearest').squeeze(0).squeeze(0)
            
            # Pad or crop to original size if needed
            if new_h != h or new_w != w:
                if new_h < h or new_w < w:
                    # Pad
                    pad_h = max(0, h - new_h)
                    pad_w = max(0, w - new_w)
                    img = F.pad(img, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2))
                    mask = F.pad(mask, (pad_w//2, pad_w-pad_w//2, pad_h//2, pad_h-pad_h//2))
                else:
                    # Crop
                    start_h = (new_h - h) // 2
                    start_w = (new_w - w) // 2
                    img = img[:, start_h:start_h+h, start_w:start_w+w]
                    mask = mask[start_h:start_h+h, start_w:start_w+w]
        
        return img, mask

class EnsembleModel(nn.Module):
    """Ensemble of multiple models for better performance."""
    
    def __init__(self, models, weights=None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0] * len(models)
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        
        # Weighted average
        weighted_sum = sum(w * out for w, out in zip(self.weights, outputs))
        return weighted_sum / self.weights.sum()

def multiscale_inference(model, x, scales=[0.75, 1.0, 1.25], flip_tta=True):
    """Perform multi-scale inference with test-time augmentation."""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for scale in scales:
            if scale != 1.0:
                # Scale input
                _, _, h, w = x.shape
                new_h, new_w = int(h * scale), int(w * scale)
                x_scaled = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
            else:
                x_scaled = x
            
            # Forward pass
            pred = model(x_scaled)
            
            # Scale back to original size
            if scale != 1.0:
                pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)
            
            predictions.append(pred)
            
            # Flip augmentation
            if flip_tta:
                # Horizontal flip
                x_flip = torch.flip(x_scaled, dims=[3])
                pred_flip = model(x_flip)
                pred_flip = torch.flip(pred_flip, dims=[3])
                if scale != 1.0:
                    pred_flip = F.interpolate(pred_flip, size=(h, w), mode='bilinear', align_corners=False)
                predictions.append(pred_flip)
                
                # Vertical flip
                x_flip = torch.flip(x_scaled, dims=[2])
                pred_flip = model(x_flip)
                pred_flip = torch.flip(pred_flip, dims=[2])
                if scale != 1.0:
                    pred_flip = F.interpolate(pred_flip, size=(h, w), mode='bilinear', align_corners=False)
                predictions.append(pred_flip)
    
    # Average all predictions
    return torch.stack(predictions).mean(dim=0)

def train_with_multiscale(model, train_loader, val_loader, criterion, optimizer, 
                         scheduler, args):
    """Train model with multi-scale approach."""
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Setup AMP if enabled
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == 'cuda' else None
    
    best_val_mcc = -1.0
    patience_counter = 0
    
    # Create multi-scale training dataset
    multiscale_train = MultiScaleDataset(train_loader.dataset, scales=[0.8, 1.0, 1.2])
    multiscale_loader = torch.utils.data.DataLoader(
        multiscale_train, 
        batch_size=max(1, args.batch_size // 2),  # Reduce batch size for multi-scale
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        pbar = tqdm(multiscale_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                
                # Backward pass
                scaler.scale(loss).backward()
                
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{train_loss/train_batches:.4f}'
            })
        
        # Validation with multi-scale inference
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(val_loader, desc="Validation"):
                data, targets = data.to(device), targets.to(device)
                
                # Multi-scale inference
                outputs = multiscale_inference(model, data, scales=[0.9, 1.0, 1.1], flip_tta=True)
                
                val_predictions.append(torch.sigmoid(outputs).cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        # Compute metrics
        val_predictions = np.concatenate(val_predictions, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        # Find optimal threshold
        thresholds = np.arange(0.3, 0.8, 0.05)
        best_threshold = 0.5
        best_mcc = -1.0
        
        for thresh in thresholds:
            val_pred_binary = (val_predictions > thresh).astype(np.uint8)
            metrics = compute_threshold_metrics(val_targets, val_pred_binary)
            if metrics['mcc'] > best_mcc:
                best_mcc = metrics['mcc']
                best_threshold = thresh
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss/train_batches:.4f}, "
              f"Val MCC={best_mcc:.4f} (thresh={best_threshold:.2f})")
        
        # Learning rate scheduling
        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(best_mcc)
            else:
                scheduler.step()
        
        # Early stopping and model saving
        if best_mcc > best_val_mcc:
            best_val_mcc = best_mcc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_mcc': best_val_mcc,
                'best_threshold': best_threshold,
            }, os.path.join(args.model_save_path, f"{args.model_type}_multiscale_best.pth"))
            
        else:
            patience_counter += 1
            if patience_counter >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"Training completed. Best validation MCC: {best_val_mcc:.4f}")
    return model

def main(args):
    """Main training function."""
    set_seed(args.seed)
    
    # Create data loaders with global normalization
    print("Creating dataloaders with global normalization...")
    train_loader, val_loader = create_segmentation_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        augment=True,
        use_global_stats=True
    )
    
    # Create model
    print(f"Creating {args.model_type} model...")
    if args.model_type == "unet":
        model = UNet(in_channels=5, num_classes=1)
    elif args.model_type == "deeplabv3plus":
        model = DeepLabV3Plus(in_channels=5, num_classes=1)
    elif args.model_type == "efficientunet":
        model = EfficientUNet(in_channels=5, num_classes=1)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Create loss function
    loss_functions = {
        "bce": nn.BCEWithLogitsLoss(),
        "wbce": WeightedBCELoss(pos_weight=args.pos_weight),
        "focal": FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma),
        "dice": DiceLoss(),
        "combined": CombinedLoss(alpha=args.combined_alpha, beta=args.combined_beta),
        "tversky": TverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta),
        "boundary": BoundaryLoss(),
        "adaptive": AdaptiveLoss()
    }
    
    criterion = loss_functions[args.loss]
    
    # Create optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, 
                             weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, 
                            momentum=0.9, weight_decay=args.weight_decay)
    
    # Create scheduler
    scheduler = None
    if args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5)
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Create save directory
    os.makedirs(args.model_save_path, exist_ok=True)
    
    # Train model
    print("Starting multi-scale training...")
    model = train_with_multiscale(model, train_loader, val_loader, criterion, 
                                 optimizer, scheduler, args)
    
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-scale training for glacier segmentation")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--model_save_path", type=str, default="./models/multiscale", help="Path to save models")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="efficientunet", 
                       choices=["unet", "deeplabv3plus", "efficientunet"], 
                       help="Model architecture")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    
    # Loss function parameters
    parser.add_argument("--loss", type=str, default="tversky", 
                       choices=["bce", "wbce", "focal", "dice", "combined", "tversky", "boundary", "adaptive"], 
                       help="Loss function")
    parser.add_argument("--pos_weight", type=float, default=1.0, help="Positive class weight for WBCE")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Focal loss alpha")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--combined_alpha", type=float, default=0.5, help="Weight for BCE part in CombinedLoss")
    parser.add_argument("--combined_beta", type=float, default=0.5, help="Weight for Dice part in CombinedLoss")
    parser.add_argument("--tversky_alpha", type=float, default=0.7, help="False positive penalty for Tversky loss")
    parser.add_argument("--tversky_beta", type=float, default=0.3, help="False negative penalty for Tversky loss")
    
    # Optimizer and scheduler
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="plateau", 
                       choices=["plateau", "cosine", "none"], help="Learning rate scheduler")
    
    # Regularization and optimization
    parser.add_argument("--early_stopping_patience", type=int, default=15, help="Patience for early stopping")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping max norm")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision")
    
    # System parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    main(args)