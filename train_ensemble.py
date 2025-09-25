#!/usr/bin/env python3
"""
Ensemble training script for glacier segmentation.
Trains multiple models and combines their predictions for maximum performance.
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
import json

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

class EnsembleTrainer:
    """Class to handle ensemble training and validation."""
    
    def __init__(self, model_configs, data_loaders, device):
        self.model_configs = model_configs
        self.train_loader, self.val_loader = data_loaders
        self.device = device
        self.models = []
        self.optimizers = []
        self.schedulers = []
        self.criterions = []
        
        # Initialize models
        for config in model_configs:
            model = self._create_model(config['model_type'])
            model = model.to(device)
            self.models.append(model)
            
            # Create optimizer
            if config['optimizer'] == 'adam':
                optimizer = optim.Adam(model.parameters(), 
                                     lr=config['learning_rate'],
                                     weight_decay=config['weight_decay'])
            else:
                optimizer = optim.SGD(model.parameters(),
                                    lr=config['learning_rate'],
                                    momentum=0.9,
                                    weight_decay=config['weight_decay'])
            self.optimizers.append(optimizer)
            
            # Create scheduler
            if config['scheduler'] == 'plateau':
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=5)
            elif config['scheduler'] == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config['epochs'], eta_min=1e-6)
            else:
                scheduler = None
            self.schedulers.append(scheduler)
            
            # Create loss function
            criterion = self._create_loss_function(config)
            self.criterions.append(criterion)
    
    def _create_model(self, model_type):
        """Create model based on type."""
        if model_type == "unet":
            return UNet(in_channels=5, num_classes=1)
        elif model_type == "deeplabv3plus":
            return DeepLabV3Plus(in_channels=5, num_classes=1)
        elif model_type == "efficientunet":
            return EfficientUNet(in_channels=5, num_classes=1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_loss_function(self, config):
        """Create loss function based on config."""
        loss_type = config['loss']
        if loss_type == "bce":
            return nn.BCEWithLogitsLoss()
        elif loss_type == "wbce":
            return WeightedBCELoss(pos_weight=config.get('pos_weight', 1.0))
        elif loss_type == "focal":
            return FocalLoss(alpha=config.get('focal_alpha', 0.25),
                           gamma=config.get('focal_gamma', 2.0))
        elif loss_type == "dice":
            return DiceLoss()
        elif loss_type == "combined":
            return CombinedLoss(alpha=config.get('combined_alpha', 0.5),
                              beta=config.get('combined_beta', 0.5))
        elif loss_type == "tversky":
            return TverskyLoss(alpha=config.get('tversky_alpha', 0.7),
                             beta=config.get('tversky_beta', 0.3))
        elif loss_type == "boundary":
            return BoundaryLoss()
        elif loss_type == "adaptive":
            return AdaptiveLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def train_single_model(self, model_idx, epochs, save_path):
        """Train a single model in the ensemble."""
        model = self.models[model_idx]
        optimizer = self.optimizers[model_idx]
        scheduler = self.schedulers[model_idx]
        criterion = self.criterions[model_idx]
        config = self.model_configs[model_idx]
        
        print(f"Training {config['model_type']} with {config['loss']} loss...")
        
        best_val_mcc = -1.0
        patience_counter = 0
        
        # Setup AMP if enabled
        scaler = torch.cuda.amp.GradScaler() if config.get('amp', False) and self.device.type == 'cuda' else None
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            pbar = tqdm(self.train_loader, desc=f"Model {model_idx+1} Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (data, targets) in enumerate(pbar):
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(data)
                        loss = criterion(outputs, targets)
                    
                    scaler.scale(loss).backward()
                    
                    if config.get('grad_clip', 0) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    
                    if config.get('grad_clip', 0) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                    
                    optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Validation
            val_mcc = self._validate_single_model(model, model_idx)
            
            print(f"Model {model_idx+1} Epoch {epoch+1}: Train Loss={train_loss/train_batches:.4f}, Val MCC={val_mcc:.4f}")
            
            # Learning rate scheduling
            if scheduler is not None:
                if config['scheduler'] == "plateau":
                    scheduler.step(val_mcc)
                else:
                    scheduler.step()
            
            # Save best model
            if val_mcc > best_val_mcc:
                best_val_mcc = val_mcc
                patience_counter = 0
                
                model_save_path = os.path.join(save_path, f"model_{model_idx}_{config['model_type']}_best.pth")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'epoch': epoch,
                    'best_mcc': best_val_mcc,
                }, model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= config.get('early_stopping_patience', 10):
                    print(f"Early stopping for model {model_idx+1} at epoch {epoch+1}")
                    break
        
        print(f"Model {model_idx+1} training completed. Best MCC: {best_val_mcc:.4f}")
        return best_val_mcc
    
    def _validate_single_model(self, model, model_idx):
        """Validate a single model."""
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = torch.sigmoid(model(data))
                
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        val_predictions = np.concatenate(val_predictions, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        # Find optimal threshold
        thresholds = np.arange(0.3, 0.8, 0.05)
        best_mcc = -1.0
        
        for thresh in thresholds:
            val_pred_binary = (val_predictions > thresh).astype(np.uint8)
            metrics = compute_threshold_metrics(val_targets, val_pred_binary)
            if metrics['mcc'] > best_mcc:
                best_mcc = metrics['mcc']
        
        return best_mcc
    
    def validate_ensemble(self, weights=None):
        """Validate ensemble performance."""
        if weights is None:
            weights = [1.0] * len(self.models)
        
        # Set all models to eval mode
        for model in self.models:
            model.eval()
        
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc="Ensemble validation"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Get predictions from all models
                ensemble_pred = None
                total_weight = 0.0
                
                for i, (model, weight) in enumerate(zip(self.models, weights)):
                    pred = torch.sigmoid(model(data))
                    if ensemble_pred is None:
                        ensemble_pred = weight * pred
                    else:
                        ensemble_pred += weight * pred
                    total_weight += weight
                
                ensemble_pred /= total_weight
                
                val_predictions.append(ensemble_pred.cpu().numpy())
                val_targets.append(targets.cpu().numpy())
        
        val_predictions = np.concatenate(val_predictions, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        # Find optimal threshold
        thresholds = np.arange(0.3, 0.8, 0.05)
        best_mcc = -1.0
        best_threshold = 0.5
        
        for thresh in thresholds:
            val_pred_binary = (val_predictions > thresh).astype(np.uint8)
            metrics = compute_threshold_metrics(val_targets, val_pred_binary)
            if metrics['mcc'] > best_mcc:
                best_mcc = metrics['mcc']
                best_threshold = thresh
        
        return best_mcc, best_threshold
    
    def optimize_ensemble_weights(self):
        """Optimize ensemble weights using grid search."""
        print("Optimizing ensemble weights...")
        
        # Generate weight combinations
        weight_grid = np.arange(0.5, 1.5, 0.25)
        best_weights = None
        best_mcc = -1.0
        
        # Try different weight combinations (simplified grid search)
        for w1 in weight_grid:
            for w2 in weight_grid:
                if len(self.models) == 2:
                    weights = [w1, w2]
                else:
                    for w3 in weight_grid:
                        weights = [w1, w2, w3]
                        if len(self.models) > 3:
                            # For more models, use equal weights for simplicity
                            weights.extend([1.0] * (len(self.models) - 3))
                
                mcc, _ = self.validate_ensemble(weights)
                
                if mcc > best_mcc:
                    best_mcc = mcc
                    best_weights = weights.copy()
                
                if len(self.models) == 2:
                    break
        
        print(f"Best ensemble weights: {best_weights}, MCC: {best_mcc:.4f}")
        return best_weights, best_mcc

def create_ensemble_configs():
    """Create configurations for different models in the ensemble."""
    configs = [
        {
            'model_type': 'efficientunet',
            'loss': 'tversky',
            'tversky_alpha': 0.7,
            'tversky_beta': 0.3,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'optimizer': 'adam',
            'scheduler': 'plateau',
            'epochs': 80,
            'amp': True,
            'grad_clip': 1.0,
            'early_stopping_patience': 15
        },
        {
            'model_type': 'unet',
            'loss': 'boundary',
            'learning_rate': 8e-4,
            'weight_decay': 2e-4,
            'optimizer': 'adam',
            'scheduler': 'cosine',
            'epochs': 80,
            'amp': True,
            'grad_clip': 1.0,
            'early_stopping_patience': 15
        },
        {
            'model_type': 'deeplabv3plus',
            'loss': 'adaptive',
            'learning_rate': 5e-4,
            'weight_decay': 1e-4,
            'optimizer': 'adam',
            'scheduler': 'plateau',
            'epochs': 80,
            'amp': True,
            'grad_clip': 1.0,
            'early_stopping_patience': 15
        }
    ]
    return configs

def main(args):
    """Main ensemble training function."""
    set_seed(args.seed)
    
    # Create data loaders
    print("Creating dataloaders with global normalization...")
    train_loader, val_loader = create_segmentation_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        augment=True,
        use_global_stats=True
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create ensemble configurations
    model_configs = create_ensemble_configs()
    if args.num_models < len(model_configs):
        model_configs = model_configs[:args.num_models]
    
    # Create ensemble trainer
    ensemble_trainer = EnsembleTrainer(model_configs, (train_loader, val_loader), device)
    
    # Create save directory
    os.makedirs(args.model_save_path, exist_ok=True)
    
    # Train individual models
    individual_scores = []
    for i in range(len(model_configs)):
        score = ensemble_trainer.train_single_model(i, model_configs[i]['epochs'], args.model_save_path)
        individual_scores.append(score)
    
    print("\nIndividual model scores:")
    for i, (config, score) in enumerate(zip(model_configs, individual_scores)):
        print(f"Model {i+1} ({config['model_type']} + {config['loss']}): {score:.4f}")
    
    # Optimize ensemble weights
    best_weights, ensemble_mcc = ensemble_trainer.optimize_ensemble_weights()
    
    print(f"\nEnsemble performance: {ensemble_mcc:.4f}")
    print(f"Improvement over best individual: {ensemble_mcc - max(individual_scores):.4f}")
    
    # Save ensemble configuration
    ensemble_config = {
        'model_configs': model_configs,
        'best_weights': best_weights,
        'ensemble_mcc': ensemble_mcc,
        'individual_scores': individual_scores
    }
    
    config_path = os.path.join(args.model_save_path, "ensemble_config.json")
    with open(config_path, 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    print(f"Ensemble configuration saved to {config_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble training for glacier segmentation")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--model_save_path", type=str, default="./models/ensemble", help="Path to save models")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--num_models", type=int, default=3, help="Number of models in ensemble")
    
    # System parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    main(args)