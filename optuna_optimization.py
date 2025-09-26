"""
Optuna Hyperparameter Optimization for Glacier Segmentation
Optimizes UNet + Tversky training to push beyond 63.66% MCC baseline
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import json
from datetime import datetime

# Add project directory to path
sys.path.append('/content/glacier-hack')

from data_utils import GlacierTileDataset, compute_global_stats
from models import UNet
from train_utils import TverskyLoss, train_model

def objective(trial):
    """Optuna objective function for hyperparameter optimization"""
    
    # Sample hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 5e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 6, 8, 10, 12])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    # Tversky loss parameters
    tversky_alpha = trial.suggest_float('tversky_alpha', 0.5, 0.8)
    tversky_beta = trial.suggest_float('tversky_beta', 0.2, 0.5)
    
    # Scheduler choice
    scheduler_type = trial.suggest_categorical('scheduler', ['cosine', 'plateau', 'step'])
    
    # Regularization
    grad_clip = trial.suggest_float('grad_clip', 0.5, 2.0)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
    
    try:
        # Setup model and training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model with dropout
        model = UNet(in_channels=5, out_channels=1, dropout_rate=dropout_rate)
        model = model.to(device)
        
        # Setup optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Setup scheduler
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        else:  # step
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # Setup loss function with optimized parameters
        criterion = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        
        # Load data with global normalization
        train_dataset = GlacierTileDataset(
            data_dir='/content/glacier-hack/Train',
            normalize_type='global',
            augment=True,
            train=True
        )
        
        val_dataset = GlacierTileDataset(
            data_dir='/content/glacier-hack/Train',
            normalize_type='global',
            augment=False,
            train=False
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True
        )
        
        # Training configuration
        config = {
            'epochs': 40,  # Shorter for optimization
            'early_stopping_patience': 10,
            'use_amp': True,
            'gradient_clip_val': grad_clip,
            'device': device
        }
        
        # Train model
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            **config
        )
        
        # Return best validation MCC
        best_val_mcc = max([epoch['val_mcc'] for epoch in history])
        
        # Log trial results
        trial.set_user_attr('best_val_mcc', best_val_mcc)
        trial.set_user_attr('params', {
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'tversky_alpha': tversky_alpha,
            'tversky_beta': tversky_beta,
            'scheduler': scheduler_type,
            'grad_clip': grad_clip,
            'dropout_rate': dropout_rate
        })
        
        return best_val_mcc
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0  # Return poor score for failed trials

def run_optuna_optimization(n_trials=20, timeout_hours=4):
    """Run Optuna hyperparameter optimization"""
    
    print("🚀 Starting Optuna Hyperparameter Optimization")
    print(f"Target: Beat baseline MCC 0.6366 (63.66%)")
    print(f"Trials: {n_trials}, Timeout: {timeout_hours} hours")
    
    # Create study
    study_name = f"glacier_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )
    
    # Run optimization
    study.optimize(
        objective, 
        n_trials=n_trials, 
        timeout=timeout_hours * 3600,
        show_progress_bar=True
    )
    
    # Save results
    results_dir = '/content/drive/MyDrive/glacier_hack/optuna_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Best trial results
    best_trial = study.best_trial
    print(f"\n🎯 OPTIMIZATION RESULTS:")
    print(f"Best MCC: {best_trial.value:.4f}")
    print(f"Improvement: {best_trial.value - 0.6366:.4f} (+{((best_trial.value/0.6366)-1)*100:.1f}%)")
    
    print(f"\n📋 Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save detailed results
    results = {
        'study_name': study_name,
        'best_mcc': best_trial.value,
        'baseline_mcc': 0.6366,
        'improvement': best_trial.value - 0.6366,
        'best_params': best_trial.params,
        'n_trials': len(study.trials),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f'{results_dir}/optimization_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate optimized training command
    params = best_trial.params
    optimized_command = f"""
# OPTUNA-OPTIMIZED TRAINING COMMAND
!python train_model.py \\
  --data_dir "./Train" \\
  --model_type unet \\
  --loss tversky \\
  --tversky_alpha {params['tversky_alpha']:.3f} \\
  --tversky_beta {params['tversky_beta']:.3f} \\
  --learning_rate {params['learning_rate']:.6f} \\
  --weight_decay {params['weight_decay']:.6f} \\
  --batch_size {params['batch_size']} \\
  --epochs 80 \\
  --optimizer adam \\
  --scheduler {params['scheduler']} \\
  --amp \\
  --grad_clip {params['grad_clip']:.2f} \\
  --global_stats \\
  --threshold_sweep \\
  --early_stopping_patience 15 \\
  --num_workers 2 \\
  --model_save_path "/content/drive/MyDrive/glacier_hack/models/unet_optuna_optimized"
"""
    
    with open(f'{results_dir}/optimized_command.txt', 'w') as f:
        f.write(optimized_command)
    
    print(f"\n💾 Results saved to: {results_dir}")
    print(f"📝 Optimized command saved for final training!")
    
    return study, best_trial

if __name__ == "__main__":
    # Run optimization
    study, best_trial = run_optuna_optimization(n_trials=15, timeout_hours=3)
    
    print("\n🎯 Ready to train with optimized hyperparameters!")
    print("Expected MCC improvement: 4-8% above baseline 63.66%")