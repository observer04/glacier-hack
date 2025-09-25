import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

# Import our modules
from data_utils import create_dataloaders, create_segmentation_dataloaders
from models import PixelANN, UNet, DeepLabV3Plus, EfficientUNet
from train_utils import train_model, CombinedLoss, FocalLoss, DiceLoss, WeightedBCELoss, TverskyLoss, BoundaryLoss, AdaptiveLoss, collect_validation_probs
import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    # Reproducibility
    set_seed(args.seed)
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataloaders: pixel-wise for PixelANN, tile-wise for UNet/DeepLab
    if args.model_type == "pixelann":
        train_loader, val_loader = create_dataloaders(
            args.data_dir,
            batch_size=args.batch_size,
            val_split=args.val_split,
            num_workers=args.num_workers,
            use_global_stats=args.global_stats,
        )
    else:
        # For segmentation models, batch size is per tile; default smaller
        seg_batch = max(1, min(args.batch_size, 4))  # keep small to avoid OOM
        train_loader, val_loader = create_segmentation_dataloaders(
            args.data_dir,
            batch_size=seg_batch,
            val_split=args.val_split,
            num_workers=args.num_workers,
            use_global_stats=args.global_stats,
            augment=args.augment,
        )
    
    # Create model
    if args.model_type == "pixelann":
        model = PixelANN(
            in_channels=5,
            hidden_dims=[32, 64, 128, 64, 32],
            dropout_rate=args.dropout_rate
        )
    elif args.model_type == "unet":
        model = UNet(in_channels=5, out_channels=1)
    elif args.model_type == "deeplabv3plus":
        model = DeepLabV3Plus(in_channels=5, out_channels=1)
    elif args.model_type == "efficient_unet":
        model = EfficientUNet(
            in_channels=5, 
            out_channels=1,
            width_mult=1.0  # Can be tuned for model size/performance trade-off
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    print(f"Model created: {args.model_type}")
    
    # Define loss function
    loss_functions = {
        "bce": nn.BCELoss(),
        "wbce": WeightedBCELoss(pos_weight=args.pos_weight),
        "focal": FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma),
        "dice": DiceLoss(),
        "combined": CombinedLoss(alpha=args.combined_alpha, beta=args.combined_beta),
        "tversky": TverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta),
        "boundary": BoundaryLoss(),
        "adaptive": AdaptiveLoss()
    }
    
    if args.loss not in loss_functions:
        raise ValueError(f"Unknown loss function: {args.loss}")
    
    criterion = loss_functions[args.loss]
    
    # Define optimizer
    if args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Define scheduler
    if args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    else:
        scheduler = None
    
    # Create directory for saving models
    os.makedirs(args.model_save_path, exist_ok=True)
    
    # Train model
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.epochs,
        device=device_str,
        model_save_path=args.model_save_path,
        early_stopping_patience=args.early_stopping_patience,
        accum_steps=args.accum_steps,
        grad_clip=args.grad_clip,
        use_amp=args.amp,
        use_swa=args.swa,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Optional threshold sweep
    if args.threshold_sweep:
        print("\nRunning threshold sweep on validation set (saved best model loaded)...")
        # reload best model weights (best_model.pth inside save path)
        best_path = os.path.join(args.model_save_path, "best_model.pth")
        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device_str))
        model.eval()
        _, val_loader = (train_loader, val_loader)
        y_true, y_prob = collect_validation_probs(model, val_loader, device_str)
        import numpy as np
        best_thr = 0.5
        best_mcc = -1.0
        from sklearn.metrics import matthews_corrcoef
        for thr in np.linspace(0.3, 0.8, 51):  # 0.01 steps
            preds = (y_prob > thr).astype(int)
            mcc = matthews_corrcoef(y_true, preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thr = thr
        with open(os.path.join(args.model_save_path, "best_threshold.txt"), "w") as f:
            f.write(f"best_threshold={best_thr}\n")
            f.write(f"best_mcc={best_mcc:.5f}\n")
        print(f"Best threshold sweep result: thr={best_thr:.3f} MCC={best_mcc:.4f}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.model_save_path, "final_model.pth"))
    print(f"Final model saved to {os.path.join(args.model_save_path, 'final_model.pth')}")
    
    # Save model for submission (renamed to model.pth)
    torch.save(model.state_dict(), os.path.join(args.model_save_path, "model.pth"))
    print(f"Submission model saved to {os.path.join(args.model_save_path, 'model.pth')}")
    
    return model, history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train glacier segmentation model")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./Train", help="Path to training data directory")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="pixelann", choices=["pixelann", "unet", "deeplabv3plus", "efficient_unet"], help="Type of model to train")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate for PixelANN")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "wbce", "focal", "dice", "combined", "tversky", "boundary", "adaptive"], help="Loss function")
    parser.add_argument("--pos_weight", type=float, default=1.0, help="Positive class weight for WBCE")
    parser.add_argument("--focal_alpha", type=float, default=0.25, help="Focal loss alpha")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Focal loss gamma")
    parser.add_argument("--combined_alpha", type=float, default=0.5, help="Weight for BCE part in CombinedLoss")
    parser.add_argument("--combined_beta", type=float, default=0.5, help="Weight for Dice part in CombinedLoss")
    parser.add_argument("--tversky_alpha", type=float, default=0.7, help="False positive penalty for Tversky loss")
    parser.add_argument("--tversky_beta", type=float, default=0.3, help="False negative penalty for Tversky loss")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine", "none"], help="Learning rate scheduler")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--accum_steps", type=int, default=1, help="Gradient accumulation steps (segmentation models)")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision (CUDA only)")
    parser.add_argument("--swa", action="store_true", help="Enable Stochastic Weight Averaging")
    parser.add_argument("--checkpoint_interval", type=int, default=0, help="Save intermediate checkpoints every N epochs (0 to disable)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--threshold_sweep", action="store_true", help="Run threshold sweep on validation set after training")
    parser.add_argument("--global_stats", action="store_true", help="Use global normalization statistics instead of per-tile")
    
    # System parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--model_save_path", type=str, default="./models", help="Path to save models")
    parser.add_argument("--augment", action="store_true", help="Enable simple flips/rotations for segmentation models")
    
    args = parser.parse_args()
    
    if args.scheduler == "none":
        args.scheduler = None
    
    main(args)