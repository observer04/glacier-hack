import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

# Import our modules
from data_utils import create_dataloaders, create_segmentation_dataloaders
from models import PixelANN, UNet, DeepLabV3Plus
from train_utils import train_model, CombinedLoss, FocalLoss, DiceLoss

def main(args):
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
        )
    else:
        # For segmentation models, batch size is per tile; default smaller
        seg_batch = max(1, min(args.batch_size, 4))  # keep small to avoid OOM
        train_loader, val_loader = create_segmentation_dataloaders(
            args.data_dir,
            batch_size=seg_batch,
            val_split=args.val_split,
            num_workers=args.num_workers,
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
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    print(f"Model created: {args.model_type}")
    
    # Define loss function
    if args.model_type == "pixelann":
        if args.loss == "bce":
            criterion = nn.BCELoss()
        elif args.loss == "focal":
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
        elif args.loss == "dice":
            criterion = DiceLoss()
        elif args.loss == "combined":
            criterion = CombinedLoss(alpha=0.5, beta=0.5)
        else:
            raise ValueError(f"Unknown loss function: {args.loss}")
    else:
        # For segmentation models, operate on (N,1,H,W) probabilities vs (N,1,H,W) masks using BCELoss
        criterion = nn.BCELoss()
    
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
        early_stopping_patience=args.early_stopping_patience
    )
    
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
    parser.add_argument("--model_type", type=str, default="pixelann", choices=["pixelann", "unet", "deeplabv3plus"], help="Type of model to train")
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="Dropout rate for PixelANN")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--loss", type=str, default="bce", choices=["bce", "focal", "dice", "combined"], help="Loss function")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine", "none"], help="Learning rate scheduler")
    parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping")
    
    # System parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--model_save_path", type=str, default="./models", help="Path to save models")
    parser.add_argument("--augment", action="store_true", help="Enable simple flips/rotations for segmentation models")
    
    args = parser.parse_args()
    
    if args.scheduler == "none":
        args.scheduler = None
    
    main(args)