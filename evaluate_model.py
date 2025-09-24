import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns

# Import our modules
from data_utils import GlacierDataset
from models import PixelANN, UNet, DeepLabV3Plus

def evaluate_model(model, dataloader, device):
    """Evaluate model on dataset."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1)
            
            # Forward pass
            outputs = model(inputs)
            
            # Store predictions and targets for metrics
            preds = (outputs > 0.5).float().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    # Calculate metrics
    mcc = matthews_corrcoef(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    metrics = {
        "mcc": mcc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm
    }
    
    return metrics

def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=["Non-Glacier", "Glacier"],
        yticklabels=["Non-Glacier", "Glacier"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    dataset = GlacierDataset(args.data_dir, is_training=False, val_split=args.val_split)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    if args.model_type == "pixelann":
        model = PixelANN(
            in_channels=5,
            hidden_dims=[32, 64, 128, 64, 32],
            dropout_rate=0.0  # No dropout during evaluation
        )
    elif args.model_type == "unet":
        model = UNet(in_channels=5, out_channels=1)
    elif args.model_type == "deeplabv3plus":
        model = DeepLabV3Plus(in_channels=5, out_channels=1)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Load model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    print(f"Model loaded from {args.model_path}")
    
    # Evaluate model
    metrics = evaluate_model(model, dataloader, device)
    
    # Print metrics
    print(f"MCC: {metrics['mcc']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot confusion matrix
    plot_confusion_matrix(metrics['confusion_matrix'], save_path=os.path.join(args.output_dir, "confusion_matrix.png"))
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write(f"MCC: {metrics['mcc']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate glacier segmentation model")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="./Train", help="Path to data directory")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    
    # Model parameters
    parser.add_argument("--model_type", type=str, default="pixelann", choices=["pixelann", "unet", "deeplabv3plus"], help="Type of model to evaluate")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model weights")
    
    # Evaluation parameters
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size for evaluation")
    
    # System parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--output_dir", type=str, default="./results", help="Path to save results")
    
    args = parser.parse_args()
    
    main(args)