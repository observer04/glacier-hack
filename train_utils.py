import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score

class MCC(nn.Module):
    """Matthews Correlation Coefficient loss function."""
    
    def forward(self, y_pred, y_true):
        """
        Compute MCC loss.
        
        Args:
            y_pred: Predicted probabilities (batch_size, 1)
            y_true: Ground truth labels (batch_size, 1)
        
        Returns:
            Negative MCC (to minimize)
        """
        y_pred_binary = (y_pred > 0.5).float()
        
        tp = torch.sum(y_true * y_pred_binary)
        tn = torch.sum((1 - y_true) * (1 - y_pred_binary))
        fp = torch.sum((1 - y_true) * y_pred_binary)
        fn = torch.sum(y_true * (1 - y_pred_binary))
        
        numerator = tp * tn - fp * fn
        denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        # Add small epsilon to avoid division by zero
        denominator = torch.clamp(denominator, min=1e-8)
        
        # Return negative MCC for minimization
        return -numerator / denominator

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced datasets."""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, y_pred, y_true):
        bce_loss = self.bce(y_pred, y_true)
        
        # Apply focal scaling
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return torch.mean(focal_loss)

class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, y_pred, y_true):
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        
        intersection = torch.sum(y_pred_flat * y_true_flat)
        union = torch.sum(y_pred_flat) + torch.sum(y_true_flat)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice

class CombinedLoss(nn.Module):
    """Combined loss function (BCE + Dice)."""
    
    def __init__(self, alpha=0.5, beta=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        
    def forward(self, y_pred, y_true):
        return self.alpha * self.bce(y_pred, y_true) + self.beta * self.dice(y_pred, y_true)

class WeightedBCELoss(nn.Module):
    """Binary Cross-Entropy with a scalar positive-class weight, operating on probabilities.

    Note: Models here output probabilities via sigmoid. For better numerical stability,
    we clamp predictions to (eps, 1-eps) before computing -[ y*log(p)*pos_w + (1-y)*log(1-p) ].
    """

    def __init__(self, pos_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        loss_pos = -torch.log(y_pred) * y_true
        loss_neg = -torch.log(1.0 - y_pred) * (1.0 - y_true)
        if self.pos_weight != 1.0:
            loss_pos = loss_pos * self.pos_weight
        loss = loss_pos + loss_neg
        return loss.mean()

def _compute_metrics_from_logits(outputs, targets):
    # outputs and targets can be (N,1) or (N,1,H,W) or (N,H,W)
    with torch.no_grad():
        if outputs.dim() == 4:  # (N,1,H,W)
            preds = (outputs > 0.5).float().cpu().numpy().reshape(-1)
            targs = targets.float().cpu().numpy().reshape(-1)
        elif outputs.dim() == 3:  # (N,H,W)
            preds = (outputs > 0.5).float().cpu().numpy().reshape(-1)
            targs = targets.float().cpu().numpy().reshape(-1)
        else:  # (N,1)
            preds = (outputs > 0.5).float().cpu().numpy().reshape(-1)
            targs = targets.float().cpu().numpy().reshape(-1)
    mcc = matthews_corrcoef(targs, preds)
    f1 = f1_score(targs, preds)
    precision = precision_score(targs, preds, zero_division=0)
    recall = recall_score(targs, preds, zero_division=0)
    return mcc, f1, precision, recall


def train_epoch(model, dataloader, criterion, optimizer, device, accum_steps: int = 1, grad_clip: float = 0.0):
    """Train model for one epoch (supports pixel-wise and tile-wise)."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    optimizer.zero_grad()
    for step, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
        inputs = inputs.to(device)
        # Targets: pixel (N,) or (N,1) vs tile (N,H,W)
        # Normalize shapes for loss
        if targets.dim() == 1:
            targets_t = targets.view(-1, 1).to(device)
        elif targets.dim() == 2:
            targets_t = targets.unsqueeze(1).to(device)  # (N,1,H,W) later
        else:
            targets_t = targets.to(device)

        outputs = model(inputs)

        # Ensure output shape matches targets for segmentation
        if outputs.dim() == 4 and targets_t.dim() == 3:
            # outputs: (N,1,H,W), targets_t: (N,H,W)
            loss = criterion(outputs, targets_t.unsqueeze(1))
        elif outputs.dim() == 4 and targets_t.dim() == 4:
            loss = criterion(outputs, targets_t)
        else:
            loss = criterion(outputs, targets_t)

        # Gradient accumulation
        if accum_steps > 1:
            (loss / accum_steps).backward()
            do_step = ((step + 1) % accum_steps == 0) or (step + 1 == len(dataloader))
            if do_step:
                if grad_clip and grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad()
        else:
            loss.backward()
            if grad_clip and grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * inputs.size(0)

        # Metrics accumulation
        if outputs.dim() == 4:
            preds_np = (outputs > 0.5).float().cpu().numpy().reshape(-1)
            targs_np = targets_t.float().cpu().numpy().reshape(-1)
        else:
            preds_np = (outputs > 0.5).float().cpu().numpy().reshape(-1)
            targs_np = targets_t.float().cpu().numpy().reshape(-1)
        all_preds.extend(preds_np)
        all_targets.extend(targs_np)

    mcc = matthews_corrcoef(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    return running_loss / len(dataloader.dataset), mcc, f1, precision, recall

def validate(model, dataloader, criterion, device):
    """Validate model on validation set (supports pixel-wise and tile-wise)."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            if targets.dim() == 1:
                targets_t = targets.view(-1, 1).to(device)
            elif targets.dim() == 2:
                targets_t = targets.unsqueeze(1).to(device)
            else:
                targets_t = targets.to(device)

            outputs = model(inputs)

            if outputs.dim() == 4 and targets_t.dim() == 3:
                loss = criterion(outputs, targets_t.unsqueeze(1))
            elif outputs.dim() == 4 and targets_t.dim() == 4:
                loss = criterion(outputs, targets_t)
            else:
                loss = criterion(outputs, targets_t)

            running_loss += loss.item() * inputs.size(0)

            preds_np = (outputs > 0.5).float().cpu().numpy().reshape(-1)
            targs_np = targets_t.float().cpu().numpy().reshape(-1)
            all_preds.extend(preds_np)
            all_targets.extend(targs_np)

    mcc = matthews_corrcoef(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    return running_loss / len(dataloader.dataset), mcc, f1, precision, recall

def train_model(model, train_loader, val_loader, criterion, optimizer, 
               scheduler=None, num_epochs=50, device="cuda", 
               model_save_path="models", early_stopping_patience=10,
               accum_steps: int = 1, grad_clip: float = 0.0):
    """Train model with validation and early stopping."""
    # Create directory for saving models if it doesn't exist
    os.makedirs(model_save_path, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Initialize tracking variables
    best_mcc = -1.0
    best_model_path = os.path.join(model_save_path, "best_model.pth")
    no_improve_epochs = 0
    
    # Initialize history for plotting
    history = {
        "train_loss": [], "val_loss": [],
        "train_mcc": [], "val_mcc": [],
        "train_f1": [], "val_f1": [],
        "train_precision": [], "val_precision": [],
        "train_recall": [], "val_recall": []
    }
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        start_time = time.time()
        
        # Train epoch
        train_loss, train_mcc, train_f1, train_precision, train_recall = train_epoch(
            model, train_loader, criterion, optimizer, device, accum_steps=accum_steps, grad_clip=grad_clip
        )
        
        # Validate epoch
        val_loss, val_mcc, val_f1, val_precision, val_recall = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_mcc)  # Use MCC for ReduceLROnPlateau
            else:
                scheduler.step()
        
        # Print metrics
        epoch_time = time.time() - start_time
        print(f"Train Loss: {train_loss:.4f}, MCC: {train_mcc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, MCC: {val_mcc:.4f}, F1: {val_f1:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        
        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mcc"].append(train_mcc)
        history["val_mcc"].append(val_mcc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["train_precision"].append(train_precision)
        history["val_precision"].append(val_precision)
        history["train_recall"].append(train_recall)
        history["val_recall"].append(val_recall)
        
        # Save best model
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with MCC: {best_mcc:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs")
        
        # Early stopping
        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path))
    
    # Plot training curves
    plot_training_curves(history, os.path.join(model_save_path, "training_curves.png"))
    
    return model, history

def plot_training_curves(history, save_path=None):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history["train_loss"], label="Train")
    axes[0, 0].plot(history["val_loss"], label="Validation")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    
    # MCC plot
    axes[0, 1].plot(history["train_mcc"], label="Train")
    axes[0, 1].plot(history["val_mcc"], label="Validation")
    axes[0, 1].set_title("MCC")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MCC")
    axes[0, 1].legend()
    
    # F1 plot
    axes[1, 0].plot(history["train_f1"], label="Train")
    axes[1, 0].plot(history["val_f1"], label="Validation")
    axes[1, 0].set_title("F1 Score")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1")
    axes[1, 0].legend()
    
    # Precision-Recall plot
    axes[1, 1].plot(history["train_precision"], label="Train Precision")
    axes[1, 1].plot(history["val_precision"], label="Val Precision")
    axes[1, 1].plot(history["train_recall"], label="Train Recall")
    axes[1, 1].plot(history["val_recall"], label="Val Recall")
    axes[1, 1].set_title("Precision and Recall")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()