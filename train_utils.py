

# train_utils.py - High-Performance Version
# Normalization and Augmentation are performed on the GPU.

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
import random

# --- NEW: GPU-Side Data Processing Functions ---

def normalize_on_gpu(x: torch.Tensor) -> torch.Tensor:
    """Normalizes a batch of images on the GPU, channel-wise."""
    # x shape is (N, C, H, W)
    x = x.float()
    for i in range(x.shape[1]): # Iterate over channels
        channel_data = x[:, i, :, :]
        mean = channel_data.mean()
        std = channel_data.std()
        if std > 0:
            x[:, i, :, :] = (channel_data - mean) / std
        else:
            x[:, i, :, :] = channel_data - mean
    return x

def augment_on_gpu(images: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies random augmentations to a batch of images and masks on the GPU."""
    # Add a channel dimension to masks for compatibility with flips if it's (N, H, W)
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)

    # Random horizontal flips
    if random.random() > 0.5:
        images = torch.flip(images, dims=[3])
        masks = torch.flip(masks, dims=[3])

    # Random vertical flips
    if random.random() > 0.5:
        images = torch.flip(images, dims=[2])
        masks = torch.flip(masks, dims=[2])

    # Photometric augmentations (on images only)
    if random.random() > 0.5:
        contrast = torch.empty(1, device=images.device).uniform_(0.8, 1.2)
        brightness = torch.empty(1, device=images.device).uniform_(-0.1, 0.1)
        images = images * contrast + brightness

    if random.random() > 0.5:
        noise = torch.randn_like(images) * 0.05
        images = images + noise

    # Clip to a reasonable range
    images = torch.clamp(images, -5.0, 5.0)

    return images, masks.squeeze(1) # Return mask as (N, H, W)

# --- Loss Functions (Unchanged) ---
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_true_pos = y_true.view(-1)
        y_pred_pos = y_pred.view(-1)
        true_pos = (y_true_pos * y_pred_pos).sum()
        false_neg = (y_true_pos * (1 - y_pred_pos)).sum()
        false_pos = ((1 - y_true_pos) * y_pred_pos).sum()
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        return 1 - tversky

# --- Core Training & Validation Loops (Modified for GPU processing) ---

def train_epoch(model, dataloader, criterion, optimizer, device, accum_steps: int = 1, grad_clip: float = 0.0, use_amp: bool = False):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    scaler = torch.amp.GradScaler(enabled=use_amp)

    optimizer.zero_grad()
    for step, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # GPU-SIDE PROCESSING
        inputs = normalize_on_gpu(inputs)
        inputs, targets = augment_on_gpu(inputs, targets)

        with torch.amp.autocast(device_type=device, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))

        scaler.scale(loss / accum_steps).backward()

        if ((step + 1) % accum_steps == 0) or (step + 1 == len(dataloader)):
            if grad_clip > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * inputs.size(0)
        preds_np = (torch.sigmoid(outputs) > 0.5).byte().cpu().numpy().reshape(-1)
        targs_np = targets.byte().cpu().numpy().reshape(-1)
        all_preds.extend(preds_np)
        all_targets.extend(targs_np)

    mcc = matthews_corrcoef(all_targets, all_preds)
    return running_loss / len(dataloader.dataset), mcc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # GPU-SIDE NORMALIZATION (No augmentation for validation)
            inputs = normalize_on_gpu(inputs)

            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))

            running_loss += loss.item() * inputs.size(0)
            preds_np = (torch.sigmoid(outputs) > 0.5).byte().cpu().numpy().reshape(-1)
            targs_np = targets.byte().cpu().numpy().reshape(-1)
            all_preds.extend(preds_np)
            all_targets.extend(targs_np)

    mcc = matthews_corrcoef(all_targets, all_preds)
    return running_loss / len(dataloader.dataset), mcc

def train_model(model, train_loader, val_loader, criterion, optimizer, 
               scheduler=None, num_epochs=50, device="cuda", 
               model_save_path="models", early_stopping_patience=10,
               accum_steps: int = 1, grad_clip: float = 0.0,
               use_amp: bool = False, use_swa: bool = False, # SWA not implemented in this version
               checkpoint_interval: int = 0):
    os.makedirs(model_save_path, exist_ok=True)
    model = model.to(device)
    
    best_mcc = -1.0
    best_model_path = os.path.join(model_save_path, "best_model.pth")
    no_improve_epochs = 0
    history = {"train_loss": [], "val_loss": [], "train_mcc": [], "val_mcc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        start_time = time.time()
        
        train_loss, train_mcc = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            accum_steps=accum_steps, grad_clip=grad_clip, use_amp=use_amp
        )
        
        val_loss, val_mcc = validate(model, val_loader, criterion, device)
        
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_mcc)
            else:
                scheduler.step()
        
        epoch_time = time.time() - start_time
        print(f"Train Loss: {train_loss:.4f}, MCC: {train_mcc:.4f} | Val Loss: {val_loss:.4f}, MCC: {val_mcc:.4f} | Time: {epoch_time:.2f}s")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mcc"].append(train_mcc)
        history["val_mcc"].append(val_mcc)
        
        if val_mcc > best_mcc:
            best_mcc = val_mcc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with MCC: {best_mcc:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs")
        
        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    model.load_state_dict(torch.load(best_model_path))
    return model, history
