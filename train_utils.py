

# train_utils.py - High-Performance Version v3
# Uses stable, global statistics for GPU-side normalization.

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef
import random
import json

# --- GPU-Side Data Processing Functions ---
class GlobalNormalizer(nn.Module):
    """Applies stable normalization using pre-calculated global stats."""
    def __init__(self, stats_path):
        super().__init__()
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            # Reshape for broadcasting: (C) -> (1, C, 1, 1)
            self.mean = torch.tensor(stats['mean']).view(1, 5, 1, 1)
            self.std = torch.tensor(stats['std']).view(1, 5, 1, 1)
            print(f"Loaded global stats from {stats_path}")
        except FileNotFoundError:
            print(f"Warning: stats.json not found at {stats_path}. Using fallback per-batch normalization.")
            self.mean = None
            self.std = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.mean is not None:
            # Use global stats
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
            return (x - self.mean) / (self.std + 1e-8)
        else:
            # Fallback to per-batch normalization (less stable)
            for i in range(x.shape[1]):
                channel_data = x[:, i, :, :]
                mean, std = channel_data.mean(), channel_data.std()
                x[:, i, :, :] = (channel_data - mean) / (std + 1e-8)
            return x

def augment_on_gpu(images: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies random augmentations to a batch of images and masks on the GPU."""
    if masks.dim() == 3:
        masks = masks.unsqueeze(1)

    if random.random() > 0.5:
        images, masks = torch.flip(images, dims=[3]), torch.flip(masks, dims=[3])
    if random.random() > 0.5:
        images, masks = torch.flip(images, dims=[2]), torch.flip(masks, dims=[2])

    if random.random() > 0.5:
        contrast = torch.empty(1, device=images.device).uniform_(0.8, 1.2)
        brightness = torch.empty(1, device=images.device).uniform_(-0.1, 0.1)
        images = images * contrast + brightness
    if random.random() > 0.5:
        noise = torch.randn_like(images) * 0.05
        images = images + noise

    images = torch.clamp(images, -5.0, 5.0)
    return images, masks.squeeze(1)

# --- Loss Functions ---
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, y_pred, y_true):
        y_true_pos, y_pred_pos = y_true.view(-1), y_pred.view(-1)
        true_pos = (y_true_pos * y_pred_pos).sum()
        false_neg = (y_true_pos * (1 - y_pred_pos)).sum()
        false_pos = ((1 - y_true_pos) * y_pred_pos).sum()
        return 1 - (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)

# --- Core Training & Validation Loops ---
def train_epoch(model, dataloader, criterion, optimizer, device, normalizer, accum_steps: int, grad_clip: float, use_amp: bool, augment: bool):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    scaler = torch.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad()

    for step, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        inputs = normalizer(inputs)
        if augment:
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

def validate(model, dataloader, criterion, device, normalizer):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            inputs = normalizer(inputs)
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
               stats_path: str = None, # Path to stats.json
               accum_steps: int = 1, grad_clip: float = 0.0,
               use_amp: bool = False, augment: bool = False):
    os.makedirs(model_save_path, exist_ok=True)
    model = model.to(device)
    normalizer = GlobalNormalizer(stats_path).to(device)

    best_mcc = -1.0
    best_model_path = os.path.join(model_save_path, "best_model.pth")
    no_improve_epochs = 0
    history = {"train_loss": [], "val_loss": [], "train_mcc": [], "val_mcc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        start_time = time.time()
        
        train_loss, train_mcc = train_epoch(
            model, train_loader, criterion, optimizer, device, normalizer,
            accum_steps=accum_steps, grad_clip=grad_clip, use_amp=use_amp, augment=augment
        )
        
        val_loss, val_mcc = validate(model, val_loader, criterion, device, normalizer)
        
        if scheduler:
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
