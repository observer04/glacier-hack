

# ==================================================================================
# FINETUNE.PY - Fine-tuning script for the Glacier-Hack Model
#
# This script:
# 1. Loads the best model saved from the main `kaggle.py` run.
# 2. Uses the same best hyperparameters found by Optuna.
# 3. Trains for 30 more epochs with a 10x smaller learning rate.
# ==================================================================================

# --- Imports ---
import os
import re
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==================================================================================
# --- 1. MODEL ARCHITECTURE (Copied from kaggle.py) ---
# ==================================================================================

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class MultiScaleUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MultiScaleUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels * 2, 64)
        self.se1 = SEBlock(64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x_down = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_down_up = F.interpolate(x_down, size=x.shape[2:], mode='bilinear', align_corners=False)
        x_multi_scale = torch.cat([x, x_down_up], dim=1)
        x1 = self.inc(x_multi_scale)
        x1 = self.se1(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# ==================================================================================
# --- 2. DATA & TRAINING UTILITIES (Copied from kaggle.py) ---
# ==================================================================================

def get_tile_id(filename):
    match = re.search(r'(\d{2}_\d{2})', filename)
    return match.group(1) if match else None

def _find_label_path(label_dir, tile_id):
    for name in [f"Y{tile_id}.tif", f"Y_output_resized_{tile_id}.tif"]:
        path = os.path.join(label_dir, name)
        if os.path.exists(path):
            return path
    return None

def normalize_band(band, mean, std):
    band = band.astype(np.float32)
    if std > 0:
        return (band - mean) / std
    return band - mean

class GlacierTileDataset(Dataset):
    def __init__(self, data_dir, tile_ids, global_stats, augment=False):
        self.data_dir = data_dir
        self.tile_ids = tile_ids
        self.global_stats = global_stats
        self.augment = augment
        self.band_dirs = [os.path.join(data_dir, f"Band{i}") for i in range(1, 6)]
        self.label_dir = os.path.join(data_dir, "label")
        self.band_tile_map = {i: {} for i in range(5)}
        for band_idx, band_dir in enumerate(self.band_dirs):
            for f in os.listdir(band_dir):
                if f.endswith(".tif"):
                    tid = get_tile_id(f)
                    if tid:
                        self.band_tile_map[band_idx][tid] = f
    def __len__(self):
        return len(self.tile_ids)
    def __getitem__(self, index):
        tid = self.tile_ids[index]
        bands = []
        for b in range(5):
            fp = os.path.join(self.band_dirs[b], self.band_tile_map[b][tid])
            arr = np.array(Image.open(fp))
            if arr.ndim == 3: arr = arr[..., 0]
            arr = np.nan_to_num(arr)
            mean, std = self.global_stats[str(b)]
            arr = normalize_band(arr, mean, std)
            bands.append(arr)
        x = np.stack(bands, axis=0).astype(np.float32)
        label_path = _find_label_path(self.label_dir, tid)
        y = np.array(Image.open(label_path))
        if y.ndim == 3: y = y[..., 0]
        y = (y > 0).astype(np.float32)
        if self.augment:
            if random.random() > 0.5: x, y = np.flip(x, axis=2).copy(), np.flip(y, axis=1).copy()
            if random.random() > 0.5: x, y = np.flip(x, axis=1).copy(), np.flip(y, axis=0).copy()
            k = random.randint(0, 3)
            if k > 0: x, y = np.rot90(x, k, axes=(1, 2)).copy(), np.rot90(y, k, axes=(0, 1)).copy()
        return torch.from_numpy(x), torch.from_numpy(y)

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth
    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        y_true_pos, y_pred_pos = y_true.view(-1), y_pred.view(-1)
        true_pos = (y_true_pos * y_pred_pos).sum()
        false_neg = (y_true_pos * (1 - y_pred_pos)).sum()
        false_pos = ((1 - y_true_pos) * y_pred_pos).sum()
        return 1 - (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=True):
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=True):
                outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).byte().cpu().numpy().flatten()
            all_preds.extend(preds)
            all_targets.extend(targets.byte().cpu().numpy().flatten())
    if len(all_targets) == 0 or len(all_preds) == 0:
        return 0.0
    return matthews_corrcoef(all_targets, all_preds)

# ==================================================================================
# --- 3. MAIN FINE-TUNING WORKFLOW ---
# ==================================================================================

if __name__ == '__main__':
    # --- Configuration ---
    DATA_DIR = "/kaggle/working/Train"
    WORK_DIR = "/kaggle/working/"
    STATS_PATH = os.path.join(WORK_DIR, "global_stats.json")
    MODEL_TO_FINETUNE_PATH = os.path.join(WORK_DIR, "best_model.pth")
    FINETUNED_MODEL_SAVE_PATH = os.path.join(WORK_DIR, "finetuned_model.pth")
    
    # --- Best Hyperparameters (from previous run) ---
    # Paste the best hyperparameters from your Optuna run here
    best_params = {
        'learning_rate': 9.609162091855998e-05,
        'weight_decay': 3.891039322734828e-05,
        'tversky_alpha': 0.7603887319847192,
        'batch_size': 4
    }

    # --- Fine-tuning Settings ---
    FINETUNE_LR_DIVISOR = 10
    FINETUNE_EPOCHS = 30
    
    print("--- Starting Fine-Tuning Session ---")
    print(f"Loading model from: {MODEL_TO_FINETUNE_PATH}")

    # --- Load Data ---
    with open(STATS_PATH, 'r') as f:
        global_stats = json.load(f)
    label_dir = os.path.join(DATA_DIR, "label")
    all_tile_ids = sorted([get_tile_id(f) for f in os.listdir(label_dir) if f.endswith(".tif")])
    train_ids, val_ids = train_test_split(all_tile_ids, test_size=0.2, random_state=42)
    
    train_dataset = GlacierTileDataset(DATA_DIR, train_ids, global_stats, augment=True)
    val_dataset = GlacierTileDataset(DATA_DIR, val_ids, global_stats, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    # --- Load Model and Optimizer ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiScaleUNet(n_channels=5, n_classes=1)
    
    # Load the state dict
    # Important: The model was saved with DataParallel, so the keys have a 'module.' prefix.
    state_dict = torch.load(MODEL_TO_FINETUNE_PATH)
    if torch.cuda.device_count() > 1:
        print(f"--- Using {torch.cuda.device_count()} GPUs for fine-tuning! ---")
        model = nn.DataParallel(model)
        model.load_state_dict(state_dict)
    else:
        # If not using DataParallel, we need to remove the 'module.' prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    
    model.to(device)

    criterion = TverskyLoss(alpha=best_params['tversky_alpha'], beta=(1.0 - best_params['tversky_alpha']))
    optimizer = optim.AdamW(model.parameters(), lr=best_params['learning_rate'] / FINETUNE_LR_DIVISOR, weight_decay=best_params['weight_decay'])
    scaler = torch.cuda.amp.GradScaler("cuda")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

    # --- Fine-Tuning Loop ---
    best_mcc = 0.0
    print(f"Starting fine-tuning for {FINETUNE_EPOCHS} epochs with LR = {optimizer.param_groups[0]['lr']:.2e}")

    for epoch in range(FINETUNE_EPOCHS):
        print(f"\nEpoch {epoch+1}/{FINETUNE_EPOCHS}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_mcc = validate(model, val_loader, device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Val MCC: {val_mcc:.4f}")

        if val_mcc > best_mcc:
            best_mcc = val_mcc
            torch.save(model.state_dict(), FINETUNED_MODEL_SAVE_PATH)
            print(f"Saved new best fine-tuned model with MCC: {best_mcc:.4f}")

    print(f"\n--- Fine-Tuning Complete ---")
    print(f"Best fine-tuned model saved to {FINETUNED_MODEL_SAVE_PATH} with MCC: {best_mcc:.4f}")
