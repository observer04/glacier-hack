# ==================================================================================
# KAGGLE.PY - SELF-CONTAINED SCRIPT FOR GLACIER SEGMENTATION
#
# This script includes:
# 1. A Multi-Scale U-Net with Channel Attention.
# 2. Data loading and pre-processing utilities.
# 3. A full training and validation pipeline with Tversky Loss.
# 4. Hyperparameter search using Optuna.
# 5. Final model training using the best found parameters.
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
import optuna
import shutil

# ==================================================================================
# --- 1. MODEL ARCHITECTURE: Multi-Scale U-Net with Squeeze-and-Excitation ---
# ==================================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention."""
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
    """(Convolution => BatchNorm => ReLU) * 2"""
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
    """Downscaling with MaxPool then DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then DoubleConv"""
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
    """
    A U-Net architecture that processes multi-scale inputs and uses channel attention.
    The multi-scale input is generated internally.
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MultiScaleUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # The input will be the original channels + downscaled-and-upscaled channels
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
        # Create multi-scale input on the fly
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
# --- 2. DATA HANDLING UTILITIES ---
# ==================================================================================

def get_tile_id(filename):
    """Extracts tile ID (e.g., '01_02') from a filename."""
    match = re.search(r'(\d{2}_\d{2})', filename)
    return match.group(1) if match else None

def _find_label_path(label_dir, tile_id):
    """Finds the path to a label file given a tile ID."""
    # This supports multiple naming conventions for label files
    for name in [f"Y{tile_id}.tif", f"Y_output_resized_{tile_id}.tif"]:
        path = os.path.join(label_dir, name)
        if os.path.exists(path):
            return path
    return None

def normalize_band(band, mean, std):
    """Normalizes a single band using pre-computed stats."""
    band = band.astype(np.float32)
    if std > 0:
        return (band - mean) / std
    return band - mean

class GlacierTileDataset(Dataset):
    """Dataset that returns full 5-band tiles and masks for segmentation."""
    def __init__(self, data_dir, tile_ids, global_stats, augment=False):
        self.data_dir = data_dir
        self.tile_ids = tile_ids
        self.global_stats = global_stats
        self.augment = augment
        self.band_dirs = [os.path.join(data_dir, f"Band{i}") for i in range(1, 6)]
        self.label_dir = os.path.join(data_dir, "label")

        # Create a map of tile_id -> filename for each band
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

def compute_and_save_global_stats(data_dir, stats_path, sample_ratio=0.2):
    """Computes and saves global mean/std for each band."""
    print("Computing global normalization statistics...")
    band_dirs = [os.path.join(data_dir, f"Band{i}") for i in range(1, 6)]
    all_files = []
    for band_idx, band_dir in enumerate(band_dirs):
        files = [os.path.join(band_dir, f) for f in os.listdir(band_dir) if f.endswith(".tif")]
        all_files.extend([(band_idx, p) for p in files])
    
    random.shuffle(all_files)
    sample_size = int(len(all_files) * sample_ratio)
    sampled_files = all_files[:sample_size]
    
    band_values = {i: [] for i in range(5)}
    for band_idx, file_path in tqdm(sampled_files, desc="Sampling files for stats"):
        arr = np.array(Image.open(file_path))
        valid_pixels = arr[arr > 0]
        if len(valid_pixels) > 0:
            band_values[band_idx].extend(valid_pixels)

    global_stats = {}
    for i in range(5):
        values = np.array(band_values[i])
        mean, std = np.mean(values), np.std(values)
        # FIX: Use string keys to be consistent with JSON format
        global_stats[str(i)] = (float(mean), float(std))
        print(f"Band {i+1}: mean={mean:.2f}, std={std:.2f}")

    with open(stats_path, 'w') as f:
        json.dump(global_stats, f)
    return global_stats

# ==================================================================================
# --- 3. TRAINING & VALIDATION PIPELINE ---
# ==================================================================================

class TverskyLoss(nn.Module):
    """Tversky Loss for segmentation, good for imbalanced data."""
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

def plot_history(history, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['val_mcc'], label='Validation MCC')
    plt.title('MCC')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# ==================================================================================
# --- 4. OPTUNA HYPERPARAMETER SEARCH ---
# ==================================================================================

def objective(trial, data_dir, all_tile_ids, global_stats, trial_epochs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Hyperparameters to Tune ---
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    tversky_alpha = trial.suggest_float("tversky_alpha", 0.5, 0.9)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16]) # Reduced batch size to prevent OOM
    
    # --- Data Loaders ---
    train_ids, val_ids = train_test_split(all_tile_ids, test_size=0.2, random_state=42)
    train_dataset = GlacierTileDataset(data_dir, train_ids, global_stats, augment=True)
    val_dataset = GlacierTileDataset(data_dir, val_ids, global_stats, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    # --- Model, Loss, Optimizer ---
    model = MultiScaleUNet(n_channels=5, n_classes=1).to(device)
    if torch.cuda.device_count() > 1:
        print(f"--- Using {torch.cuda.device_count()} GPUs for trial! ---")
        model = nn.DataParallel(model)

    criterion = TverskyLoss(alpha=tversky_alpha, beta=(1.0 - tversky_alpha))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()

    # --- Training Loop for Trial ---
    try:
        best_trial_mcc = -1.0
        for epoch in range(trial_epochs):
            train_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_mcc = validate(model, val_loader, device)
            print(f"Trial {trial.number}, Epoch {epoch+1}: Val MCC = {val_mcc:.4f}")
            if val_mcc > best_trial_mcc:
                best_trial_mcc = val_mcc
            trial.report(val_mcc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return best_trial_mcc
    except torch.cuda.OutOfMemoryError:
        print(f"--- Trial {trial.number} ran out of memory and will be pruned. ---")
        raise optuna.exceptions.TrialPruned()

# ==================================================================================
# --- 5. MAIN EXECUTION WORKFLOW ---
# ==================================================================================

if __name__ == '__main__':
    # --- Configuration ---
    DATA_DIR = "/kaggle/working/Train" # Set to your training data path
    WORK_DIR = "/kaggle/working/"
    STATS_PATH = os.path.join(WORK_DIR, "global_stats.json")
    
    # Optuna settings
    N_TRIALS = 15
    TRIAL_EPOCHS = 20
    
    # Final training settings
    FINAL_EPOCHS = 75
    EARLY_STOPPING_PATIENCE = 10

    # --- Step 1: Compute Global Stats ---
    global_stats = None
    if os.path.exists(STATS_PATH):
        try:
            with open(STATS_PATH, 'r') as f:
                global_stats = json.load(f)
            print("Loaded existing global stats.")
        except json.JSONDecodeError:
            print(f"Warning: Found corrupted '{STATS_PATH}'. Re-computing stats.")
            
    if global_stats is None:
        global_stats = compute_and_save_global_stats(DATA_DIR, STATS_PATH)

    # Get all valid tile IDs for splitting
    label_dir = os.path.join(DATA_DIR, "label")
    all_tile_ids = sorted([get_tile_id(f) for f in os.listdir(label_dir) if f.endswith(".tif")])
    
    # --- Step 2: Run Optuna Study ---
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, DATA_DIR, all_tile_ids, global_stats, TRIAL_EPOCHS), n_trials=N_TRIALS)

    print("\n--- Optuna Search Complete ---")
    best_trial = study.best_trial
    print(f"Best trial MCC: {best_trial.value:.4f}")
    print("Best hyperparameters:", best_trial.params)

    # --- Step 3: Train Final Model with Best Hyperparameters ---
    print("\n--- Training Final Model ---")
    
    # Data loaders for final training
    train_ids, val_ids = train_test_split(all_tile_ids, test_size=0.2, random_state=42)
    final_train_dataset = GlacierTileDataset(DATA_DIR, train_ids, global_stats, augment=True)
    final_val_dataset = GlacierTileDataset(DATA_DIR, val_ids, global_stats, augment=False)
    final_train_loader = DataLoader(final_train_dataset, batch_size=best_trial.params['batch_size'], shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    final_val_loader = DataLoader(final_val_dataset, batch_size=best_trial.params['batch_size'], shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    # Model and optimizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    final_model = MultiScaleUNet(n_channels=5, n_classes=1).to(device)
    if torch.cuda.device_count() > 1:
        print(f"--- Using {torch.cuda.device_count()} GPUs for final training! ---")
        final_model = nn.DataParallel(final_model)

    final_criterion = TverskyLoss(alpha=best_trial.params['tversky_alpha'], beta=(1.0 - best_trial.params['tversky_alpha']))
    final_optimizer = optim.AdamW(final_model.parameters(), lr=best_trial.params['learning_rate'], weight_decay=best_trial.params['weight_decay'])
    final_scaler = torch.cuda.amp.GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(final_optimizer, 'max', factor=0.2, patience=5)

    # Final training loop
    best_mcc = -1.0
    no_improve_epochs = 0
    history = {'train_loss': [], 'val_mcc': []}
    
    for epoch in range(FINAL_EPOCHS):
        print(f"\nEpoch {epoch+1}/{FINAL_EPOCHS}")
        train_loss = train_epoch(final_model, final_train_loader, final_criterion, final_optimizer, device, final_scaler)
        val_mcc = validate(final_model, final_val_loader, device)
        scheduler.step(val_mcc)
        
        print(f"Train Loss: {train_loss:.4f}, Val MCC: {val_mcc:.4f}")
        history['train_loss'].append(train_loss)
        history['val_mcc'].append(val_mcc)

        if val_mcc > best_mcc:
            best_mcc = val_mcc
            no_improve_epochs = 0
            torch.save(final_model.state_dict(), os.path.join(WORK_DIR, 'best_model.pth'))
            print(f"Saved new best model with MCC: {best_mcc:.4f}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs.")

        if no_improve_epochs >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break
            
    # --- Step 4: Save Artifacts ---
    plot_history(history, os.path.join(WORK_DIR, "training_history.png"))
    print(f"\nFinal model and training history saved in {WORK_DIR}")