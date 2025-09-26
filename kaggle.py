# kaggle.py - MONOLITHIC SCRIPT - V5
# All code is self-contained in this single file to eliminate import/NameError issues.

print("--- Starting Monolithic Kaggle Workflow ---")

# 1. Install Optuna
!pip install optuna -q

# 2. All Imports
import os
import torch
import torch.optim as optim
import torch.nn as nn
import optuna
import shutil
import numpy as np
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import json

# ======================================================================================
# SECTION 1: MODEL DEFINITION (from models.py)
# ======================================================================================
print("Defining model architectures...")

class UNet(nn.Module):
    """U-Net model for semantic segmentation of satellite imagery."""
    def __init__(self, in_channels=5, out_channels=1):
        super(UNet, self).__init__()
        def _block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        self.enc1 = _block(in_channels, 64)
        self.enc2 = _block(64, 128)
        self.enc3 = _block(128, 256)
        self.enc4 = _block(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _block(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = _block(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = _block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _block(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = _block(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))
        dec1 = self.up1(bottleneck)
        dec1 = torch.cat((dec1, enc4), dim=1)
        dec1 = self.dec1(dec1)
        dec2 = self.up2(dec1)
        dec2 = torch.cat((dec2, enc3), dim=1)
        dec2 = self.dec2(dec2)
        dec3 = self.up3(dec2)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.dec3(dec3)
        dec4 = self.up4(dec3)
        dec4 = torch.cat((dec4, enc1), dim=1)
        dec4 = self.dec4(dec4)
        return self.final(dec4)

# ======================================================================================
# SECTION 2: DATA UTILITIES (from data_utils_combo.py)
# ======================================================================================
print("Defining data utilities...")

def _find_label_path(label_dir: str, tile_id: str):
    candidates = [os.path.join(label_dir, f"Y{tile_id}.tif"), os.path.join(label_dir, f"Y_output_resized_{tile_id}.tif")]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

class GlacierDatasetCombo(Dataset):
    def __init__(self, processed_dir, is_training=True, val_split=0.2, random_state=42, augment: bool=False):
        super().__init__()
        self.processed_dir, self.is_training, self.augment = processed_dir, is_training, augment
        self.label_dir = os.path.join(processed_dir, "label")
        if not os.path.exists(self.label_dir):
            raise ValueError(f"Label directory not found in {self.processed_dir}")
        all_tile_ids = [os.path.splitext(f)[0] for f in os.listdir(processed_dir) if f.endswith(".npy")]
        self.valid_tile_ids = [tid for tid in all_tile_ids if _find_label_path(self.label_dir, tid) is not None]
        train_ids, val_ids = train_test_split(self.valid_tile_ids, test_size=val_split, random_state=random_state)
        self.tile_ids = train_ids if is_training else val_ids
        print(f"{'Training' if is_training else 'Validation'} combo dataset with {len(self.tile_ids)} tiles from {processed_dir}")

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, index: int):
        tid = self.tile_ids[index]
        x_path = os.path.join(self.processed_dir, f"{tid}.npy")
        x = np.load(x_path)
        label_path = _find_label_path(self.label_dir, tid)
        y = np.array(Image.open(label_path))
        if y.ndim == 3:
            y = y[..., 0]
        y = (y > 0).astype(np.float32)
        x = torch.from_numpy(x.copy()).permute(2, 0, 1)
        y = torch.from_numpy(y.copy())
        return x, y

def create_segmentation_dataloaders_combo(processed_dir, batch_size=2, val_split=0.2, num_workers=4, augment: bool=False):
    train_dataset = GlacierDatasetCombo(processed_dir, is_training=True, val_split=val_split, augment=augment)
    val_dataset = GlacierDatasetCombo(processed_dir, is_training=False, val_split=val_split, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# ======================================================================================
# SECTION 3: TRAINING UTILITIES (from train_utils.py)
# ======================================================================================
print("Defining training utilities...")

from sklearn.metrics import matthews_corrcoef

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

def augment_on_gpu(images: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

def train_epoch(model, dataloader, criterion, optimizer, device, normalizer, accum_steps: int, grad_clip: float, use_amp: bool, augment: bool):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    scaler = torch.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad()
    for step, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training")):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
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
    return running_loss / len(dataloader.dataset), matthews_corrcoef(all_targets, all_preds)

def validate(model, dataloader, criterion, device, normalizer):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Validation"):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            inputs = normalizer(inputs)
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            running_loss += loss.item() * inputs.size(0)
            preds_np = (torch.sigmoid(outputs) > 0.5).byte().cpu().numpy().reshape(-1)
            targs_np = targets.byte().cpu().numpy().reshape(-1)
            all_preds.extend(preds_np)
            all_targets.extend(targs_np)
    return running_loss / len(dataloader.dataset), matthews_corrcoef(all_targets, all_preds)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, model_save_path, stats_path, early_stopping_patience, accum_steps, grad_clip, use_amp, augment):
    class GlobalNormalizer(nn.Module):
        def __init__(self, stats_path):
            super().__init__()
            try:
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                self.mean = torch.tensor(stats['mean']).view(1, 5, 1, 1)
                self.std = torch.tensor(stats['std']).view(1, 5, 1, 1)
                print(f"Loaded global stats from {stats_path}")
            except (FileNotFoundError, TypeError):
                print(f"Warning: stats.json not found or invalid. Using fallback per-batch normalization.")
                self.mean = None
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.float()
            if self.mean is not None:
                self.mean, self.std = self.mean.to(x.device), self.std.to(x.device)
                return (x - self.mean) / (self.std + 1e-8)
            else:
                for i in range(x.shape[1]):
                    channel_data = x[:, i, :, :]
                    mean, std = channel_data.mean(), channel_data.std()
                    x[:, i, :, :] = (channel_data - mean) / (std + 1e-8)
                return x

    os.makedirs(model_save_path, exist_ok=True)
    model.to(device)
    normalizer = GlobalNormalizer(stats_path).to(device)
    best_mcc = -1.0
    no_improve_epochs = 0
    history = {"train_loss": [], "val_loss": [], "train_mcc": [], "val_mcc": []}
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        start_time = time.time()
        train_loss, train_mcc = train_epoch(model, train_loader, criterion, optimizer, device, normalizer, accum_steps, grad_clip, use_amp, augment)
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
            torch.save(model.state_dict(), os.path.join(model_save_path, "best_model.pth"))
            print(f"Saved best model with MCC: {best_mcc:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs")
        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    model.load_state_dict(torch.load(os.path.join(model_save_path, "best_model.pth")))
    return model, history

# ======================================================================================
# SECTION 4: MAIN EXECUTION BLOCK
# ======================================================================================
print("Defining main execution block...")

# --- Configuration ---
ORIGINAL_DATA_DIR = "/kaggle/working/Train"
PROCESSED_DATA_DIR = "/kaggle/working/Train_processed"
STATS_PATH = os.path.join(PROCESSED_DATA_DIR, "stats.json")
KAGGLE_WORKING_DIR = "/kaggle/working/"
N_TRIALS = 15
TRIAL_EPOCHS = 30

# --- Preprocessing ---
def get_tile_id_preprocess(filename):
    match = re.search(r'(\d{2}_\d{2})', filename)
    return match.group(1) if match else None

def run_preprocessing(input_dir, output_dir):
    print("\n" + "*"*80)
    print("STEP 1: PRE-PROCESSING DATASET FOR FASTER TRAINING")
    print("*"*80 + "\n")
    os.makedirs(output_dir, exist_ok=True)
    band_dirs = {f"Band{i}": os.path.join(input_dir, f"Band{i}") for i in range(1, 6)}
    all_files = os.listdir(band_dirs["Band1"])
    tile_ids = sorted(list(set(tid for f in all_files if (tid := get_tile_id_preprocess(f)) is not None)))
    print(f"Found {len(tile_ids)} unique tiles to process.")

    print("--- Pass 1: Computing Global Statistics ---")
    channel_sums, channel_sq_sums, pixel_counts = np.zeros(5, dtype=np.float64), np.zeros(5, dtype=np.float64), np.zeros(5, dtype=np.int64)
    for tid in tqdm(tile_ids, desc="Calculating Stats"):
        for i in range(1, 6):
            band_dir = band_dirs[f"Band{i}"]
            try:
                matching_files = [f for f in os.listdir(band_dir) if tid in f]
                if not matching_files: raise IndexError
                img = tifffile.imread(os.path.join(band_dir, matching_files[0])).astype(np.float64)
                valid_pixels = img[img > 0]
                if valid_pixels.size > 0:
                    channel_sums[i-1] += valid_pixels.sum()
                    channel_sq_sums[i-1] += (valid_pixels**2).sum()
                    pixel_counts[i-1] += valid_pixels.size
            except (IndexError, FileNotFoundError): continue
    mean, std = channel_sums / pixel_counts, np.sqrt((channel_sq_sums / pixel_counts) - (mean**2))
    stats = {'mean': mean.tolist(), 'std': std.tolist()}
    with open(STATS_PATH, 'w') as f:
        json.dump(stats, f)
    print(f"Global stats saved to {STATS_PATH}")

    print("--- Pass 2: Saving Combined .npy Files ---")
    for tid in tqdm(tile_ids, desc="Processing Tiles"):
        bands = []
        all_bands_exist = True
        for i in range(1, 6):
            band_dir = band_dirs[f"Band{i}"]
            try:
                matching_files = [f for f in os.listdir(band_dir) if tid in f]
                if not matching_files: raise IndexError
                bands.append(tifffile.imread(os.path.join(band_dir, matching_files[0])))
            except (IndexError, FileNotFoundError):
                all_bands_exist = False
                break
        if not all_bands_exist: continue
        combined_array = np.stack(bands, axis=-1).astype(np.int16)
        np.save(os.path.join(output_dir, f"{tid}.npy"), combined_array)
    
    input_label_dir, output_label_dir = os.path.join(input_dir, "label"), os.path.join(output_dir, "label")
    if os.path.exists(input_label_dir):
        shutil.copytree(input_label_dir, output_label_dir, dirs_exist_ok=True)
    print("\n--- Pre-processing complete. ---")

# --- Optuna Objective ---
def objective(trial: optuna.trial.Trial) -> float:
    print(f"\n--- Starting Trial {trial.number} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    tversky_alpha = trial.suggest_float("tversky_alpha", 0.4, 0.6)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam"])
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    augment = trial.suggest_categorical("augment", [True, False])

    train_loader, val_loader = create_segmentation_dataloaders_combo(PROCESSED_DATA_DIR, batch_size=batch_size, num_workers=2, augment=augment)
    model = UNet(in_channels=5, out_channels=1)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    criterion = TverskyLoss(alpha=tversky_alpha, beta=(1.0-tversky_alpha))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5)

    try:
        trial_save_path = os.path.join(KAGGLE_WORKING_DIR, f"optuna_trial_{trial.number}")
        _, history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, TRIAL_EPOCHS, device, trial_save_path, STATS_PATH, 10, 1, 0.0, True, augment)
        best_val_mcc = max(history.get("val_mcc", [0.0]))
        shutil.rmtree(trial_save_path, ignore_errors=True)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise optuna.exceptions.TrialPruned()
        else:
            raise e
    return best_val_mcc

# --- Main Runner ---
if __name__ == '__main__':
    run_preprocessing(ORIGINAL_DATA_DIR, PROCESSED_DATA_DIR)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n\n--- Optuna Search Complete ---")
    best_trial = study.best_trial
    print(f"Best trial achieved a validation MCC of: {best_trial.value:.4f}")
    print("\nOptimal hyperparameters found:")
    for key, value in best_trial.params.items():
        print(f"  --{key}: {value}")

    augment_flag = "--augment" if best_trial.params.get("augment", False) else ""
    print("\n" + "*"*80)
    print("COPY AND RUN THIS COMMAND TO TRAIN YOUR FINAL MODEL:")
    print("*"*80 + "\n")
    final_model_path = "/kaggle/working/final_model"
    final_command = f"python train_model.py --data_dir '{PROCESSED_DATA_DIR}' --use_combo_loader --stats_path '{STATS_PATH}' --model_type unet --loss tversky --tversky_alpha {best_trial.params['tversky_alpha']:.4f} --tversky_beta {1.0 - best_trial.params['tversky_alpha']:.4f} --learning_rate {best_trial.params['learning_rate']:.6f} --weight_decay {best_trial.params['weight_decay']:.6f} --batch_size {best_trial.params['batch_size']} --epochs 150 --optimizer {best_trial.params['optimizer'].lower()} --scheduler plateau --amp {augment_flag} --threshold_sweep --early_stopping_patience 20 --num_workers 4 --model_save_path '{final_model_path}'"
    print(final_command)
    print("\n" + "*"*80)