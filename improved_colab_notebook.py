
# Improved Colab Notebook for Glacier Segmentation

from google.colab import drive
drive.mount('/content/drive')

epochs = 90

# 1. Setup
# Install the required libraries

import os
import cv2
import tifffile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms



# Helper functions from data_utils.py
def _find_label_path(label_dir: str, tile_id: str):
    """Return the first existing label path for a tile id among known patterns."""
    candidates = [
        os.path.join(label_dir, f"Y{tile_id}.tif"),
        os.path.join(label_dir, f"Y_output_resized_{tile_id}.tif"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def get_tile_id(filename):
    """Extract tile ID from filename."""
    match = re.search(r"img(\d+)\\.tif", filename)
    if match:
        return match.group(1)
    
    match = re.search(r"(\d{2}_\d{2})", filename)
    return match.group(1) if match else None

def normalize_band(band):
    """Normalize a single band using mean and std."""
    band = band.astype(np.float32)
    mean = np.mean(band)
    std = np.std(band)
    if std > 0:
        return (band - mean) / std
    return band

# --- Dataset Definition ---

class GlacierTileDataset(Dataset):
    """Dataset that returns full 5-band tiles and masks for segmentation models.

    Returns:
      - X: FloatTensor of shape (5, H, W)
      - y: FloatTensor of shape (H, W) with values {0,1}
    """

    def __init__(self, data_dir, is_training=True, val_split=0.2, random_state=42, augment: bool=False):
        super().__init__()
        self.data_dir = data_dir
        self.is_training = is_training
        self.augment = augment

        # Band dirs
        self.band_dirs = []
        for i in range(1, 6):
            band_dir = os.path.join(data_dir, f"Band{i}")
            if not os.path.exists(band_dir):
                raise ValueError(f"Band{i} directory not found in {data_dir}")
            self.band_dirs.append(band_dir)

        # Label dir
        self.label_dir = os.path.join(data_dir, "label")
        if not os.path.exists(self.label_dir):
            raise ValueError(f"Label directory not found in {data_dir}")

        # Map band -> tile_id -> filename
        self.band_tile_map = {i: {} for i in range(5)}
        all_ids = set()
        for band_idx, band_dir in enumerate(self.band_dirs):
            for f in os.listdir(band_dir):
                if not f.endswith(".tif"):
                    continue
                tile_id = get_tile_id(f)
                if tile_id:
                    self.band_tile_map[band_idx][tile_id] = f
                    all_ids.add(tile_id)

        # Keep only tiles present in all bands with a matching label file
        self.valid_tile_ids = []
        for tid in all_ids:
            all_bands = all(tid in self.band_tile_map[b] for b in range(5))
            label_exists = _find_label_path(self.label_dir, tid) is not None
            if all_bands and label_exists:
                self.valid_tile_ids.append(tid)

        train_ids, val_ids = train_test_split(self.valid_tile_ids, test_size=val_split, random_state=42)
        self.tile_ids = train_ids if is_training else val_ids
        print(f"{'Training' if is_training else 'Validation'} tile dataset with {len(self.tile_ids)} tiles")

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, index: int):
        tid = self.tile_ids[index]
        bands = []
        H = W = None
        for b in range(5):
            fp = os.path.join(self.band_dirs[b], self.band_tile_map[b][tid])
            arr = np.array(Image.open(fp))
            if arr.ndim == 3:
                arr = arr[..., 0]
            H, W = arr.shape
            # Clean NaN/Inf
            if np.isnan(arr).any() or np.isinf(arr).any():
                arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            arr = normalize_band(arr)
            bands.append(arr)

        # Assert all bands share the same shape
        for k in range(1, len(bands)):
            if bands[k].shape != bands[0].shape:
                raise ValueError(f"Band shape mismatch for tile {tid}: {bands[k].shape} vs {bands[0].shape}")

        x = np.stack(bands, axis=0).astype(np.float32)  # (5, H, W)

        label_path = _find_label_path(self.label_dir, tid)
        if label_path is None:
            raise FileNotFoundError(f"Label not found for tile {tid} in {self.label_dir}")
        y = np.array(Image.open(label_path))
        if y.ndim == 3:
            y = y[..., 0]
        if np.isnan(y).any() or np.isinf(y).any():
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = (y > 0).astype(np.float32)  # (H, W)

        # Simple augmentations for training
        if self.is_training and self.augment:
            # Geometric flips
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=2).copy()
                y = np.flip(y, axis=1).copy()
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=1).copy()
                y = np.flip(y, axis=0).copy()
            # Random 90 deg rotation
            k = np.random.randint(0, 4)
            if k:
                x = np.rot90(x, k=k, axes=(1, 2)).copy()
                y = np.rot90(y, k=k, axes=(0, 1)).copy()

        return torch.from_numpy(x), torch.from_numpy(y)

# 3. Model Definition

# Use a pre-trained Unet with a ResNet34 backbone
model = UNet(in_channels=5, out_channels=1)

# 4. Training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
criterion = nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

# Training loop
best_mcc = -1
no_improve_epochs = 0 # For early stopping

for epoch in range(epochs):
    model.train()
    train_loss = 0
    all_preds = []
    all_masks = []
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, masks.unsqueeze(1)) # Add channel dimension to masks

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0) # Accumulate loss correctly

        preds = (outputs.detach() > 0.5).float().cpu().numpy().reshape(-1)
        all_preds.extend(preds)
        all_masks.extend(masks.float().cpu().numpy().reshape(-1))

    train_loss /= len(train_loader.dataset)
    train_mcc = matthews_corrcoef(all_masks, all_preds)

    # Validation
    model.eval()
    val_loss = 0
    all_preds = []
    all_masks = []
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            print(f"Outputs shape before loss in validation: {outputs.shape}")
            
            loss = criterion(outputs, masks.unsqueeze(1)) # Add channel dimension to masks

            val_loss += loss.item() * images.size(0)

            preds = (outputs.detach() > 0.5).float().cpu().numpy().reshape(-1)
            all_preds.extend(preds)
            all_masks.extend(masks.float().cpu().numpy().reshape(-1))

    val_loss /= len(val_loader.dataset)
    val_mcc = matthews_corrcoef(all_masks, all_preds)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train MCC: {train_mcc:.4f}, Val Loss: {val_loss:.4f}, Val MCC: {val_mcc:.4f}")

    if val_mcc > best_mcc:
        best_mcc = val_mcc
        torch.save(model.state_dict(), model_save_path)
        print(f"New best model saved with MCC: {best_mcc:.4f}")
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        print(f"No improvement for {no_improve_epochs} epochs")

    scheduler.step(val_mcc) # For ReduceLROnPlateau

    if no_improve_epochs >= 10: # Early stopping patience
        print(f"Early stopping after {epoch+1} epochs")
        break

print(f"Training complete. Best MCC: {best_mcc:.4f}")

# 5. Evaluation

# You can now download the 'best_model.pth' file and use it in your 'solution.py'.
# Remember to update your 'solution.py' to use the same model architecture:
# model = smp.Unet(
#     encoder_name="resnet34",
#     encoder_weights=None, # Don't load imagenet weights for inference
#     in_channels=5,
#     classes=1,
# )

