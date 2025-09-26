
# data_utils_combo.py
# Contains a high-performance Dataset and DataLoader for pre-processed data.

import os
import numpy as np
import re
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF
import random

# --- Re-used utility functions ---
def _find_label_path(label_dir: str, tile_id: str):
    candidates = [
        os.path.join(label_dir, f"Y{tile_id}.tif"),
        os.path.join(label_dir, f"Y_output_resized_{tile_id}.tif"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize a 5-channel image using per-channel mean and std."""
    image = image.astype(np.float32)
    for i in range(image.shape[-1]): # Iterate over channels
        band = image[..., i]
        mean = np.mean(band)
        std = np.std(band)
        if std > 0:
            image[..., i] = (band - mean) / std
        else:
            image[..., i] = band - mean
    return image

# --- High-Performance Dataset for Combined Data ---
class GlacierDatasetCombo(Dataset):
    """Dataset that reads pre-combined 5-band NumPy arrays for speed."""

    def __init__(self, processed_dir, is_training=True, val_split=0.2, random_state=42, augment: bool=False):
        super().__init__()
        self.processed_dir = processed_dir
        self.is_training = is_training
        self.augment = augment
        self.label_dir = os.path.join(processed_dir, "label")

        if not os.path.exists(self.label_dir):
            raise ValueError(f"Label directory not found in {self.processed_dir}")

        # Get tile IDs from the .npy filenames
        all_tile_ids = [os.path.splitext(f)[0] for f in os.listdir(processed_dir) if f.endswith(".npy")]
        
        # Ensure labels exist for all found tiles
        self.valid_tile_ids = [tid for tid in all_tile_ids if _find_label_path(self.label_dir, tid) is not None]

        train_ids, val_ids = train_test_split(self.valid_tile_ids, test_size=val_split, random_state=random_state)
        self.tile_ids = train_ids if is_training else val_ids
        print(f"{'Training' if is_training else 'Validation'} combo dataset with {len(self.tile_ids)} tiles from {processed_dir}")

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, index: int):
        tid = self.tile_ids[index]
        
        # Load the single pre-combined .npy file
        # Shape is expected to be (H, W, 5)
        x_path = os.path.join(self.processed_dir, f"{tid}.npy")
        x = np.load(x_path)

        # Convert to tensors. Augmentation and Normalization will be done on the GPU.
        x = torch.from_numpy(x.copy()).permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        y = torch.from_numpy(y.copy()) # (H, W)

        return x, y


def create_segmentation_dataloaders_combo(processed_dir, batch_size=2, val_split=0.2, num_workers=4, augment: bool=False):
    """Create dataloaders for segmentation models using the combo dataset."""
    train_dataset = GlacierDatasetCombo(processed_dir, is_training=True, val_split=val_split, augment=augment)
    val_dataset = GlacierDatasetCombo(processed_dir, is_training=False, val_split=val_split, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
