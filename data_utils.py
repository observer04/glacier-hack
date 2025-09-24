import os
import numpy as np
import re
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import glob

def _find_label_path(label_dir: str, tile_id: str):
    """Return the first existing label path for a tile id among known patterns.

    Supports both:
    - Y{tile_id}.tif
    - Y_output_resized_{tile_id}.tif
    """
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
    match = re.search(r"img(\d+)\.tif", filename)
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

class GlacierDataset(Dataset):
    """Dataset for glacier segmentation with pixel-wise approach."""
    
    def __init__(self, data_dir, is_training=True, val_split=0.2, random_state=42):
        super().__init__()
        self.data_dir = data_dir
        self.is_training = is_training
        
        # Get all band directories
        self.band_dirs = []
        for i in range(1, 6):
            band_dir = os.path.join(data_dir, f"Band{i}")
            if os.path.exists(band_dir):
                self.band_dirs.append(band_dir)
            else:
                raise ValueError(f"Band{i} directory not found in {data_dir}")
        
        # Get label directory
        self.label_dir = os.path.join(data_dir, "label")
        if not os.path.exists(self.label_dir):
            raise ValueError(f"Label directory not found in {data_dir}")
        
        # Map band -> tile_id -> filename
        self.band_tile_map = {i: {} for i in range(len(self.band_dirs))}
        self.all_tile_ids = set()
        
        # Collect all tiles
        for band_idx, band_dir in enumerate(self.band_dirs):
            files = [f for f in os.listdir(band_dir) if f.endswith(".tif")]
            for f in files:
                tile_id = get_tile_id(f)
                if tile_id:
                    self.band_tile_map[band_idx][tile_id] = f
                    self.all_tile_ids.add(tile_id)
        
        # Check if all bands have all tiles
        self.valid_tile_ids = []
        for tile_id in self.all_tile_ids:
            # Check if all bands have this tile
            all_bands_have_tile = all(tile_id in self.band_tile_map[band_idx] for band_idx in range(len(self.band_dirs)))
            # Check if label exists for this tile
            label_exists = _find_label_path(self.label_dir, tile_id) is not None
            
            if all_bands_have_tile and label_exists:
                self.valid_tile_ids.append(tile_id)
        
        # Split into train and validation
        train_ids, val_ids = train_test_split(
            self.valid_tile_ids, test_size=val_split, random_state=random_state
        )
        
        self.tile_ids = train_ids if is_training else val_ids
        print(f"{'Training' if is_training else 'Validation'} dataset with {len(self.tile_ids)} tiles")
        
        # Initialize arrays; load_data will populate real values
        self.pixels = np.empty((0, 5), dtype=np.float32)
        self.labels = np.empty((0,), dtype=np.float32)
        # Populate via loader (uses local accumulators)
        self.load_data()
        
    def load_data(self):
        """Load all pixel data and labels for the dataset."""
        pixels_list = []
        labels_list = []
        for tile_id in self.tile_ids:
            # Load all bands (keep raw for valid mask; store normalized for features)
            band_arrays_norm = []
            zero_masks = []
            H = W = None
            shapes = []
            for band_idx in range(len(self.band_dirs)):
                file_path = os.path.join(
                    self.band_dirs[band_idx], 
                    self.band_tile_map[band_idx][tile_id]
                )
                arr = np.array(Image.open(file_path))
                if arr.ndim == 3:
                    arr = arr[..., 0]
                if H is None:
                    H, W = arr.shape
                shapes.append(arr.shape)
                # Clean NaN/Inf in raw band
                if np.isnan(arr).any() or np.isinf(arr).any():
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                zero_masks.append(arr == 0)
                # Normalize the band
                arr_norm = normalize_band(arr)
                band_arrays_norm.append(arr_norm)

            # Shape consistency across bands
            if any(s != shapes[0] for s in shapes):
                raise ValueError(f"Band shape mismatch for tile {tile_id}: {shapes}")

            # Stack normalized bands to form a 5-channel image
            bands_stacked = np.stack(band_arrays_norm, axis=0).astype(np.float32)  # (5, H, W)
            
            # Load label (support multiple naming patterns)
            label_path = _find_label_path(self.label_dir, tile_id)
            if label_path is None:
                # Skip if label missing (should be rare after valid_tile_ids filtering)
                continue
            label = np.array(Image.open(label_path))
            if label.ndim == 3:
                label = label[..., 0]
            if np.isnan(label).any() or np.isinf(label).any():
                label = np.nan_to_num(label, nan=0.0, posinf=0.0, neginf=0.0)
            label = (label > 0).astype(np.float32)

            if label.shape != (H, W):
                raise ValueError(f"Label shape {label.shape} does not match bands {(H, W)} for tile {tile_id}")
            
            # Compute valid pixels from raw bands: pixels where NOT all bands are zero
            all_zero = np.all(np.stack(zero_masks, axis=0), axis=0)  # (H, W)
            valid_pixels = ~all_zero
            if not valid_pixels.any():
                # Skip tiles with no valid pixels
                continue

            # Convert to pixel-wise samples and filter by valid mask
            X = np.transpose(bands_stacked, (1, 2, 0))  # (H, W, 5)
            X = X.reshape(-1, 5)
            y = label.reshape(-1)
            valid_flat = valid_pixels.reshape(-1)
            pixels_list.append(X[valid_flat])
            labels_list.append(y[valid_flat])
        
        # Concatenate all pixels and labels
        if len(pixels_list) == 0:
            raise ValueError("No valid pixels found after filtering; check data integrity and masks.")
        self.pixels = np.concatenate(pixels_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)
        print(f"Dataset contains {len(self.pixels)} pixels")
        
    def __len__(self):
        return len(self.pixels)
    
    def __getitem__(self, idx):
        pixel = self.pixels[idx]
        label = self.labels[idx]
        return torch.tensor(pixel, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def create_dataloaders(data_dir, batch_size=4096, val_split=0.2, num_workers=4):
    """Create train and validation dataloaders."""
    train_dataset = GlacierDataset(data_dir, is_training=True, val_split=val_split)
    val_dataset = GlacierDataset(data_dir, is_training=False, val_split=val_split)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


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

        train_ids, val_ids = train_test_split(self.valid_tile_ids, test_size=val_split, random_state=random_state)
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
            # Random horizontal flip
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=2).copy()
                y = np.flip(y, axis=1).copy()
            # Random vertical flip
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=1).copy()
                y = np.flip(y, axis=0).copy()
            # Random 90-degree rotation
            k = np.random.randint(0, 4)
            if k:
                x = np.rot90(x, k=k, axes=(1, 2)).copy()
                y = np.rot90(y, k=k, axes=(0, 1)).copy()

        return torch.from_numpy(x), torch.from_numpy(y)


def create_segmentation_dataloaders(data_dir, batch_size=2, val_split=0.2, num_workers=2, augment: bool=False):
    """Create dataloaders for segmentation models operating on full tiles."""
    train_dataset = GlacierTileDataset(data_dir, is_training=True, val_split=val_split, augment=augment)
    val_dataset = GlacierTileDataset(data_dir, is_training=False, val_split=val_split, augment=False)

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