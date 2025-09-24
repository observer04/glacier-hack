import os
import numpy as np
import re
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import glob

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
            label_exists = os.path.exists(os.path.join(self.label_dir, f"Y{tile_id}.tif"))
            
            if all_bands_have_tile and label_exists:
                self.valid_tile_ids.append(tile_id)
        
        # Split into train and validation
        train_ids, val_ids = train_test_split(
            self.valid_tile_ids, test_size=val_split, random_state=random_state
        )
        
        self.tile_ids = train_ids if is_training else val_ids
        print(f"{'Training' if is_training else 'Validation'} dataset with {len(self.tile_ids)} tiles")
        
        # Initialize pixel data and labels
        self.pixels = []
        self.labels = []
        self.load_data()
        
    def load_data(self):
        """Load all pixel data and labels for the dataset."""
        for tile_id in self.tile_ids:
            # Load all bands
            band_arrays = []
            for band_idx in range(len(self.band_dirs)):
                file_path = os.path.join(
                    self.band_dirs[band_idx], 
                    self.band_tile_map[band_idx][tile_id]
                )
                arr = np.array(Image.open(file_path))
                if arr.ndim == 3:
                    arr = arr[..., 0]
                    
                # Normalize the band
                arr = normalize_band(arr)
                band_arrays.append(arr)
                
            # Stack bands to form a 5-channel image
            bands_stacked = np.stack(band_arrays, axis=0)  # (5, H, W)
            
            # Load label
            label_path = os.path.join(self.label_dir, f"Y{tile_id}.tif")
            label = np.array(Image.open(label_path))
            label = (label > 0).astype(np.float32)
            
            # Convert to pixel-wise samples
            H, W = label.shape
            X = np.transpose(bands_stacked, (1, 2, 0))  # (H, W, 5)
            X = X.reshape(-1, 5)  # (H*W, 5)
            y = label.reshape(-1)  # (H*W,)
            
            # Find pixels where not all bands are 0 (non-cloud pixels)
            valid_pixels = ~np.all(X == 0, axis=1)
            
            # Add to dataset
            self.pixels.append(X[valid_pixels])
            self.labels.append(y[valid_pixels])
        
        # Concatenate all pixels and labels
        self.pixels = np.concatenate(self.pixels, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
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