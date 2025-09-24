# pip install torch torchvision numpy pillow tifffile

import os
import re
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from tqdm import tqdm

class PixelANN(nn.Module):
    """Simple pixel-wise ANN for glacier segmentation."""
    
    def __init__(self, in_channels=5, hidden_dims=[32, 64, 128, 64, 32], dropout_rate=0.0):
        super().__init__()
        
        layers = []
        prev_dim = in_channels
        
        # Build hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def get_tile_id(filename):
    """Extract tile ID from filename."""
    match = re.search(r"img(\d+)\.tif", filename)
    if match:
        return match.group(1)
    
    match = re.search(r"(\d{2}_\d{2})", filename)
    return match.group(1) if match else None

def normalize_band(band_data):
    """Normalize a single band using mean and std."""
    band_data = band_data.astype(np.float32)
    mean = np.mean(band_data)
    std = np.std(band_data)
    if std > 0:
        return (band_data - mean) / std
    return band_data

def maskgeration(imagepath, out_dir):
    """Generate binary masks for glacier segmentation.
    
    Args:
        imagepath: Dictionary mapping band names to directories
        out_dir: Directory to save output masks
    """
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Load model
    model = PixelANN(in_channels=5, hidden_dims=[32, 64, 128, 64, 32], dropout_rate=0.0)
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    
    # Map band -> tile_id -> filename
    band_tile_map = {band: {} for band in imagepath}
    for band, folder in imagepath.items():
        if not os.path.exists(folder):
            continue
            
        files = os.listdir(folder)
        
        for f in files:
            if f.endswith(".tif"):
                tile_id = get_tile_id(f)
                if tile_id:
                    band_tile_map[band][tile_id] = f
    
    # Process each tile
    ref_band = sorted(imagepath.keys())[0]
    tile_ids = sorted(band_tile_map[ref_band].keys())
    
    for tile_id in tqdm(tile_ids, desc="Processing tiles"):
        # Load all bands
        band_arrays = []
        H, W = None, None
        
        for band_name in sorted(imagepath.keys()):
            if tile_id not in band_tile_map[band_name]:
                continue
                
            file_path = os.path.join(
                imagepath[band_name], 
                band_tile_map[band_name][tile_id]
            )
            
            if not os.path.exists(file_path):
                continue
                
            arr = np.array(Image.open(file_path))
            if arr.ndim == 3:
                arr = arr[..., 0]
                
            H, W = arr.shape
            
            # Normalize the band
            arr_normalized = normalize_band(arr)
            band_arrays.append(arr_normalized.flatten())
        
        if not band_arrays or len(band_arrays) != len(imagepath):
            continue
            
        # Stack bands to create feature matrix
        X = np.stack(band_arrays, axis=1)  # (H*W, 5)
        
        # Create cloud mask (all zeros)
        cloud_mask = np.all(X == 0, axis=1)
        
        # Process non-cloud pixels
        X_valid = X[~cloud_mask]
        
        if X_valid.shape[0] > 0:
            # Process in batches to avoid memory issues
            batch_size = 10000
            all_preds = []
            
            for i in range(0, X_valid.shape[0], batch_size):
                batch = torch.tensor(X_valid[i:i+batch_size], dtype=torch.float32)
                with torch.no_grad():
                    preds = model(batch).squeeze()
                    all_preds.append((preds < 0.5).int().numpy())  # Note: < 0.5 because 1 is glacier
            
            # Combine batches
            preds = np.concatenate(all_preds)
            
            # Create full mask (0=non-glacier, 255=glacier)
            full_mask = np.zeros(H * W, dtype=np.uint8)
            full_mask[~cloud_mask] = preds
            full_mask = full_mask.reshape(H, W) * 255
        else:
            # All pixels are cloud, create empty mask
            full_mask = np.zeros((H, W), dtype=np.uint8)
        
        # Save the mask
        output_path = os.path.join(out_dir, f"{tile_id}.tif")
        Image.fromarray(full_mask).save(output_path)

# The main function will be provided by the evaluation system
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate glacier segmentation masks")
    parser.add_argument("--imagepath", type=str, required=True, help="Path to input image directory")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    
    args = parser.parse_args()
    
    # Map band directories
    imagepath = {
        f"Band{i}": os.path.join(args.imagepath, f"Band{i}") for i in range(1, 6)
    }
    
    # Generate masks
    maskgeration(imagepath, args.output)

if __name__ == "__main__":
    main()