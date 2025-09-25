# pip install torch torchvision numpy pillow tifffile segmentation-models-pytorch albumentations ttach

import os
import re
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import tifffile

from models import UNet

# --- Model Definition ---

def get_model():
    model = UNet(
        in_channels=5,
        out_channels=1,
    )
    return model

# --- Dataset Definition ---

def normalize_band(band):
    """Normalize a single band using mean and std."""
    band = band.astype(np.float32)
    mean = np.mean(band)
    std = np.std(band)
    if std > 0:
        return (band - mean) / std
    return band

class GlacierTestDataset(Dataset):
    def __init__(self, imagepath):
        self.imagepath = imagepath
        self.band_dirs = self.imagepath
        
        self.band_tile_map = {f"Band{i}": {} for i in range(1, 6)}
        self.tile_ids = []

        for i in range(1, 6):
            band_name = f"Band{i}"
            band_dir = self.band_dirs[band_name]
            for f in os.listdir(band_dir):
                if f.endswith(".tif"):
                    match = re.search(r"(\d{2}_\d{2})", f)
                    if match:
                        tile_id = match.group(1)
                        if tile_id not in self.tile_ids:
                            self.tile_ids.append(tile_id)
                        self.band_tile_map[band_name][tile_id] = f

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, idx):
        tile_id = self.tile_ids[idx]
        
        bands = []
        for i in range(1, 6):
            band_name = f"Band{i}"
            filename = self.band_tile_map[band_name][tile_id]
            band_file = os.path.join(self.band_dirs[band_name], filename)
            arr = tifffile.imread(band_file)
            arr = normalize_band(arr) # Normalize here
            bands.append(arr)
        
        image = np.stack(bands, axis=0).astype(np.float32) # Stack on axis=0 for (C, H, W)

        return image, self.band_tile_map["Band1"][tile_id]

# --- Main Prediction Function ---

def maskgeration(imagepath, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Data Loading ---
    # No transforms here, as normalization is done in GlacierTestDataset
    dataset = GlacierTestDataset(imagepath)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # --- Model Loading ---
    model = get_model()
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()
    
    # --- Prediction ---
    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        for images, image_files in dataloader:
            images = images.to(device)
            outputs = model(images) # No TTA
            preds = torch.sigmoid(outputs).cpu().numpy()
            
            for i in range(preds.shape[0]):
                pred_mask = (preds[i, 0] > 0.5).astype(np.uint8) * 255
                mask_image = Image.fromarray(pred_mask)
                mask_image.save(os.path.join(out_dir, image_files[i]))

# --- Main Function (as provided in the instructions) ---

# Do not update this section
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to test images folder")
    parser.add_argument("--masks", required=True, help="Path to masks folder (unused)")
    parser.add_argument("--output", required=True, help="Path to output predictions")
    args = parser.parse_args()

    # Build band → folder map
    imagepath = {}
    for band in os.listdir(args.data):
        band_path = os.path.join(args.data, band)
        if os.path.isdir(band_path):
            imagepath[band] = band_path

    print(f"Processing bands: {list(imagepath.keys())}")

    # Run mask generation and save predictions
    maskgeration(imagepath, args.output)

if __name__ == "__main__":
    main()