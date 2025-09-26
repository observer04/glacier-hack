# pip install torch torchvision numpy pillow tifffile scikit-learn tqdm

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import tifffile

from models import UNet, EfficientUNet, DeepLabV3Plus

# --- Model Definition ---

def get_model():
    """Load the best available trained model with priority order."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Priority order: EfficientUNet > UNet > DeepLabV3Plus
    # Check both local and Google Drive paths
    model_paths = [
        ("model.pth", None),  # Main submission model
        ("/content/drive/MyDrive/glacier_hack/model.pth", None),  # Google Drive submission model
        ("/content/drive/MyDrive/glacier_hack/models/efficientunet_tversky/efficientunet_best.pth", "efficientunet"),
        ("/content/drive/MyDrive/glacier_hack/models/multiscale_efficientunet/efficientunet_multiscale_best.pth", "efficientunet"),
        ("/content/drive/MyDrive/glacier_hack/models/ensemble_full/model_0_efficientunet_best.pth", "efficientunet"),
        ("models/efficientunet_tversky/efficientunet_best.pth", "efficientunet"),
        ("models/multiscale_efficientunet/efficientunet_multiscale_best.pth", "efficientunet"),
        ("models/ensemble_full/model_0_efficientunet_best.pth", "efficientunet"),
        ("models/unet_adaptive/unet_best.pth", "unet"),
        ("models/deeplabv3plus_boundary/deeplabv3plus_best.pth", "deeplabv3plus"),
    ]
    
    for model_path, model_type in model_paths:
        if os.path.exists(model_path):
            print(f"Found model: {model_path}")
            
            # Determine model type from path if not specified
            if model_type is None:
                if "efficientunet" in model_path.lower():
                    model_type = "efficientunet"
                elif "deeplabv3" in model_path.lower():
                    model_type = "deeplabv3plus"
                else:
                    model_type = "unet"  # default
            
            # Create model
            if model_type == "efficientunet":
                model = EfficientUNet(in_channels=5, out_channels=1)
            elif model_type == "deeplabv3plus":
                model = DeepLabV3Plus(in_channels=5, out_channels=1)
            else:
                model = UNet(in_channels=5, out_channels=1)
            
            # Load weights
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded checkpoint with MCC: {checkpoint.get('best_mcc', 'unknown')}")
                else:
                    model.load_state_dict(checkpoint)
                
                model.to(device)
                model.eval()
                return model, model_type
                
            except Exception as e:
                print(f"Failed to load {model_path}: {e}")
                continue
    
    # Fallback to default UNet if no models found
    print("No pre-trained models found, using default UNet")
    model = UNet(in_channels=5, out_channels=1)
    model.to(device)
    model.eval()
    return model, "unet"

# --- Dataset Definition ---

def normalize_band(band, global_stats=None):
    """Normalize a single band using mean and std."""
    band = band.astype(np.float32)
    
    if global_stats is not None:
        mean, std = global_stats
        if std > 0:
            return (band - mean) / std
        return band - mean
    else:
        # Fallback to per-tile normalization
        mean = np.mean(band)
        std = np.std(band)
        if std > 0:
            return (band - mean) / std
        return band

# Pre-computed global statistics (computed from training data)
GLOBAL_STATS = {
    0: (1842.5, 456.2),  # Band1 (Blue)
    1: (1654.3, 398.7),  # Band2 (Green) 
    2: (1423.8, 367.9),  # Band3 (Red)
    3: (2156.7, 512.4),  # Band4 (SWIR)
    4: (287.6, 8.3),     # Band5 (TIR)
}

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
            
            # Use global normalization if available
            band_idx = i - 1  # Convert to 0-based index
            global_stats = GLOBAL_STATS.get(band_idx)
            arr = normalize_band(arr, global_stats)
            bands.append(arr)
        
        image = np.stack(bands, axis=0).astype(np.float32) # Stack on axis=0 for (C, H, W)

        return image, self.band_tile_map["Band1"][tile_id]

# --- Main Prediction Function ---

def tta_predict(model, x):
    """Test-Time Augmentation for more robust predictions."""
    predictions = []
    
    # Original
    with torch.no_grad():
        pred = torch.sigmoid(model(x))
        predictions.append(pred)
        
        # Horizontal flip
        pred_hflip = torch.sigmoid(model(torch.flip(x, dims=[3])))
        pred_hflip = torch.flip(pred_hflip, dims=[3])
        predictions.append(pred_hflip)
        
        # Vertical flip
        pred_vflip = torch.sigmoid(model(torch.flip(x, dims=[2])))
        pred_vflip = torch.flip(pred_vflip, dims=[2])
        predictions.append(pred_vflip)
        
        # Horizontal + Vertical flip
        pred_hvflip = torch.sigmoid(model(torch.flip(x, dims=[2, 3])))
        pred_hvflip = torch.flip(pred_hvflip, dims=[2, 3])
        predictions.append(pred_hvflip)
    
    # Average predictions
    return torch.stack(predictions).mean(dim=0)

def maskgeration(imagepath, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Data Loading ---
    dataset = GlacierTestDataset(imagepath)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # --- Model Loading ---
    model, model_type = get_model()
    print(f"Using {model_type} model")
    
    # --- Prediction with TTA ---
    os.makedirs(out_dir, exist_ok=True)
    
    # Use TTA for better predictions
    use_tta = True
    # Adaptive threshold based on model type (from training experience)
    if model_type == "efficientunet":
        threshold = 0.55  # EfficientUNet typically works better with slightly higher threshold
    else:
        threshold = 0.5
        
    print(f"Using threshold: {threshold}, TTA: {use_tta}")
    
    with torch.no_grad():
        for images, image_files in dataloader:
            images = images.to(device)
            
            if use_tta:
                preds = tta_predict(model, images).cpu().numpy()
            else:
                outputs = model(images)
                preds = torch.sigmoid(outputs).cpu().numpy()
            
            for i in range(preds.shape[0]):
                pred_mask = (preds[i, 0] > threshold).astype(np.uint8) * 255
                mask_image = Image.fromarray(pred_mask)
                mask_image.save(os.path.join(out_dir, image_files[i]))

# --- Main Function (as provided in the instructions) ---

# Do not update this section
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to test images folder")
    parser.add_argument("--masks", required=True, help="Path to masks folder (unused)")
    parser.add_argument("--out", required=True, help="Path to output predictions")
    args = parser.parse_args()

    # Build band → folder map
    imagepath = {}
    for band in os.listdir(args.data):
        band_path = os.path.join(args.data, band)
        if os.path.isdir(band_path):
            imagepath[band] = band_path

    print(f"Processing bands: {list(imagepath.keys())}")

    # Run mask generation and save predictions
    maskgeration(imagepath, args.out)

if __name__ == "__main__":
    main()