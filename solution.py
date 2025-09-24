# pip install torch torchvision numpy pillow tifffile

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

# Try to import UNet from repo models for exact architecture match
try:
    from models import UNet as RepoUNet
    _HAS_REPO_UNET = True
except Exception:
    _HAS_REPO_UNET = False

class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ model for semantic segmentation."""
    
    def __init__(self, in_channels=5, out_channels=1):
        super(DeepLabV3Plus, self).__init__()
        
        # Use a lighter backbone for model size constraints
        self.backbone = nn.Sequential(
            # Initial conv to increase channels
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Downsample blocks
            self._make_layer(64, 128, stride=2),
            self._make_layer(128, 256, stride=2),
            self._make_layer(256, 512, stride=2),
            self._make_layer(512, 512, stride=1, dilation=2),
        )
        
        self.low_level_features = nn.Sequential(
            nn.Conv2d(128, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.aspp = self._build_aspp(512, 256)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def _make_layer(self, in_channels, out_channels, stride=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                     padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, 
                     dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _build_aspp(self, in_channels, out_channels):
        # 1x1 convolution
        self.aspp_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Atrous convolutions with different rates
        self.aspp_3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.aspp_3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.aspp_3x3_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.aspp_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Project to reduce channels
        self.aspp_project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        return nn.ModuleList([self.aspp_1x1, self.aspp_3x3_1, self.aspp_3x3_2, self.aspp_3x3_3, self.aspp_pool])
        
    def forward(self, x):
        # Save input size for final upsampling
        input_size = x.shape[-2:]
        
        # Extract features
        x1 = self.backbone[0:3](x)  # Low-level features
        x = self.backbone[3:](x1)   # High-level features
        
        # Apply ASPP modules
        aspp_1x1_out = self.aspp_1x1(x)
        aspp_3x3_1_out = self.aspp_3x3_1(x)
        aspp_3x3_2_out = self.aspp_3x3_2(x)
        aspp_3x3_3_out = self.aspp_3x3_3(x)
        
        # Apply global pooling module separately (needs special handling for interpolation)
        global_pool = self.aspp_pool(x)
        global_pool = F.interpolate(global_pool, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # Concatenate all ASPP results
        x = torch.cat([aspp_1x1_out, aspp_3x3_1_out, aspp_3x3_2_out, aspp_3x3_3_out, global_pool], dim=1)
        
        # Apply projection
        x = self.aspp_project(x)
        
        # Process low-level features
        x1 = self.low_level_features(x1)
        
        # Upsample high-level features
        x = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        
        # Concatenate low and high level features
        x = torch.cat([x, x1], dim=1)
        
        # Decoder
        x = self.decoder(x)
        
        # Final upsampling to original image size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return self.sigmoid(x)

# Fallback to simpler model if needed
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

def _get_threshold():
    # Allow env override; default to tuned 0.60
    try:
        return float(os.environ.get("SOLUTION_THRESHOLD", "0.6"))
    except Exception:
        return 0.6

def maskgeration(imagepath: dict, out_dir: str):
    """Generate binary masks for glacier segmentation.
    
    Args:
        imagepath: Dictionary mapping band names to directories
        out_dir: Directory to save output masks
    """
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Config: allow model path/type/threshold via environment
    model_path = os.getenv("SOLUTION_MODEL_PATH", "model.pth")
    model_pref = os.getenv("SOLUTION_MODEL_TYPE", "auto").lower()  # auto|unet|deeplabv3plus|pixelann
    threshold = _get_threshold()

    def _try_load(model_ctor, label: str):
        m = model_ctor()
        m.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Using {label} model from {model_path} with threshold {threshold}")
        return m

    model = None
    use_segmentation = False  # True for UNet/DeepLab

    # Determine load order
    load_order = []
    if model_pref == "unet":
        load_order = ["unet"]
    elif model_pref == "deeplabv3plus":
        load_order = ["deeplabv3plus"]
    elif model_pref == "pixelann":
        load_order = ["pixelann"]
    else:
        # Auto: prefer UNet -> DeepLab -> PixelANN
        load_order = ["unet", "deeplabv3plus", "pixelann"]

    last_err = None
    for kind in load_order:
        try:
            if kind == "unet":
                if not _HAS_REPO_UNET:
                    raise RuntimeError("UNet class not available in solution.py environment")
                model = _try_load(lambda: RepoUNet(in_channels=5, out_channels=1), "UNet")
                use_segmentation = True
                break
            elif kind == "deeplabv3plus":
                model = _try_load(lambda: DeepLabV3Plus(in_channels=5, out_channels=1), "DeepLabV3+")
                use_segmentation = True
                break
            else:
                model = _try_load(lambda: PixelANN(in_channels=5, hidden_dims=[32, 64, 128, 64, 32], dropout_rate=0.0), "PixelANN")
                use_segmentation = False
                break
        except Exception as e:
            last_err = e
            print(f"Could not load {kind}: {e}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load any model from {model_path}. Last error: {last_err}")
    
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
            band_arrays.append(arr_normalized)
        
        if not band_arrays or len(band_arrays) != len(imagepath):
            continue
        
        if not band_arrays or len(band_arrays) != len(imagepath) or H is None or W is None:
            # Skip tiles with missing bands
            continue
            
        if use_segmentation:
            # Stack bands to create an image tensor (C, H, W)
            X = np.stack(band_arrays, axis=0)
            
            # Convert to tensor
            X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            
            # Process with DeepLabV3+
            with torch.no_grad():
                pred = model(X_tensor).squeeze()
                
            # Create binary mask
            full_mask = (pred > threshold).cpu().numpy().astype(np.uint8) * 255
            
        else:
            # For PixelANN, we need to flatten the bands
            band_arrays_flat = [band.flatten() for band in band_arrays]
            
            # Stack bands to create feature matrix
            X = np.stack(band_arrays_flat, axis=1)  # (H*W, 5)
            
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
                        all_preds.append((preds > threshold).int().numpy())  # threshold corresponds to glacier pixels
                
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