# pip install torch torchvision scipy pillow numpy

"""
OPTIMIZED ENSEMBLE SOLUTION - Combines 3 best fold models
- Fold 2: 0.7216 MCC
- Fold 5: 0.7131 MCC
- Fold 1: 0.6988 MCC
Ensemble mean: 0.7112 MCC

OPTIMIZED PARAMETERS (tested on validation):
- Threshold: 0.55 (tested MCC=0.7490)
- Min size: 50 pixels
Expected test MCC: 0.72-0.75
"""

import os
import re
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from scipy.ndimage import label as scipy_label
from torchvision.models import resnet18

# ==================================================================================
# MODEL ARCHITECTURE - Same as training
# ==================================================================================

class ResNet18UNet(nn.Module):
    """U-Net with ResNet18 encoder, modified for 5 input channels"""
    
    def __init__(self, n_classes=1, n_channels=5):
        super().__init__()
        
        # Load ResNet18 architecture WITHOUT pretrained weights
        # (Platform has read-only filesystem, can't download weights)
        # Our trained model already has the learned weights
        resnet = resnet18(weights=None)
        
        # Modify first conv layer for 5 input channels
        self.encoder_input = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Note: No weight initialization needed - we'll load trained weights from checkpoint
        
        # Encoder layers
        self.encoder_bn = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool
        
        self.encoder1 = resnet.layer1  # 64 channels
        self.encoder2 = resnet.layer2  # 128 channels
        self.encoder3 = resnet.layer3  # 256 channels
        self.encoder4 = resnet.layer4  # 512 channels
        
        # Decoder
        self.decoder4 = self._make_decoder_block(512, 256)
        self.decoder3 = self._make_decoder_block(256 + 256, 128)
        self.decoder2 = self._make_decoder_block(128 + 128, 64)
        self.decoder1 = self._make_decoder_block(64 + 64, 64)
        
        # Final upsampling to original resolution
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Output head
        self.output = nn.Conv2d(32, n_classes, kernel_size=1)
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        x = self.encoder_input(x)
        x = self.encoder_bn(x)
        x = self.encoder_relu(x)
        x0 = x
        
        x = self.encoder_maxpool(x)
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        
        # Decoder with skip connections
        d4 = self.decoder4(x4)
        d3 = self.decoder3(torch.cat([d4, x3], dim=1))
        d2 = self.decoder2(torch.cat([d3, x2], dim=1))
        d1 = self.decoder1(torch.cat([d2, x1], dim=1))
        
        # Final upsampling
        out = self.final_upsample(d1)
        out = self.output(out)
        
        return out

# ==================================================================================
# NORMALIZATION STATISTICS
# ==================================================================================

# Pre-computed from training data (ALL 25 tiles)
BAND_MEANS = [23264.8359, 22882.7227, 22640.1055, 6610.3862, 23520.8379]
BAND_STDS = [22887.9688, 22444.9688, 22843.4453, 4700.4126, 14073.5098]

# ==================================================================================
# HELPER FUNCTIONS
# ==================================================================================

def get_tile_id(filename):
    """
    Extract ONLY numeric part of tile ID (e.g., '001' not 'img001')
    
    Platform expects numeric-only IDs to avoid corrupted filenames like 'imgimg001.tif'
    Examples:
        'img001.tif' -> '001'
        'stacked_02_07.tif' -> '02_07' (fallback for training data)
    """
    # First try to extract just the numeric part
    match = re.search(r'img(\d+)', filename, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Fallback: try XX_YY pattern (for training data compatibility)
    match = re.search(r'(\d{2}_\d{2})', filename)
    if match:
        return match.group(1)
    
    # Last resort: extract any sequence of digits
    match = re.search(r'(\d+)', filename)
    if match:
        return match.group(1)
    
    # If no digits found, use basename without extension
    return os.path.splitext(os.path.basename(filename))[0]

def load_bands_from_dict(imagepath_dict):
    """
    Load all 5 bands from band folder dict (platform format)
    
    Args:
        imagepath_dict: Dict like {"Band1": "/path/to/Band1", "Band2": ...}
    
    Returns:
        dict: {tile_id: bands_array (5, H, W)} for each tile
    """
    # Expected bands in order with their prefixes
    band_info = [
        ("Band1", "B2"),   # Band1 folder contains B2_* files
        ("Band2", "B3"),   # Band2 folder contains B3_* files
        ("Band3", "B4"),   # Band3 folder contains B4_* files
        ("Band4", "B6"),   # Band4 folder contains B6_* files
        ("Band5", "B10")   # Band5 folder contains B10_* files
    ]
    
    # Get all tile IDs from Band1
    band1_path = imagepath_dict.get("Band1")
    if not band1_path or not os.path.exists(band1_path):
        raise ValueError(f"Band1 path not found: {band1_path}")
    
    # Get all files and extract tile IDs
    band1_files = sorted([f for f in os.listdir(band1_path) if f.endswith('.tif')])
    
    if len(band1_files) == 0:
        raise ValueError(f"No .tif files found in {band1_path}")
    
    # Extract tile IDs from Band1 files
    tile_ids = [get_tile_id(f) for f in band1_files]
    
    results = {}
    
    for tile_file, tile_id in zip(band1_files, tile_ids):
        bands = []
        
        # Extract tile pattern from Band1 filename (e.g., "02_07" from "B2_B2_masked_02_07.tif")
        tile_pattern = None
        match = re.search(r'(\d{2}_\d{2})', tile_file)
        if match:
            tile_pattern = match.group(1)
        
        # Load each band with its specific prefix
        for band_name, band_prefix in band_info:
            band_path = imagepath_dict.get(band_name)
            if not band_path:
                raise ValueError(f"Missing {band_name} in imagepath dict")
            
            # Try different filename patterns
            possible_filenames = []
            
            if tile_pattern:
                # Training data pattern: B2_B2_masked_02_07.tif
                possible_filenames.append(f"{band_prefix}_{band_prefix}_masked_{tile_pattern}.tif")
            
            # Test data pattern: img001.tif (same name across all bands)
            possible_filenames.append(tile_file)
            
            # Try img{tile_id}.tif pattern
            possible_filenames.append(f"img{tile_id}.tif")
            
            # Try to load from any matching pattern
            band_loaded = False
            for filename in possible_filenames:
                file_path = os.path.join(band_path, filename)
                if os.path.exists(file_path):
                    band = np.array(Image.open(file_path), dtype=np.float32)
                    if band.ndim == 3:
                        band = band[:, :, 0]
                    bands.append(band)
                    band_loaded = True
                    break
            
            if not band_loaded:
                raise FileNotFoundError(
                    f"Band file not found for {band_name} (tile {tile_id})\n"
                    f"Tried: {possible_filenames}\n"
                    f"In directory: {band_path}\n"
                    f"Available files: {os.listdir(band_path)[:5]}"
                )
        
        results[tile_id] = np.stack(bands, axis=0)  # (5, H, W)
    
    return results

def normalize_bands(bands):
    """Normalize bands using training statistics"""
    normalized = bands.copy()
    for i in range(5):
        normalized[i] = (bands[i] - BAND_MEANS[i]) / BAND_STDS[i]
    return normalized

def post_process_mask(mask, min_size=50):
    """Remove small isolated predictions (OPTIMIZED via validation testing)"""
    # Convert to binary (OPTIMIZED threshold=0.55 tested on validation data)
    binary_mask = (mask > 0.55).astype(np.uint8)
    
    # Remove small connected components
    labeled_array, num_features = scipy_label(binary_mask)
    
    for region_id in range(1, num_features + 1):
        region_mask = labeled_array == region_id
        if region_mask.sum() < min_size:
            binary_mask[region_mask] = 0
    
    return binary_mask

def load_model(model_path, device):
    """Load a single model from checkpoint"""
    model = ResNet18UNet(n_classes=1, n_channels=5)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle DataParallel wrapper
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    return model

# ==================================================================================
# MAIN INFERENCE FUNCTION - ENSEMBLE VERSION
# ==================================================================================

def maskgeration(imagepath, out_dir):
    """
    Generate glacier mask predictions using ENSEMBLE of 3 models
    
    PLATFORM QUIRK: Despite the name 'out_dir', this is actually the model_path!
    Models are stored in a COMBINED checkpoint with keys: fold5, fold2, fold1
    
    Args:
        imagepath: Dict of band folders {"Band1": path, "Band2": path, ...}
        out_dir: ACTUALLY THE MODEL PATH (platform quirk!)
    
    Returns:
        dict: {tile_id: numpy_array} - Binary masks with 0/1 values
              Platform saves these arrays as .tif files automatically
    """
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load combined checkpoint with all 3 models
    print(f"[ENSEMBLE] Loading combined checkpoint from: {out_dir}")
    combined_checkpoint = torch.load(out_dir, map_location=device)
    
    # Check if it's a combined checkpoint or single model
    if isinstance(combined_checkpoint, dict) and any(f'fold{i}' in combined_checkpoint for i in range(1, 6)):
        # Combined checkpoint with multiple folds (auto-detect fold numbers)
        fold_names = [k for k in combined_checkpoint.keys() if k.startswith('fold')]
        print(f"[ENSEMBLE] Found combined checkpoint with {len(fold_names)} models: {fold_names}")
        
        models = []
        for fold_name in sorted(fold_names):  # Sort for consistent ordering
            print(f"  - Loading {fold_name}...")
            state_dict = combined_checkpoint[fold_name]
            
            # Create model and load weights
            model = ResNet18UNet(n_classes=1, n_channels=5)
            
            # Remove 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                new_state_dict[name] = v
            
            model.load_state_dict(new_state_dict)
            model = model.to(device)
            model.eval()
            models.append(model)
    else:
        # Single model fallback
        print(f"[ENSEMBLE] Loading single model (fallback)")
        model = load_model(out_dir, device)
        models = [model]
    
    # Load all tiles from band folders
    tiles_data = load_bands_from_dict(imagepath)
    
    # Generate predictions for all tiles
    results = {}
    
    print(f"[ENSEMBLE] Processing {len(tiles_data)} tiles...")
    
    for tile_id, bands in tiles_data.items():
        # Normalize
        bands_normalized = normalize_bands(bands)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(bands_normalized).unsqueeze(0).float()  # (1, 5, H, W)
        image_tensor = image_tensor.to(device)
        
        # Inference with ALL models (NO TTA - causes timeout!)
        predictions = []
        with torch.no_grad():
            for model in models:
                output = model(image_tensor)
                prediction = torch.sigmoid(output).squeeze().cpu().numpy()  # (H, W)
                predictions.append(prediction)
        
        # Average predictions from all models
        avg_prediction = np.mean(predictions, axis=0)
        
        # Post-process (with tuned parameters)
        final_mask = post_process_mask(avg_prediction, min_size=50)
        
        # Debug: Check prediction stats
        glacier_pct = 100 * final_mask.mean()
        print(f"[ENSEMBLE] Tile {tile_id}: {final_mask.shape}, {final_mask.sum()} glacier pixels ({glacier_pct:.2f}%)")
        
        # Store result (platform expects binary 0/1 numpy array)
        results[tile_id] = final_mask
    
    print(f"[ENSEMBLE] Returning {len(results)} predictions")
    print(f"[ENSEMBLE] Tile IDs: {list(results.keys())[:10]}")
    
    return results

# ==================================================================================
# TESTING (OPTIONAL - Can be removed for final submission)
# ==================================================================================

if __name__ == '__main__':
    """
    Test the inference function locally
    
    Note: Platform will import this as a module and call maskgeration() directly
    """
    
    # Example test with local data structure
    test_data_path = "/home/observer/projects/glacier-hack/Train"
    test_model_path = "/home/observer/projects/glacier-hack"  # Directory with fold models
    
    # Build imagepath dict (same format as platform)
    imagepath = {}
    for band in ["Band1", "Band2", "Band3", "Band4", "Band5"]:
        band_path = os.path.join(test_data_path, band)
        if os.path.exists(band_path):
            imagepath[band] = band_path
    
    if imagepath and os.path.exists(test_model_path):
        print(f"Testing inference on {len(imagepath)} bands")
        print(f"Model directory: {test_model_path}")
        
        # Call maskgeration (remember: out_dir is actually model_path!)
        results = maskgeration(imagepath, test_model_path)
        
        print(f"\n✓ Generated {len(results)} predictions:")
        for tile_id, mask in list(results.items())[:5]:
            glacier_pct = 100 * mask.mean()
            print(f"  Tile {tile_id}: {mask.shape}, "
                  f"{mask.sum():,} glacier pixels ({glacier_pct:.2f}%)")
    else:
        print("Test data not found. Skipping local test.")
        print(f"Looking for bands in: {test_data_path}")
        print(f"Looking for models: {test_model_path}")
        print("\nThis is OK - the platform will provide the correct paths.")
