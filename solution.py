# pip install torch torchvision pillow tifffile numpy

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import tifffile
from collections import OrderedDict

# ==================================================================================
# --- Model Architecture ---
# ==================================================================================

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class MultiScaleUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(MultiScaleUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels * 2, 64)
        self.se1 = SEBlock(64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x_down = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_down_up = F.interpolate(x_down, size=x.shape[2:], mode='bilinear', align_corners=False)
        x_multi_scale = torch.cat([x, x_down_up], dim=1)
        x1 = self.inc(x_multi_scale)
        x1 = self.se1(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# ==================================================================================
# --- Inference Dataset and Utilities ---
# ==================================================================================

# Normalization stats from our training run
GLOBAL_STATS = {
    "0": (31920.07, 20553.20),
    "1": (27754.48, 20273.15),
    "2": (31145.41, 21443.17),
    "3": (8874.63, 2982.37),
    "4": (32289.35, 2997.46)
}

def normalize_band(band, mean, std):
    band = band.astype(np.float32)
    if std > 0:
        return (band - mean) / std
    return band - mean

class GlacierTestDataset(Dataset):
    def __init__(self, imagepath_dict):
        self.imagepath_dict = imagepath_dict
        self.band_dirs = self.imagepath_dict
        
        # Find all unique tile IDs from Band1
        self.tile_files = sorted(os.listdir(self.band_dirs["Band1"]))
        
    def __len__(self):
        return len(self.tile_files)

    def __getitem__(self, idx):
        tile_filename = self.tile_files[idx]
        
        bands = []
        for i in range(1, 6):
            band_name = f"Band{i}"
            band_file = os.path.join(self.band_dirs[band_name], tile_filename)
            arr = tifffile.imread(band_file)
            
            mean, std = GLOBAL_STATS[str(i-1)]
            arr = normalize_band(arr, mean, std)
            bands.append(arr)
        
        image = np.stack(bands, axis=0).astype(np.float32)
        return torch.from_numpy(image), tile_filename

# ==================================================================================
# --- Main Inference Function ---
# ==================================================================================

def tta_predict(model, x):
    """Test-Time Augmentation for more robust predictions."""
    x = x.to(next(model.parameters()).device)
    predictions = []
    
    # Original
    with torch.no_grad():
        pred = torch.sigmoid(model(x))
    predictions.append(pred)
    
    # Horizontal flip
    with torch.no_grad():
        pred_hflip = torch.sigmoid(model(torch.flip(x, dims=[3])))
    predictions.append(torch.flip(pred_hflip, dims=[3]))
    
    # Vertical flip
    with torch.no_grad():
        pred_vflip = torch.sigmoid(model(torch.flip(x, dims=[2])))
    predictions.append(torch.flip(pred_vflip, dims=[2]))
    
    return torch.stack(predictions).mean(dim=0)

def maskgeration(imagepath, out_dir):
    """
    Loads the trained model and generates segmentation masks for the test dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Loading ---
    model = MultiScaleUNet(n_channels=5, n_classes=1)
    
    # Load the state dict, handling the 'module.' prefix from DataParallel
    try:
        state_dict = torch.load("model.pth", map_location=device)
        if next(iter(state_dict)).startswith('module.'):
            print("Loading DataParallel model.")
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.'
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            print("Loading standard model.")
            model.load_state_dict(state_dict)
    except FileNotFoundError:
        print("ERROR: model.pth not found. Please ensure the model weights file is in the same directory.")
        return

    model.to(device)
    model.eval()

    # --- Data Loading ---
    dataset = GlacierTestDataset(imagepath)
    # Use num_workers=0 in submission environments for reliability
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0) 
    
    # --- Prediction Loop ---
    os.makedirs(out_dir, exist_ok=True)
    
    threshold = 0.5 # Use a standard 0.5 threshold for submission
    print(f"Starting prediction with threshold: {threshold}")

    with torch.no_grad():
        for images, filenames in dataloader:
            images = images.to(device)
            
            # Use Test-Time Augmentation
            preds = tta_predict(model, images).cpu().numpy()
            
            for i in range(preds.shape[0]):
                pred_mask = (preds[i, 0] > threshold).astype(np.uint8) * 255
                mask_image = Image.fromarray(pred_mask)
                mask_image.save(os.path.join(out_dir, filenames[i]))
    
    print("Prediction complete.")


# ==================================================================================
# --- Boilerplate Main Function (Do Not Modify) ---
# ==================================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to test images folder")
    parser.add_argument("--masks", required=True, help="Path to masks folder (unused)")
    parser.add_argument("--out", required=True, help="Path to output predictions")
    args = parser.parse_args()

    imagepath = {}
    for band in os.listdir(args.data):
        band_path = os.path.join(args.data, band)
        if os.path.isdir(band_path):
            imagepath[band] = band_path

    print(f"Processing bands: {list(imagepath.keys())}")
    maskgeration(imagepath, args.out)

if __name__ == "__main__":
    main()
