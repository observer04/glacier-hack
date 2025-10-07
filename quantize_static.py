# ==================================================================================
# QUANTIZE_STATIC.PY (Complete)
#
# This script performs post-training static quantization on an existing model.
# ==================================================================================

import os
import re
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models import resnet101
from torchvision.models.segmentation.deeplabv3 import ASPP
from collections import OrderedDict

# ==================================================================================
# --- 1. MODEL & DATA DEFINITIONS (Must match training) ---
# ==================================================================================

class DeepLabV3Plus(nn.Module):
    def __init__(self, n_classes=1, n_channels=5):
        super().__init__()
        backbone = resnet101(weights=None, replace_stride_with_dilation=[False, True, True])
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self._modify_backbone_input(backbone, n_channels)

        self.encoder_stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.encoder_layer1 = backbone.layer1
        self.encoder_layer2 = backbone.layer2
        self.encoder_layer3 = backbone.layer3
        self.encoder_layer4 = backbone.layer4
        
        self.aspp = ASPP(in_channels=2048, atrous_rates=[12, 24, 36], out_channels=256)

        self.decoder_conv1 = nn.Conv2d(256, 48, kernel_size=1, bias=False)
        self.decoder_bn1 = nn.BatchNorm2d(48)
        self.decoder_relu1 = nn.ReLU()

        self.decoder_conv2 = nn.Conv2d(256 + 48, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder_bn2 = nn.BatchNorm2d(256)
        self.decoder_relu2 = nn.ReLU()

        self.final_conv = nn.Conv2d(256, n_classes, kernel_size=1, stride=1)

    def _modify_backbone_input(self, backbone, n_channels):
        new_conv = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        backbone.conv1 = new_conv

    def forward(self, x):
        x = self.quant(x)
        input_shape = x.shape[-2:]
        x = self.encoder_stem(x)
        low_level_features = self.encoder_layer1(x)
        x = self.encoder_layer2(low_level_features)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)

        x = self.aspp(x)
        x = F.interpolate(x, size=low_level_features.shape[-2:], mode='bilinear', align_corners=False)

        low_level_features = self.decoder_conv1(low_level_features)
        low_level_features = self.decoder_bn1(low_level_features)
        low_level_features = self.decoder_relu1(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.decoder_conv2(x)
        x = self.decoder_bn2(x)
        x = self.decoder_relu2(x)
        
        x = self.final_conv(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        x = self.dequant(x)
        return x

def get_tile_id(filename):
    match = re.search(r'(\d{2}_\d{2})', filename)
    return match.group(1) if match else None

def _find_label_path(label_dir, tile_id):
    for name in [f"Y{tile_id}.tif", f"Y_output_resized_{tile_id}.tif"]:
        path = os.path.join(label_dir, name)
        if os.path.exists(path):
            return path
    return None

class GlacierTileDataset(Dataset):
    def __init__(self, data_dir, tile_ids, global_stats):
        self.data_dir = data_dir
        self.tile_ids = tile_ids
        self.stats = global_stats
        self.band_dirs = [os.path.join(data_dir, f"Band{i}") for i in range(1, 6)]
        self.label_dir = os.path.join(data_dir, "label")
        self.band_tile_map = {i: {} for i in range(5)}
        for band_idx, band_dir in enumerate(self.band_dirs):
            for f in os.listdir(band_dir):
                if f.endswith(".tif"):
                    tid = get_tile_id(f)
                    if tid:
                        self.band_tile_map[band_idx][tid] = f

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, index):
        tid = self.tile_ids[index]
        bands = []
        for b in range(5):
            fp = os.path.join(self.band_dirs[b], self.band_tile_map[b][tid])
            arr = np.array(Image.open(fp), dtype=np.float32)
            bands.append(np.nan_to_num(arr))
        x = np.stack(bands, axis=-1)

        label_path = _find_label_path(self.label_dir, tid)
        if label_path is None: raise FileNotFoundError(f"Could not find label for tile ID: {tid}")
        y = np.array(Image.open(label_path), dtype=np.float32)[..., np.newaxis]

        x = TF.to_tensor(x)
        y = TF.to_tensor(y)
        x = TF.normalize(x, self.stats['means'], self.stats['stds'])
        y = (y > 0).float()
        return x, y.squeeze(0)

# ==================================================================================
# --- 2. SCRIPT LOGIC ---
# ==================================================================================

if __name__ == '__main__':
    # --- Configuration ---
    DATA_DIR = "/kaggle/working/Train"
    WORK_DIR = "/kaggle/working/"
    STATS_PATH = os.path.join(WORK_DIR, "global_stats.json")
    INPUT_MODEL_PATH = os.path.join(WORK_DIR, 'model.pth')
    OUTPUT_MODEL_PATH = os.path.join(WORK_DIR, 'model_quantized.pth')

    # --- Load Model ---
    device = torch.device("cpu")
    model = DeepLabV3Plus()
    model.load_state_dict(torch.load(INPUT_MODEL_PATH, map_location=device))
    model.eval()
    print("Loaded ensembled float32 model.")

    # --- Prepare for Static Quantization ---
    print("Preparing model for static quantization...")
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_prepared = torch.quantization.prepare(model)

    # --- Calibrate the Model ---
    print("Calibrating model with a few batches of data...")
    with open(STATS_PATH, 'r') as f: global_stats = json.load(f)
    if 'means' not in global_stats:
        means = [global_stats[str(i)][0] for i in range(5)]; stds = [global_stats[str(i)][1] for i in range(5)]
        global_stats = {'means': means, 'stds': stds}
    
    label_dir = os.path.join(DATA_DIR, "label")
    all_tile_ids = sorted([get_tile_id(f) for f in os.listdir(label_dir) if f.endswith(".tif")])
    calib_dataset = GlacierTileDataset(DATA_DIR, all_tile_ids, global_stats)
    calib_loader = DataLoader(calib_dataset, batch_size=4, shuffle=True)
    
    with torch.no_grad():
        for i, (images, _) in enumerate(calib_loader):
            if i >= 5: # Use 5 batches for calibration
                break
            model_prepared(images)

    # --- Convert to Quantized Model ---
    print("Converting model to quantized version...")
    model_quantized = torch.quantization.convert(model_prepared)

    # --- Save the Quantized Model ---
    print(f"Saving quantized model to {OUTPUT_MODEL_PATH}")
    torch.save(model_quantized.state_dict(), OUTPUT_MODEL_PATH)

    original_size = os.path.getsize(INPUT_MODEL_PATH) / (1024 * 1024)
    quantized_size = os.path.getsize(OUTPUT_MODEL_PATH) / (1024 * 1024)

    print("\n--- Quantization Complete ---")
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
