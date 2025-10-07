#!/usr/bin/env python3
"""
Detailed analysis: Track what happens at each processing step
"""

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from solution import ResNet18UNet, normalize_bands

# Test on one tile
TRAIN_DIR = "/home/observer/projects/glacier-hack/Train"
MODEL_PATH = "/home/observer/projects/glacier-hack/model.pth"
OUTPUT_DIR = "/home/observer/projects/glacier-hack/sanity_check_output"

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18UNet(n_classes=1, n_channels=5)
checkpoint = torch.load(MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint
new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model = model.to(device)
model.eval()

# Load one tile
tile_file = "B2_B2_masked_02_07.tif"
bands = []
for folder, prefix in zip(["Band1", "Band2", "Band3", "Band4", "Band5"],
                          ["B2", "B3", "B4", "B6", "B10"]):
    file_path = os.path.join(TRAIN_DIR, folder, tile_file.replace("B2_", f"{prefix}_"))
    band = np.array(Image.open(file_path), dtype=np.float32)
    if band.ndim == 3:
        band = band[:, :, 0]
    bands.append(band)

bands = np.stack(bands, axis=0)

# Load ground truth
gt_path = os.path.join(TRAIN_DIR, "label", tile_file.replace("B2_B2_masked_", "Y_output_resized_"))
gt_mask = np.array(Image.open(gt_path), dtype=np.float32)
if gt_mask.ndim == 3:
    gt_mask = gt_mask[:, :, 0]
gt_mask = (gt_mask > 0).astype(np.uint8)

# Inference
bands_normalized = normalize_bands(bands)
image_tensor = torch.from_numpy(bands_normalized).unsqueeze(0).float().to(device)

with torch.no_grad():
    output = model(image_tensor)
    prediction_raw = torch.sigmoid(output).squeeze().cpu().numpy()

# Step-by-step post-processing
print("=" * 80)
print(f"DETAILED ANALYSIS: {tile_file}")
print("=" * 80)

print(f"\n[STEP 1] Raw Model Output (after sigmoid)")
print(f"  Shape: {prediction_raw.shape}")
print(f"  Min: {prediction_raw.min():.4f}")
print(f"  Max: {prediction_raw.max():.4f}")
print(f"  Mean: {prediction_raw.mean():.4f}")
print(f"  Std: {prediction_raw.std():.4f}")

# Histogram of raw predictions
bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hist, _ = np.histogram(prediction_raw, bins=bins)
print(f"\n  Distribution:")
for i in range(len(bins)-1):
    pct = 100 * hist[i] / prediction_raw.size
    print(f"    {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:6d} pixels ({pct:5.2f}%)")

print(f"\n[STEP 2] After Threshold > 0.5")
binary_05 = (prediction_raw > 0.5).astype(np.uint8)
print(f"  Pixels > 0.5: {binary_05.sum():6d} ({100*binary_05.mean():.2f}%)")

print(f"\n[STEP 3] Try different thresholds")
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    binary = (prediction_raw > thresh).astype(np.uint8)
    from sklearn.metrics import matthews_corrcoef
    if binary.sum() > 0:
        mcc = matthews_corrcoef(gt_mask.flatten(), binary.flatten())
        print(f"  Threshold {thresh:.1f}: {binary.sum():6d} pixels → MCC = {mcc:.4f}")
    else:
        print(f"  Threshold {thresh:.1f}: {binary.sum():6d} pixels → MCC = 0.0000 (no predictions)")

# Component analysis at optimal threshold
print(f"\n[STEP 4] Connected Components Analysis (threshold=0.5)")
from scipy.ndimage import label as scipy_label
binary_mask = (prediction_raw > 0.5).astype(np.uint8)
labeled_array, num_features = scipy_label(binary_mask)

if num_features > 0:
    print(f"  Number of components: {num_features}")
    component_sizes = []
    for region_id in range(1, num_features + 1):
        region_mask = labeled_array == region_id
        size = region_mask.sum()
        component_sizes.append(size)
    
    component_sizes = sorted(component_sizes, reverse=True)
    print(f"  Component sizes: {component_sizes[:10]}")
    print(f"  Largest: {component_sizes[0] if component_sizes else 0}")
    print(f"  Components >= 100 pixels: {sum(1 for s in component_sizes if s >= 100)}")
    print(f"  Components < 100 pixels: {sum(1 for s in component_sizes if s < 100)}")
else:
    print(f"  No components detected!")

print(f"\n[STEP 5] After removing components < 100 pixels")
final_mask = binary_mask.copy()
if num_features > 0:
    for region_id in range(1, num_features + 1):
        region_mask = labeled_array == region_id
        if region_mask.sum() < 100:
            final_mask[region_mask] = 0

print(f"  Final glacier pixels: {final_mask.sum():6d} ({100*final_mask.mean():.2f}%)")

if final_mask.sum() > 0:
    from sklearn.metrics import matthews_corrcoef
    mcc_final = matthews_corrcoef(gt_mask.flatten(), final_mask.flatten())
    print(f"  Final MCC: {mcc_final:.4f}")
else:
    print(f"  Final MCC: 0.0000 (no predictions left!)")

print(f"\n[STEP 6] Ground Truth")
print(f"  Glacier pixels: {gt_mask.sum():6d} ({100*gt_mask.mean():.2f}%)")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Detailed Analysis: {tile_file}', fontsize=16, fontweight='bold')

# Row 1: Inputs
rgb = np.stack([
    np.clip(bands[2] / 3000, 0, 1),
    np.clip(bands[1] / 3000, 0, 1),
    np.clip(bands[0] / 3000, 0, 1),
], axis=-1)
axes[0, 0].imshow(rgb)
axes[0, 0].set_title('RGB Composite')
axes[0, 0].axis('off')

axes[0, 1].imshow(gt_mask, cmap='Blues', vmin=0, vmax=1)
axes[0, 1].set_title(f'Ground Truth\n{gt_mask.sum()} pixels ({100*gt_mask.mean():.1f}%)')
axes[0, 1].axis('off')

axes[0, 2].imshow(prediction_raw, cmap='RdYlGn', vmin=0, vmax=1)
axes[0, 2].set_title(f'Raw Prediction\n[{prediction_raw.min():.3f}, {prediction_raw.max():.3f}]')
axes[0, 2].axis('off')

# Row 2: Processing steps
axes[1, 0].imshow(binary_05, cmap='Blues', vmin=0, vmax=1)
axes[1, 0].set_title(f'After Threshold > 0.5\n{binary_05.sum()} pixels ({100*binary_05.mean():.2f}%)')
axes[1, 0].axis('off')

# Try threshold 0.3
binary_03 = (prediction_raw > 0.3).astype(np.uint8)
if binary_03.sum() > 0:
    mcc_03 = matthews_corrcoef(gt_mask.flatten(), binary_03.flatten())
    axes[1, 1].imshow(binary_03, cmap='Blues', vmin=0, vmax=1)
    axes[1, 1].set_title(f'Threshold > 0.3 (MCC={mcc_03:.3f})\n{binary_03.sum()} pixels')
else:
    axes[1, 1].text(0.5, 0.5, 'Still all zeros!', ha='center', va='center')
    axes[1, 1].set_title('Threshold > 0.3')
axes[1, 1].axis('off')

axes[1, 2].imshow(final_mask, cmap='Blues', vmin=0, vmax=1)
if final_mask.sum() > 0:
    mcc_final = matthews_corrcoef(gt_mask.flatten(), final_mask.flatten())
    axes[1, 2].set_title(f'Final (MCC={mcc_final:.3f})\n{final_mask.sum()} pixels')
else:
    axes[1, 2].set_title(f'Final (ALL REMOVED!)\n{final_mask.sum()} pixels')
axes[1, 2].axis('off')

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, 'detailed_analysis.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {output_path}")
print("=" * 80)
