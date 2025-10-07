"""
Analyze the validation-test gap to understand what's going wrong
"""

import numpy as np
import torch
from solution import ResNet18UNet, normalize_bands, BAND_MEANS, BAND_STDS
from PIL import Image
import os

print("=" * 80)
print("DEEP DIVE: Why is test performance so much worse?")
print("=" * 80)
print()

# Key findings summary
print("OBSERVATIONS:")
print("  • Validation (Fold 5): 0.7007 MCC")
print("  • Test (Ensemble):     0.6019 MCC")
print("  • Gap:                 -0.0988 (-10% absolute!)")
print()

print("HYPOTHESIS 1: Test distribution is very different")
print("  - Test tiles might be from different regions/conditions")
print("  - Training covers only certain glacier types/conditions")
print("  - Solution: Train on ALL 25 tiles (no holdout)")
print()

print("HYPOTHESIS 2: Post-processing is too aggressive")
print("  - Current: threshold=0.45, min_size=50")
print("  - Test glaciers might be smaller or different shape")
print("  - Solution: Try threshold=0.35, min_size=20")
print()

print("HYPOTHESIS 3: Overfitting to validation fold")
print("  - We chose Fold 5 because it had 0.70 MCC")
print("  - But this might be cherry-picking")
print("  - Ensemble helps but only marginally (+0.0076)")
print("  - Solution: Train from scratch on all data")
print()

print("HYPOTHESIS 4: Hidden bug in data loading/normalization")
print("  - Let's verify normalization stats are actually applied")
print()

# Check normalization stats
print("Verifying normalization stats:")
print(f"  BAND_MEANS: {BAND_MEANS}")
print(f"  BAND_STDS: {BAND_STDS}")
print()

# Load a training tile and check raw values
band_files = {
    'Band1': 'Train/Band1/B2_B2_masked_02_07.tif',
    'Band2': 'Train/Band2/B3_B3_masked_02_07.tif',
    'Band3': 'Train/Band3/B4_B4_masked_02_07.tif',
    'Band4': 'Train/Band4/B8_B8_masked_02_07.tif',
    'Band5': 'Train/Band5/B12_B12_masked_02_07.tif'
}

if all(os.path.exists(f) for f in band_files.values()):
    bands = []
    for band_name in ['Band1', 'Band2', 'Band3', 'Band4', 'Band5']:
        band = np.array(Image.open(band_files[band_name]), dtype=np.float32)
        if band.ndim == 3:
            band = band[:, :, 0]
        bands.append(band)
    
    bands = np.stack(bands, axis=0)
    
    print("Raw band statistics (one training tile):")
    for i in range(5):
        print(f"  Band {i+1}: mean={bands[i].mean():.1f}, std={bands[i].std():.1f}, "
              f"min={bands[i].min():.1f}, max={bands[i].max():.1f}")
    print()
    
    # Normalize and check
    bands_norm = normalize_bands(bands)
    print("After normalization:")
    for i in range(5):
        print(f"  Band {i+1}: mean={bands_norm[i].mean():.3f}, std={bands_norm[i].std():.3f}")
    print("  (Should be close to mean=0, std=1 if normalization is correct)")
    print()

print("=" * 80)
print("RECOMMENDED ACTION:")
print("=" * 80)
print()
print("🎯 OPTION 1: Full Dataset Training (BEST CHANCE)")
print()
print("Implement: train_full_dataset.py")
print("  - Load ALL 25 tiles (no train/val split)")
print("  - Same ResNet18-UNet architecture")
print("  - Same augmentation (6 crops, MixUp)")
print("  - Train 40-50 epochs")
print("  - Monitor training loss convergence")
print()
print("Why this will work:")
print("  ✓ +25% more training data (25 vs 20 tiles per fold)")
print("  ✓ Better distribution coverage")
print("  ✓ No overfitting to specific validation fold")
print("  ✓ Addresses test distribution difference")
print()
print("Expected MCC: 0.65-0.72 (40-50% chance of 0.70+)")
print()
print("Time: 35-40 minutes")
print()

print("=" * 80)
print("Alternative if time is short:")
print("=" * 80)
print()
print("Quick post-processing test (5 min):")
print("  Try: threshold=0.35, min_size=20")
print("  Expected: +0.01-0.02 MCC → 0.61-0.62")
print("  Probability of 0.70+: ~5%")
print()
