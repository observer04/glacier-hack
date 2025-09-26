
# preprocess_data.py
# This script converts the original 5-band TIFF dataset into a combined NumPy format
# for significantly faster data loading during training.

import os
import numpy as np
import tifffile
from tqdm import tqdm
import shutil
import argparse
import re

import json

def get_tile_id(filename):
    """Extract tile ID from filename."""
    match = re.search(r'(\d{2}_\d{2})', filename)
    return match.group(1) if match else None

def preprocess_dataset(input_dir, output_dir):
    """
    Reads the original dataset, combines the 5 bands for each tile into a single
    NumPy array, and saves it to the output directory.
    Also computes global statistics for normalization and saves them.
    """
    print(f"Starting preprocessing...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    band_dirs = {f"Band{i}": os.path.join(input_dir, f"Band{i}") for i in range(1, 6)}
    
    all_files = os.listdir(band_dirs["Band1"])
    tile_ids = sorted(list(set(tid for f in all_files if (tid := get_tile_id(f)) is not None)))

    print(f"Found {len(tile_ids)} unique tiles to process.")

    # --- Pass 1: Compute Global Statistics ---
    print("--- Pass 1: Computing Global Statistics ---")
    channel_sums = np.zeros(5, dtype=np.float64)
    channel_sq_sums = np.zeros(5, dtype=np.float64)
    pixel_counts = np.zeros(5, dtype=np.int64)

    for tid in tqdm(tile_ids, desc="Calculating Stats"):
        for i in range(1, 6):
            band_dir = band_dirs[f"Band{i}"]
            try:
                matching_files = [f for f in os.listdir(band_dir) if tid in f]
                if not matching_files:
                    raise IndexError
                fname = matching_files[0]
                fp = os.path.join(band_dir, fname)
                img = tifffile.imread(fp).astype(np.float64)
                
                # Use non-zero pixels for stats
                valid_pixels = img[img > 0]
                if valid_pixels.size > 0:
                    channel_sums[i-1] += valid_pixels.sum()
                    channel_sq_sums[i-1] += (valid_pixels**2).sum()
                    pixel_counts[i-1] += valid_pixels.size

            except (IndexError, FileNotFoundError):
                continue # Skip if a file is missing, but continue stats calculation

    mean = channel_sums / pixel_counts
    # Var(X) = E[X^2] - (E[X])^2
    variance = (channel_sq_sums / pixel_counts) - (mean**2)
    std = np.sqrt(variance)

    stats = {'mean': mean.tolist(), 'std': std.tolist()}
    stats_path = os.path.join(output_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    print(f"Global stats saved to {stats_path}")
    print(stats)

    # --- Pass 2: Save Combined .npy Files ---
    print("--- Pass 2: Saving Combined .npy Files ---")

    for tid in tqdm(tile_ids, desc="Processing Tiles"):
        bands = []
        all_bands_exist = True
        for i in range(1, 6):
            band_dir = band_dirs[f"Band{i}"]
            try:
                # Find the unique file in this directory that contains the tile ID
                matching_files = [f for f in os.listdir(band_dir) if tid in f]
                if not matching_files:
                    raise IndexError
                fname = matching_files[0]
                fp = os.path.join(band_dir, fname)
                img = tifffile.imread(fp)
                bands.append(img)
            except (IndexError, FileNotFoundError):
                print(f"Warning: Band {i} for tile {tid} not found. Skipping tile.")
                all_bands_exist = False
                break
        
        if not all_bands_exist:
            continue

        # Stack bands along the last axis (H, W, C)
        combined_array = np.stack(bands, axis=-1).astype(np.int16) # Use int16 to save space
        
        # Save the combined array as a .npy file
        output_filepath = os.path.join(output_dir, f"{tid}.npy")
        np.save(output_filepath, combined_array)

    # --- Copy Labels ---
    input_label_dir = os.path.join(input_dir, "label")
    output_label_dir = os.path.join(output_dir, "label")
    if os.path.exists(input_label_dir):
        print(f"Copying label directory to {output_label_dir}...")
        shutil.copytree(input_label_dir, output_label_dir, dirs_exist_ok=True)
    else:
        print(f"Warning: Label directory not found at {input_label_dir}")

    print("\nPreprocessing complete!")
    print(f"Processed data saved to: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess glacier dataset.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the original 'Train' directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the processed data.")
    args = parser.parse_args()

    preprocess_dataset(args.input_dir, args.output_dir)
