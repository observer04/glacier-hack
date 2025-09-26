
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

def get_tile_id(filename):
    """Extract tile ID from filename."""
    match = re.search(r'(\d{2}_\d{2})', filename)
    return match.group(1) if match else None

def preprocess_dataset(input_dir, output_dir):
    """
    Reads the original dataset, combines the 5 bands for each tile into a single
    NumPy array, and saves it to the output directory.
    Also copies the label files.
    """
    print(f"Starting preprocessing...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    
    band_dirs = {f"Band{i}": os.path.join(input_dir, f"Band{i}") for i in range(1, 6)}
    
    # Get all unique tile IDs from the first band
    all_files = os.listdir(band_dirs["Band1"])
    tile_ids = sorted(list(set(tid for f in all_files if (tid := get_tile_id(f)) is not None)))

    print(f"Found {len(tile_ids)} unique tiles to process.")

    for tid in tqdm(tile_ids, desc="Processing Tiles"):
        bands = []
        # Find a filename that contains the tile_id
        try:
            fname_pattern = [f for f in os.listdir(band_dirs["Band1"]) if tid in f][0]
        except IndexError:
            print(f"Warning: Could not find a file for tile ID {tid} in Band1. Skipping.")
            continue

        all_bands_exist = True
        for i in range(1, 6):
            band_fname = fname_pattern.replace("B1", f"B{i}").replace("b1", f"b{i}")
            fp = os.path.join(band_dirs[f"Band{i}"], band_fname)
            if not os.path.exists(fp):
                # Fallback for names like img_01_01.tif
                fp = os.path.join(band_dirs[f"Band{i}"], fname_pattern)
                if not os.path.exists(fp):
                    print(f"Warning: Band {i} for tile {tid} not found. Skipping tile.")
                    all_bands_exist = False
                    break
            
            img = tifffile.imread(fp)
            bands.append(img)
        
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
