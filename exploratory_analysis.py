import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import torch
from tqdm import tqdm

# Set paths
train_dir = "./Train"
band_dirs = [os.path.join(train_dir, f"Band{i}") for i in range(1, 6)]
label_dir = os.path.join(train_dir, "label")

def check_dataset_structure():
    """Check the structure of the dataset."""
    print("Checking dataset structure...")
    
    # Check directory structure
    print(f"Band directories: {[os.path.exists(d) for d in band_dirs]}")
    print(f"Label directory: {os.path.exists(label_dir)}")
    
    # Count files in each directory
    for i, band_dir in enumerate(band_dirs, 1):
        if os.path.exists(band_dir):
            files = glob.glob(os.path.join(band_dir, "*.tif"))
            print(f"Band {i} has {len(files)} files")
    
    if os.path.exists(label_dir):
        label_files = glob.glob(os.path.join(label_dir, "*.tif"))
        print(f"Label directory has {len(label_files)} files")
    
def analyze_sample_image():
    """Analyze a sample image from each band and the label."""
    print("\nAnalyzing sample image...")
    
    # Get a sample label file
    label_files = glob.glob(os.path.join(label_dir, "*.tif"))
    if not label_files:
        print("No label files found.")
        return
    
    sample_label_file = label_files[0]
    # Extract the tile ID from the label filename (support multiple patterns)
    base = os.path.splitext(os.path.basename(sample_label_file))[0]
    if base.startswith("Y_output_resized_"):
        sample_id = base[len("Y_output_resized_"):]
    elif base.startswith("Y"):
        sample_id = base[1:]
    else:
        import re
        m = re.search(r"(\d{2}_\d{2})", base)
        sample_id = m.group(1) if m else base
    print(f"Sample tile ID: {sample_id}")
    
    # Find corresponding band files
    sample_bands = []
    band_files = []
    
    for i, band_dir in enumerate(band_dirs, 1):
        if not os.path.exists(band_dir):
            continue
            
        # Find files that match the sample ID
        matching_files = glob.glob(os.path.join(band_dir, f"*{sample_id}*.tif"))
        if matching_files:
            band_file = matching_files[0]
            band_files.append(band_file)
            band = np.array(Image.open(band_file))
            if band.ndim == 3:
                band = band[..., 0]
                
            sample_bands.append(band)
            print(f"Band {i} shape: {band.shape}, dtype: {band.dtype}, min: {band.min()}, max: {band.max()}")
    
    # Load the label
    label = np.array(Image.open(sample_label_file))
    print(f"Label shape: {label.shape}, dtype: {label.dtype}, unique values: {np.unique(label)}")
    
    # Visualize sample data
    plt.figure(figsize=(15, 10))
    for i, band in enumerate(sample_bands):
        plt.subplot(2, 3, i+1)
        plt.imshow(band, cmap='gray')
        plt.title(f"Band {i+1}")
        plt.colorbar()
    
    plt.subplot(2, 3, 6)
    plt.imshow(label, cmap='gray')
    plt.title("Label")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("sample_data_visualization.png")
    print(f"Sample visualization saved to sample_data_visualization.png")
    
    # Check class distribution
    glacier_pixels = np.sum(label > 0)
    total_pixels = label.size
    print(f"Glacier pixels: {glacier_pixels} ({glacier_pixels/total_pixels*100:.2f}%)")
    print(f"Non-glacier pixels: {total_pixels - glacier_pixels} ({(total_pixels - glacier_pixels)/total_pixels*100:.2f}%)")

def analyze_class_distribution():
    """Analyze the class distribution across all images."""
    print("\nAnalyzing class distribution across all images...")
    
    total_glacier_pixels = 0
    total_pixels = 0
    
    label_files = sorted(glob.glob(os.path.join(label_dir, "*.tif")))
    for label_file in tqdm(label_files, desc="Processing labels"):
        label = np.array(Image.open(label_file))
        glacier_pixels = np.sum(label > 0)
        total_glacier_pixels += glacier_pixels
        total_pixels += label.size
    
    print(f"Total images: {len(label_files)}")
    print(f"Total glacier pixels: {total_glacier_pixels} ({total_glacier_pixels/total_pixels*100:.2f}%)")
    print(f"Total non-glacier pixels: {total_pixels - total_glacier_pixels} ({(total_pixels - total_glacier_pixels)/total_pixels*100:.2f}%)")
    
    # Plot class distribution
    plt.figure(figsize=(8, 6))
    # Cast to float to avoid type issues with some backends/linters
    glacier_val = float(total_glacier_pixels)
    non_glacier_val = float(total_pixels - total_glacier_pixels)
    plt.pie([glacier_val, non_glacier_val], 
            labels=["Glacier", "Non-Glacier"], 
            autopct='%1.1f%%',
            colors=['lightblue', 'lightgray'])
    plt.title("Class Distribution")
    plt.savefig("class_distribution.png")
    print(f"Class distribution visualization saved to class_distribution.png")

def analyze_band_statistics():
    """Analyze statistics for each band."""
    print("\nAnalyzing band statistics...")
    
    band_means = [[] for _ in range(len(band_dirs))]
    band_stds = [[] for _ in range(len(band_dirs))]
    
    # Get a sample of files
    all_files = []
    band_indices = []  # Keep track of band indices
    for i, band_dir in enumerate(band_dirs):
        if not os.path.exists(band_dir):
            continue
            
        files = sorted(glob.glob(os.path.join(band_dir, "*.tif")))
        if files:
            # Take 10 sample files or all if less than 10
            sample_files = files[:min(10, len(files))]
            all_files.append(sample_files)
            band_indices.append(i)  # Store the band index
    
    # Process each set of band files
    for idx, files in enumerate(all_files):
        band_i = band_indices[idx]  # Get the correct band index
        for file_path in tqdm(files, desc=f"Processing Band {band_i+1}"):
            arr = np.array(Image.open(file_path))
            if arr.ndim == 3:
                arr = arr[..., 0]
                
            # Calculate statistics
            band_means[band_i].append(np.mean(arr))
            band_stds[band_i].append(np.std(arr))
    
    # Print statistics and prepare plotting data
    x_values = []
    y_values = []
    for i in range(len(band_dirs)):
        if band_means[i]:
            mean_value = np.mean(band_means[i])
            print(f"Band {i+1} mean: {mean_value:.2f}, std: {np.mean(band_stds[i]):.2f}")
            x_values.append(i+1)
            y_values.append(mean_value)
    
    # Plot statistics
    plt.figure(figsize=(10, 6))
    plt.bar(x_values, y_values)
    plt.xlabel("Band")
    plt.ylabel("Mean Pixel Value")
    plt.title("Average Pixel Value by Band")
    plt.savefig("band_means.png")
    print(f"Band statistics visualization saved to band_means.png")

if __name__ == "__main__":
    check_dataset_structure()
    analyze_sample_image()
    analyze_class_distribution()
    analyze_band_statistics()