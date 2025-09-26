

import os
import sys
import random
import shutil
import subprocess

# ==================================================================================
# --- CONFIGURATION ---
# ==================================================================================

# --- Paths ---
# IMPORTANT: These paths assume you are running this script from the /kaggle/working/glacier-hack/
# Adjust if your structure is different.

# Path to the directory where your original training data is located.
ORIGINAL_DATA_DIR = "/kaggle/working/Train"

# Path to the solution.py script we want to test.
SOLUTION_SCRIPT_PATH = "solution.py" 

# --- Temporary Directories ---
# These will be created, used, and then deleted.
TEMP_TEST_DATA_DIR = "/tmp/sanity_check_data"
TEMP_OUTPUT_DIR = "/tmp/sanity_check_output"

# --- Test Settings ---
# Number of random image tiles to use for the test.
NUM_SAMPLES = 3

# ==================================================================================
# --- UTILITY FUNCTIONS ---
# ==================================================================================

def _find_label_path(label_dir, tile_id):
    """Finds the path to a label file given a tile ID."""
    for name in [f"Y{tile_id}.tif", f"Y_output_resized_{tile_id}.tif"]:
        path = os.path.join(label_dir, name)
        if os.path.exists(path):
            return path
    return None

def get_tile_id_from_filename(filename):
    """Extracts tile ID (e.g., '01_02') from a filename."""
    import re
    match = re.search(r'(\d{2}_\d{2})', filename)
    return match.group(1) if match else None

# ==================================================================================
# --- SANITY CHECK MAIN LOGIC ---
# ==================================================================================

def run_sanity_check():
    """Performs the end-to-end sanity check."""
    print("--- Starting Sanity Check ---")
    
    # --- 1. Cleanup and Setup ---
    print("1. Cleaning up and creating temporary directories...")
    if os.path.exists(TEMP_TEST_DATA_DIR):
        shutil.rmtree(TEMP_TEST_DATA_DIR)
    if os.path.exists(TEMP_OUTPUT_DIR):
        shutil.rmtree(TEMP_OUTPUT_DIR)
    
    os.makedirs(TEMP_OUTPUT_DIR)
    for i in range(1, 6):
        os.makedirs(os.path.join(TEMP_TEST_DATA_DIR, f"Band{i}"), exist_ok=True)

    # --- 2. Select Random Samples ---
    print(f"2. Selecting {NUM_SAMPLES} random samples from {ORIGINAL_DATA_DIR}...")
    try:
        # Get all unique tile IDs from the label directory
        label_dir = os.path.join(ORIGINAL_DATA_DIR, "label")
        all_tile_ids = list(set([get_tile_id_from_filename(f) for f in os.listdir(label_dir) if f.endswith(".tif")]))
        all_tile_ids = [tid for tid in all_tile_ids if tid is not None]

        if len(all_tile_ids) < NUM_SAMPLES:
            print(f"Error: Not enough unique tile IDs in {label_dir} to run the check.")
            return False
        
        selected_tile_ids = random.sample(all_tile_ids, NUM_SAMPLES)
        print(f"   Selected tile IDs: {selected_tile_ids}")

    except Exception as e:
        print(f"Error: Could not select random samples. Check that ORIGINAL_DATA_DIR is correct.")
        print(f"   Details: {e}")
        return False

    # --- 3. Find corresponding filenames and copy ---
    print("3. Copying sample files to temporary test directory...")
    try:
        # Create a map of tile_id -> filename from one of the band directories
        band1_dir = os.path.join(ORIGINAL_DATA_DIR, "Band1")
        id_to_filename_map = {get_tile_id_from_filename(f): f for f in os.listdir(band1_dir) if f.endswith(".tif")}
        
        selected_filenames = []
        for tid in selected_tile_ids:
            if tid not in id_to_filename_map:
                print(f"Error: Could not find a corresponding file in Band1 for tile ID: {tid}")
                return False
            selected_filenames.append(id_to_filename_map[tid])
        
        print(f"   Resolved filenames to copy: {selected_filenames}")

        for filename in selected_filenames:
            for i in range(1, 6):
                src_path = os.path.join(ORIGINAL_DATA_DIR, f"Band{i}", filename)
                dst_path = os.path.join(TEMP_TEST_DATA_DIR, f"Band{i}", filename)
                if not os.path.exists(src_path):
                    print(f"Error: Source file not found: {src_path}")
                    return False
                shutil.copy(src_path, dst_path)
    except Exception as e:
        print(f"Error: Failed to copy files. Details: {e}")
        return False

    # --- 4. Execute solution.py ---
    print(f"4. Executing {SOLUTION_SCRIPT_PATH}...")
    command = [
        sys.executable,
        SOLUTION_SCRIPT_PATH,
        "--data", TEMP_TEST_DATA_DIR,
        "--out", TEMP_OUTPUT_DIR,
        "--masks", TEMP_OUTPUT_DIR
    ]
    
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=True, timeout=300)
        print("   --- solution.py stdout ---")
        print(process.stdout)
        print("   --- solution.py stderr ---")
        print(process.stderr)
        print("   Execution successful.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {SOLUTION_SCRIPT_PATH} failed to execute.")
        print(f"   Return Code: {e.returncode}")
        print(f"   --- stdout ---")
        print(e.stdout)
        print(f"   --- stderr ---")
        print(e.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"Error: {SOLUTION_SCRIPT_PATH} timed out.")
        return False
    except FileNotFoundError:
        print(f"Error: Could not find {SOLUTION_SCRIPT_PATH}. Make sure it's in the same directory.")
        return False

    # --- 5. Verify Output ---
    print("5. Verifying output masks...")
    try:
        output_files = os.listdir(TEMP_OUTPUT_DIR)
        if len(output_files) != NUM_SAMPLES:
            print(f"Failure: Expected {NUM_SAMPLES} output files, but found {len(output_files)}.")
            return False
        
        if sorted(output_files) != sorted(selected_filenames):
            print("Failure: Output filenames do not match input filenames.")
            print(f"   Expected: {sorted(selected_filenames)}")
            print(f"   Got:      {sorted(output_files)}")
            return False
        
        print("   Number of files and filenames are correct.")

    except Exception as e:
        print(f"Error: Could not verify output directory. Details: {e}")
        return False

    # --- 6. Cleanup ---
    print("6. Cleaning up temporary directories...")
    shutil.rmtree(TEMP_TEST_DATA_DIR)
    shutil.rmtree(TEMP_OUTPUT_DIR)
    
    return True

if __name__ == "__main__":
    if run_sanity_check():
        print("\n✅✅✅ Sanity Check Passed! ✅✅✅")
        print("Your solution.py script ran successfully on a sample of the data.")
    else:
        print("\n❌❌❌ Sanity Check Failed. ❌❌❌")
        print("Please review the errors above.")

