# ==================================================================================
# LOCAL_RUNNER.PY - Test script for local execution
# ==================================================================================

import os
import sys
from PIL import Image
import numpy as np

# --- Paths for the local environment ---
# This assumes the script is in the project root directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "Train")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output_masks")
SOLUTION_PATH = os.path.join(PROJECT_DIR, "solution1.py")
MODEL_PATH = os.path.join(PROJECT_DIR, "submission_model.pth")

# Add the project directory to the path to allow importing the solution module
sys.path.insert(0, PROJECT_DIR)

def run_local_test():
    print("--- Starting Local Sanity Check ---")

    # 1. Check for necessary files and folders
    if not os.path.exists(SOLUTION_PATH):
        print(f"ERROR: solution.py not found at {SOLUTION_PATH}")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: submission_model.pth not found at {MODEL_PATH}")
        return
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: Data directory not found at {DATA_DIR}. Please ensure the 'Train' folder is in the same directory as this script.")
        return
        
    try:
        import solution
        print("--- Successfully imported solution.py ---")
    except Exception as e:
        print(f"ERROR: Failed to import solution.py. Error: {e}")
        return

    # 2. Prepare the input for the maskgeration function
    imagepath = {band: os.path.join(DATA_DIR, band) for band in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, band)) and band.startswith('Band')}
    
    if not imagepath:
        print(f"ERROR: No 'Band' folders found in {DATA_DIR}. Please check the DATA_DIR path.")
        return

    # 3. Run the mask generation function
    print("\n--- Calling solution.maskgeration... ---")
    output_masks = solution.maskgeration(imagepath, MODEL_PATH)
    print("--- solution.maskgeration finished. ---")

    # 4. Save the output masks
    if isinstance(output_masks, dict) and len(output_masks) > 0:
        print(f"\n--- Verification Succeeded! Received {len(output_masks)} masks. ---")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"--- Saving masks to: {OUTPUT_DIR} ---")
        
        for tile_id, mask_array in output_masks.items():
            # The solution produces a binary {0, 1} mask, scale to {0, 255} for image saving
            mask_to_save = (mask_array * 255).astype(np.uint8)
            mask_img = Image.fromarray(mask_to_save)
            mask_img.save(os.path.join(OUTPUT_DIR, f"mask_{tile_id}.png"))
            
        print(f"\nSUCCESS: Saved {len(output_masks)} masks.")
        print(f"You can now check the '{os.path.basename(OUTPUT_DIR)}' folder.")
    else:
        print("ERROR: The output from solution.py was not a valid, non-empty dictionary.")

if __name__ == '__main__':
    # First, ensure you have the required libraries installed:
    # pip install torch torchvision tifffile numpy opencv-python pillow
    run_local_test()
