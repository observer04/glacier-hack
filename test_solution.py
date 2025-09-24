import os
import argparse
import numpy as np
import shutil
import torch
import tempfile
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef

# Import solution
from solution import maskgeration

def _find_label_path(label_dir: str, tile_id: str):
    candidates = [
        os.path.join(label_dir, f"Y{tile_id}.tif"),
        os.path.join(label_dir, f"Y_output_resized_{tile_id}.tif"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def test_solution(data_dir, output_dir):
    """Test solution on sample data."""
    print("Testing solution.py on sample data...")
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Ensure a model.pth exists for solution.py to load
        model_needed = not os.path.exists("model.pth")
        tmp_model_copied = False
        if model_needed:
            candidate_models = [
                os.path.join("./models", "pixelann_baseline", "model.pth"),
                os.path.join("./models", "model.pth"),
                os.path.join("./models", "best_model.pth"),
            ]
            for src in candidate_models:
                if os.path.exists(src):
                    shutil.copy(src, "model.pth")
                    tmp_model_copied = True
                    print(f"Using model weights from {src}")
                    break
            if not tmp_model_copied:
                print("Warning: No model.pth found and no candidate model to copy. maskgeration may fail.")
        # Prepare input structure
        imagepath = {
            f"Band{i}": os.path.join(data_dir, f"Band{i}") for i in range(1, 6)
        }
        
        # Generate masks
        maskgeration(imagepath, temp_dir)

        # Clean up temporary model if we created one
        if tmp_model_copied and os.path.exists("model.pth"):
            try:
                os.remove("model.pth")
            except OSError:
                pass
        
        # Check if masks were generated
        masks = [f for f in os.listdir(temp_dir) if f.endswith(".tif")]
        print(f"Generated {len(masks)} masks")
        
        if len(masks) == 0:
            print("Error: No masks were generated.")
            return False
        
        # Load ground truth masks
        label_dir = os.path.join(data_dir, "label")
        if not os.path.exists(label_dir):
            print(f"Warning: Label directory not found at {label_dir}.")
            print("Cannot compute metrics without ground truth.")
            return True
        
        # Compute metrics
        all_preds = []
        all_targets = []
        
        for mask_file in tqdm(masks, desc="Computing metrics"):
            # Extract tile ID
            tile_id = os.path.basename(mask_file).replace(".tif", "")
            
            # Load predicted mask
            pred_path = os.path.join(temp_dir, mask_file)
            pred = np.array(Image.open(pred_path))
            pred = (pred > 0).astype(np.uint8)
            
            # Load ground truth mask
            gt_path = _find_label_path(label_dir, tile_id)
            if gt_path is None or not os.path.exists(gt_path):
                continue
                
            gt = np.array(Image.open(gt_path))
            gt = (gt > 0).astype(np.uint8)
            
            # Flatten
            all_preds.extend(pred.flatten())
            all_targets.extend(gt.flatten())
            
            # Save comparison visualization
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(gt, cmap='gray')
            plt.title("Ground Truth")
            plt.colorbar()
            
            plt.subplot(1, 3, 2)
            plt.imshow(pred, cmap='gray')
            plt.title("Prediction")
            plt.colorbar()
            
            plt.subplot(1, 3, 3)
            diff = gt.astype(int) - pred.astype(int)
            plt.imshow(diff, cmap='bwr', vmin=-1, vmax=1)
            plt.title("Difference (GT - Pred)")
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"comparison_{tile_id}.png"))
            plt.close()
        
        if all_preds and all_targets:
            # Compute MCC
            mcc = matthews_corrcoef(all_targets, all_preds)
            print(f"Matthews Correlation Coefficient: {mcc:.4f}")
            
            # Save MCC
            with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
                f.write(f"Matthews Correlation Coefficient: {mcc:.4f}\n")
        else:
            print("Warning: Could not compute metrics. No matching ground truth found.")
    
    return True

def main(args):
    test_solution(args.data_dir, args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test solution script")
    parser.add_argument("--data_dir", type=str, default="./Train", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, default="./test_results", help="Path to save test results")
    args = parser.parse_args()
    
    main(args)