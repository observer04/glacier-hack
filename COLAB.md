# TaskGPT: Glacier Segmentation End-to-End Plan (VS Code + Colab)

Objective
- Train UNet and DeepLabV3+ on 5-band tiles, evaluate with MCC, and finalize submission with a matching solution.py and model.pth.

Prereqs
- Linux + VS Code
- Train/ with Band1..Band5 and label/
- Python 3.9+

Phase 0 — Local setup and sanity checks (VS Code)
1) Create and activate venv
   - Terminal:
     - python -m venv .venv
     - source .venv/bin/activate
     - pip install --upgrade pip
     - pip install torch torchvision numpy pillow tifffile tqdm scikit-learn matplotlib seaborn

2) Verify dataset
   - python exploratory_analysis.py
   - Expect:
     - sample_data_visualization.png, class_distribution.png, band_means.png
   - If errors:
     - Confirm labels are named Y_output_resized_{tile_id}.tif or Y{tile_id}.tif
     - Check any band shape mismatch and fix files or report issue

3) Dataloader smoke tests
   - Pixel-wise:
     - python - <<'PY'
from data_utils import create_dataloaders
tl, vl = create_dataloaders('./Train', batch_size=128)
xb, yb = next(iter(tl))
print('pixel batch', xb.shape, yb.shape)
PY
   - Expect: torch.Size([128, 5]) torch.Size([128])
   - Tile-wise:
     - python - <<'PY'
from data_utils import create_segmentation_dataloaders
tl, vl = create_segmentation_dataloaders('./Train', batch_size=1, augment=False)
xb, yb = next(iter(tl))
print('tile batch', xb.shape, yb.shape)
PY
   - Expect: torch.Size([1, 5, H, W]) torch.Size([1, H, W])

Phase 1 — Local baseline (optional but recommended)
- Purpose: confirm train/eval loop + metrics + save/load work.

1) Quick PixelANN run
   - python train_model.py --model_type=pixelann --data_dir=./Train --epochs=2 --batch_size=2048 --device=cpu --model_save_path=./models/pixelann_quick
   - python evaluate_model.py --model_type=pixelann --model_path=./models/pixelann_quick/model.pth --data_dir=./Train --output_dir=./results/pixelann_quick

2) End-to-end check
   - cp ./models/pixelann_quick/model.pth ./model.pth
   - python test_solution.py --data_dir=./Train --output_dir=./test_results/pixelann_quick
   - Ensure masks generate and metrics.txt exists

Phase 2 — Train UNet in Colab (GPU)
- Use COLAB.md (appended below) cells.
- Recommended:
   - epochs: 60–120 (with early stopping)
   - batch_size: 2–4 (per tile)
   - lr: 1e-3 start with cosine schedule; or 1e-3 plateau to 1e-4
   - loss: combined (BCE+Dice) or focal (try both)
   - pos_weight: 1.5–2.0 for WBCE (if using wbce)
   - accum_steps: 2–4 if memory-limited to simulate larger batch
   - --augment: enable flips/rotations

Example UNet (combined loss + cosine + augment + accumulation):

```python
!python train_model.py \
   --model_type=unet \
   --data_dir=./Train \
   --epochs=100 \
   --batch_size=2 \
   --learning_rate=1e-3 \
   --optimizer=adam \
   --scheduler=cosine \
   --loss=combined \
   --combined_alpha=0.5 \
   --combined_beta=0.5 \
   --accum_steps=2 \
   --grad_clip=1.0 \
   --early_stopping_patience=10 \
   --augment \
   --model_save_path=./models/unet_colab
```

Alternative UNet (focal loss + plateau + higher pos_weight via WBCE):

```python
!python train_model.py \
   --model_type=unet \
   --data_dir=./Train \
   --epochs=80 \
   --batch_size=2 \
   --learning_rate=1e-3 \
   --optimizer=adam \
   --scheduler=plateau \
   --loss=wbce \
   --pos_weight=1.8 \
   --accum_steps=2 \
   --grad_clip=1.0 \
   --early_stopping_patience=8 \
   --augment \
   --model_save_path=./models/unet_colab_wbce
```

Artifacts:
- models/unet_colab/model.pth
- results/unet_colab/metrics.txt, confusion_matrix.png

Phase 3 — Evaluate UNet locally and threshold tuning
1) Bring model.pth from Colab to models/unet_colab/
2) Evaluate locally:
   - python evaluate_model.py --model_type=unet --model_path=./models/unet_colab/model.pth --data_dir=./Train --output_dir=./results/unet_local
3) Threshold sweep (optional, but recommended):
   - python - <<'PY'
import numpy as np, torch
from data_utils import GlacierTileDataset
from models import UNet
from sklearn.metrics import matthews_corrcoef
import os
ds = GlacierTileDataset('./Train', is_training=False)
dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = UNet(in_channels=5, out_channels=1).to(device)
m.load_state_dict(torch.load('./models/unet_colab/model.pth', map_location=device))
m.eval()
ts = np.linspace(0.3, 0.7, 9)
def eval_t(th):
    ys, ps = [], []
    with torch.no_grad():
        for x,y in dl:
            x = x.to(device)
            p = m(x).cpu().numpy()[0,0]
            ps.append((p>th).astype(np.uint8).ravel())
            ys.append((y.numpy()[0]>0).astype(np.uint8).ravel())
    ps = np.concatenate(ps); ys = np.concatenate(ys)
    return th, matthews_corrcoef(ys, ps)
print([eval_t(t) for t in ts])
PY

Phase 4 — Train DeepLabV3+ in Colab (GPU)
- Use COLAB.md cells for DeepLab.
- Key differences:
  - batch_size ≥ 2 (BatchNorm)
   - lr: 5e-4 with cosine; epochs 60–100
  - Potentially more tuning needed than UNet

Example DeepLab (focal + cosine + accumulation):

```python
!python train_model.py \
   --model_type=deeplabv3plus \
   --data_dir=./Train \
   --epochs=90 \
   --batch_size=2 \
   --learning_rate=5e-4 \
   --optimizer=adam \
   --scheduler=cosine \
   --loss=focal \
   --focal_alpha=0.25 \
   --focal_gamma=2.0 \
   --accum_steps=2 \
   --grad_clip=1.0 \
   --early_stopping_patience=10 \
   --augment \
   --model_save_path=./models/deeplab_colab
```

Artifacts:
- models/deeplab_colab/model.pth
- results/deeplab_colab/metrics.txt

Phase 5 — Compare and choose best model
- Compare MCC across results/unet_local and results/deeplab_local.
- Pick best model.pth and architecture.

Phase 6 — Align submission and final E2E test
1) Ensure solution.py model class matches trained model exactly:
   - If using UNet: ensure solution.py defines the same UNet and loads model.pth
   - If DeepLab: ensure solution.py defines same DeepLabV3+ and loads model.pth
2) Place best model at repo root or update solution.py to path:
   - cp ./models/<best>/model.pth ./model.pth
3) Final E2E:
   - python test_solution.py --data_dir=./Train --output_dir=./final_check
   - Inspect a few masks visually

Phase 7 — Optional enhancements
- Global normalization per band:
  - Compute mean/std per band on training set (Colab), persist JSON, load in solution.py
- Test-time augmentation (TTA): average predictions over flips/rotations
- Hyperparameter tuning: tweak losses, LR schedules, augmentations

Notes
- Data cleaning practiced here:
  - Shape assertions per tile (bands vs label)
  - NaN/Inf to finite via np.nan_to_num
  - Nodata filtering for all-zero pixels (pixel-wise path)


Environment controls for solution inference (optional)
- You can steer inference without changing code using environment variables consumed by solution.py:
   - SOLUTION_MODEL_PATH: path to weights. Default: model.pth
   - SOLUTION_MODEL_TYPE: auto | unet | deeplabv3plus | pixelann. Default: auto (tries UNet → DeepLab → PixelANN)
   - SOLUTION_THRESHOLD: float in [0,1] for binarization. Default: 0.5

Example usage in a Python cell before running tests/inference:

```python
import os
os.environ["SOLUTION_MODEL_PATH"] = "models/unet_small/best_model.pth"
os.environ["SOLUTION_MODEL_TYPE"] = "unet"
os.environ["SOLUTION_THRESHOLD"] = "0.47"
```


