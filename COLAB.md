# Colab Training Guide

This guide helps you train UNet or DeepLabV3+ on Google Colab with GPU.

## 1) Setup
- Runtime → Change runtime type → T4/A100 GPU → Save
- In a new notebook cell:

```python
!nvidia-smi
!pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip -q install tifffile pillow numpy tqdm seaborn scikit-learn
```

## 2) Get the project and data
Option A: Upload repo and Train folder to Colab Files (drag-and-drop) into `/content/glacier-hack`.

Option B: Clone from your own Git repo if available:
```python
!git clone <your-repo-url> glacier-hack
```
Then upload the `Train/` folder under `glacier-hack/` (ensure Band1..Band5 and label are present).

## 3) Train
UNet (recommended to start):
```python
%cd /content/glacier-hack
!python train_model.py \
  --model_type=unet \
  --data_dir=./Train \
  --epochs=40 \
  --batch_size=2 \
  --learning_rate=0.001 \
  --scheduler=cosine \
  --device=cuda \
  --num_workers=2 \
  --model_save_path=./models/unet_colab \
  --augment
```

DeepLabV3+ (requires batch_size ≥ 2 due to BatchNorm):
```python
!python train_model.py \
  --model_type=deeplabv3plus \
  --data_dir=./Train \
  --epochs=40 \
  --batch_size=2 \
  --learning_rate=0.0005 \
  --scheduler=cosine \
  --device=cuda \
  --num_workers=2 \
  --model_save_path=./models/deeplab_colab \
  --augment
```

Tips:
- Increase epochs to 60–80 if training is stable and time permits.
- Try `--loss=combined` to blend BCE and Dice; or keep BCE for simplicity.

## 4) Evaluate
```python
!python evaluate_model.py \
  --model_type=unet \
  --model_path=./models/unet_colab/model.pth \
  --data_dir=./Train \
  --batch_size=1 \
  --device=cuda \
  --num_workers=2 \
  --output_dir=./results/unet_colab
```

## 5) Prepare submission
- Copy the best `model.pth` to repo root (or keep path and adjust solution.py if needed).
- Validate `solution.py` end to end:
```python
!python test_solution.py --data_dir=./Train --output_dir=./test_results
```

## Notes
- No pandas needed; we operate on TIFF arrays directly.
- Augmentations: flips/rotations are enabled for segmentation models when `--augment` is set.
- If you see OOM, reduce `--batch_size` or `--num_workers`.
