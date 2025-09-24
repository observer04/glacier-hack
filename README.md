# GlacierHack 2025

## Overview
This project is part of the GlacierHack 2025 challenge, an IEEE GRSS activity for IEEE InGARSS 2025. The goal is to develop a machine learning model for semantic segmentation of glaciers using multispectral satellite imagery.

## Objective
Perform binary semantic segmentation on multispectral images to identify glacier pixels. The model will be evaluated using the Matthews Correlation Coefficient (MCC) on a hidden test set.

## Dataset
- **Input**: Multispectral images with 5 bands (B2: Blue, B3: Green, B4: Red, B6: SWIR, B10: TIR1)
- **Output**: Binary masks where 1 indicates glacier pixels and 0 indicates non-glacier pixels
- **Structure**:
  ```
  dataset/
    Band1/ (B2)
    Band2/ (B3)
    Band3/ (B4)
    Band4/ (B6)
    Band5/ (B10)
    label/ (ground truth masks)
  ```

## Submission
- **solution.py**: Python script with `maskgeration(imagepath, out_dir)` function
- **model.pth**: Trained PyTorch model weights (max 200MB)
- Include required packages as comments at the top of solution.py

## Environment
- Python 3.9
- PyTorch 2.0
- CUDA 11.7
- Additional packages: torch, torchvision, tqdm, opencv-python (example)

## Setup
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run training or inference as needed

## Evaluation Metric
Matthews Correlation Coefficient (MCC):
```
MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
```

## Validation
Model trained on one Himalayan glacier region and validated on an unseen region for generalization.

## References
- IEEE GRSS GlacierHack 2025
- IEEE InGARSS 2025