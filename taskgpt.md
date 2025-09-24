# GlacierHack TaskGPT Guide

This doc tracks our incremental plan and progress for the Glacier segmentation challenge. We'll validate the pipeline end-to-end quickly, then iterate on modeling.

## Current status
- Data structure validated (5 bands + labels). Labels follow pattern `Y_output_resized_{tile_id}.tif`; robust loader now supports both this and `Y{tile_id}.tif`.
- solution.py fixed thresholding and supports DeepLabV3+ path with PixelANN fallback.
- exploratory_analysis runs and produces artifacts.
- Datasets (pixel-wise and full-tile) load successfully after fixes.

## Next steps
1) Baseline training (PixelANN)
   - Train a simple pixel-wise classifier to validate training/eval loop.
   - Short run: 3–5 epochs, large batch size (CPU-friendly).
   - Evaluate on validation set; compute MCC.
2) UNet small model (5-channel input)
   - Train a lightweight UNet on full tiles with small batch size.
   - Track MCC; checkpoint best.
3) DeepLabV3+
   - Train DeepLabV3+ on full tiles; tune output stride and learning rate.
   - Evaluate; compare with UNet.
4) Submission alignment
   - Export best model weights under `models/` and confirm solution.py loads/infers correctly.
   - Run `test_solution.py` to generate sample predictions and metrics.

## Done
- Robust label path detection across loaders and tests.
- Fixes in `solution.py`, `exploratory_analysis.py`, `data_utils.py`.

## Notes
- Primary metric: MCC. Make sure evaluation scripts record it.
- If GPU is available, increase batch sizes; otherwise keep memory conservative.
