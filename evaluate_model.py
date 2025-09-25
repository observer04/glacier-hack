import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

from data_utils import create_dataloaders, create_segmentation_dataloaders
from models import UNet, DeepLabV3Plus, PixelANN  # PixelANN optional in your models.py

def plot_confusion(cm, save_path):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Non-Glacier', 'Glacier'])
    plt.yticks([0, 1], ['Non-Glacier', 'Glacier'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def compute_metrics(y_true, y_pred):
    y_true = y_true.astype(np.uint8).ravel()
    y_pred = y_pred.astype(np.uint8).ravel()
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return dict(mcc=mcc, f1=f1, precision=prec, recall=rec, confusion_matrix=cm)

def _tta_transforms(x):
    # List of lambdas producing augmented and inverse operations
    # Each entry: (forward_fn, inverse_fn)
    return [
        (lambda t: t, lambda t: t),
        (lambda t: torch.flip(t, dims=[-1]), lambda t: torch.flip(t, dims=[-1])),  # horizontal
        (lambda t: torch.flip(t, dims=[-2]), lambda t: torch.flip(t, dims=[-2])),  # vertical
        (lambda t: torch.rot90(t, 1, dims=[-2, -1]), lambda t: torch.rot90(t, -1, dims=[-2, -1])),
        (lambda t: torch.rot90(t, 2, dims=[-2, -1]), lambda t: torch.rot90(t, -2, dims=[-2, -1])),
        (lambda t: torch.rot90(t, 3, dims=[-2, -1]), lambda t: torch.rot90(t, -3, dims=[-2, -1])),
    ]

def evaluate_segmentation(model, loader, device, threshold, tta=False):
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            if not tta:
                logits = model(xb)
            else:
                # Ensemble predictions
                preds_accum = None
                tta_list = _tta_transforms(xb)
                for fwd, inv in tta_list:
                    x_aug = fwd(xb)
                    out = model(x_aug)
                    out = inv(out)
                    if preds_accum is None:
                        preds_accum = out
                    else:
                        preds_accum += out
                if preds_accum is None:
                    logits = model(xb)
                else:
                    logits = preds_accum / len(tta_list)
            if logits.dim() == 2:
                # Pixel model fallback: reshape if needed
                n, hw = logits.shape
                H, W = yb.shape[-2], yb.shape[-1]
                logits = logits.view(n, 1, H, W)
            elif logits.shape[-2:] != yb.shape[-2:]:
                logits = F.interpolate(logits, size=yb.shape[-2:], mode="bilinear", align_corners=False)
            probs = torch.sigmoid(logits)
            pred = (probs > threshold).cpu().numpy().astype(np.uint8)
            ys.append((yb.numpy() > 0).astype(np.uint8))
            ps.append(pred)
    y = np.concatenate([a.reshape(-1) for a in ys])
    p = np.concatenate([a.reshape(-1) for a in ps])
    return compute_metrics(y, p)

def evaluate_pixel(model, loader, device, threshold):
    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            pred = (probs > threshold).cpu().numpy().astype(np.uint8)
            ys.append((yb.numpy() > 0).astype(np.uint8))
            ps.append(pred)
    y = np.concatenate(ys).reshape(-1)
    p = np.concatenate(ps).reshape(-1)
    return compute_metrics(y, p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_type', type=str, required=True, choices=['unet', 'deeplabv3plus', 'pixelann'])
    ap.add_argument('--model_path', type=str, required=True)
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--batch_size', type=int, default=1)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--threshold', type=float, default=0.6)
    ap.add_argument('--output_dir', type=str, required=True)
    ap.add_argument('--tta', action='store_true', help='Enable flip/rotate test-time augmentation (segmentation models only)')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # Load model
    if args.model_type == 'unet':
        model = UNet(in_channels=5, out_channels=1)
    elif args.model_type == 'deeplabv3plus':
        # Remove unsupported pretrained argument (not in our implementation)
        model = DeepLabV3Plus(in_channels=5, out_channels=1)
    else:
        # PixelANN uses in_channels parameter in our implementation
        try:
            model = PixelANN(in_channels=5)
        except Exception:
            import torch.nn as nn
            model = nn.Sequential(nn.Linear(5, 1))
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)

    # Load validation split (tolerate different create_*dataloaders signatures)
    if args.model_type in ['unet', 'deeplabv3plus']:
        try:
            _, val_loader = create_segmentation_dataloaders(
                args.data_dir, batch_size=args.batch_size, augment=False, num_workers=args.num_workers
            )
        except TypeError:
            try:
                _, val_loader = create_segmentation_dataloaders(
                    args.data_dir, batch_size=args.batch_size, augment=False
                )
            except TypeError:
                _, val_loader = create_segmentation_dataloaders(
                    args.data_dir, batch_size=args.batch_size
                )
        metrics = evaluate_segmentation(model, val_loader, device, args.threshold, tta=args.tta)
    else:
        try:
            _, val_loader = create_dataloaders(
                args.data_dir, batch_size=4096, num_workers=args.num_workers
            )
        except TypeError:
            _, val_loader = create_dataloaders(
                args.data_dir, batch_size=4096
            )
        metrics = evaluate_pixel(model, val_loader, device, args.threshold)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"MCC: {metrics['mcc']:.4f}  F1: {metrics['f1']:.4f}  P: {metrics['precision']:.4f}  R: {metrics['recall']:.4f}")
    plot_confusion(metrics['confusion_matrix'], os.path.join(args.output_dir, "confusion_matrix.png"))
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        f.write(f"MCC: {metrics['mcc']:.4f}\n")
        f.write(f"F1: {metrics['f1']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")

if __name__ == '__main__':
    main()