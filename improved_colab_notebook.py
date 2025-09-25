
# Improved Colab Notebook for Glacier Segmentation

from google.colab import drive
drive.mount('/content/drive')

epochs = 90

# 1. Setup
# Install the required libraries

import os
import cv2
import tifffile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

from models import UNet
from data_utils import GlacierTileDataset, normalize_band, get_tile_id
from train_utils import DiceLoss, CombinedLoss

model_name = "custom_unet_scratch_loss-0.2bce-0.8dice"
model_save_path = f"/content/drive/MyDrive/glacier_hack/{model_name}_best_model.pth"

# 2. Data Loading and Augmentation

# Define the augmentations
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90), # RandomRotate90 is not directly available in torchvision
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5, 0.5)),
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5, 0.5, 0.5)),
])

# Create the datasets and dataloaders
# NOTE: You will need to upload your 'Train' directory to your Colab environment
# or mount your Google Drive if the data is there.
data_dir = "./Train"

train_dataset = GlacierTileDataset(data_dir, is_training=True, augment=True)
val_dataset = GlacierTileDataset(data_dir, is_training=False, augment=False)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

# 3. Model Definition

# Use a pre-trained Unet with a ResNet34 backbone
model = UNet(in_channels=5, out_channels=1)

# 4. Training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the loss function and optimizer
criterion = CombinedLoss(alpha=0.2, beta=0.8) # Use CombinedLoss from train_utils.py

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

# Training loop
best_mcc = -1
no_improve_epochs = 0 # For early stopping

for epoch in range(epochs):
    model.train()
    train_loss = 0
    all_preds = []
    all_masks = []
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, masks.unsqueeze(1)) # Add channel dimension to masks

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0) # Accumulate loss correctly

        preds = (outputs.detach() > 0.5).float().cpu().numpy().reshape(-1)
        all_preds.extend(preds)
        all_masks.extend(masks.float().cpu().numpy().reshape(-1))

    train_loss /= len(train_loader.dataset)
    train_mcc = matthews_corrcoef(all_masks, all_preds)

    # Validation
    model.eval()
    val_loss = 0
    all_preds = []
    all_masks = []
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            loss = criterion(outputs, masks.unsqueeze(1)) # Add channel dimension to masks

            val_loss += loss.item() * images.size(0)

            preds = (outputs.detach() > 0.5).float().cpu().numpy().reshape(-1)
            all_preds.extend(preds)
            all_masks.extend(masks.float().cpu().numpy().reshape(-1))

    val_loss /= len(val_loader.dataset)
    val_mcc = matthews_corrcoef(all_masks, all_preds)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train MCC: {train_mcc:.4f}, Val Loss: {val_loss:.4f}, Val MCC: {val_mcc:.4f}")

    if val_mcc > best_mcc:
        best_mcc = val_mcc
        torch.save(model.state_dict(), model_save_path)
        print(f"New best model saved with MCC: {best_mcc:.4f}")
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        print(f"No improvement for {no_improve_epochs} epochs")

    scheduler.step(val_mcc) # For ReduceLROnPlateau

    if no_improve_epochs >= 10: # Early stopping patience
        print(f"Early stopping after {epoch+1} epochs")
        break

print(f"Training complete. Best MCC: {best_mcc:.4f}")

# 5. Evaluation

# You can now download the 'best_model.pth' file and use it in your 'solution.py'.
# Remember to update your 'solution.py' to use the same model architecture:
# model = smp.Unet(
#     encoder_name="resnet34",
#     encoder_weights=None, # Don't load imagenet weights for inference
#     in_channels=5,
#     classes=1,
# )

