
# Enhanced Colab Training Script for Glacier Segmentation
import os
import re
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random

# --- Configuration ---
# Set your data_dir to the correct path in your Colab environment
# e.g., DATA_DIR = '/content/drive/MyDrive/glacier-data/Train'
DATA_DIR = '/home/observer/projects/glacier-hack/Train' # <-- IMPORTANT: CHANGE THIS PATH
MODEL_SAVE_PATH = '/content/best_model_enhanced.pth'
EPOCHS = 150
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
SEED = 42

# Set seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- UNet Model Definition (Compliant) ---
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        def double_conv(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.conv_down1 = double_conv(in_channels, 64)
        self.conv_down2 = double_conv(64, 128)
        self.conv_down3 = double_conv(128, 256)
        self.conv_down4 = double_conv(256, 512)
        self.maxpool = nn.MaxPool2d(2)
        self.up_trans_1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = double_conv(512, 256)
        self.up_trans_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = double_conv(256, 128)
        self.up_trans_3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up3 = double_conv(128, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv_down1(x); x2 = self.maxpool(x1)
        x3 = self.conv_down2(x2); x4 = self.maxpool(x3)
        x5 = self.conv_down3(x4); x6 = self.maxpool(x5)
        x7 = self.conv_down4(x6)
        x = self.up_trans_1(x7)
        x = self.conv_up1(torch.cat([x, x5], 1))
        x = self.up_trans_2(x)
        x = self.conv_up2(torch.cat([x, x3], 1))
        x = self.up_trans_3(x)
        x = self.conv_up3(torch.cat([x, x1], 1))
        return self.out(x)

# --- Enhanced Loss Functions ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs_flat * targets_flat).sum()
        return 1 - ((2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth))

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice

# --- Helper Functions ---
def _find_label_path(label_dir: str, tile_id: str):
    # Simplified to match common patterns
    for pattern in [f"Y{tile_id}.tif", f"Y_output_resized_{tile_id}.tif"]:
        p = os.path.join(label_dir, pattern)
        if os.path.exists(p): return p
    return None

def get_tile_id(filename):
    match = re.search(r'(\d{2}_\d{2})', filename)
    return match.group(1) if match else None

# --- Dataset with Advanced Augmentations ---
class GlacierDataset(Dataset):
    def __init__(self, data_dir, is_training=True, val_split=0.2, random_state=42):
        self.data_dir = data_dir
        self.is_training = is_training
        self.band_dirs = [os.path.join(data_dir, f"Band{i}") for i in range(1, 6)]
        self.label_dir = os.path.join(data_dir, "label")

        all_files = os.listdir(self.band_dirs[0])
        self.tile_ids = sorted(list(set(tid for f in all_files if (tid := get_tile_id(f)) is not None)))
        
        train_ids, val_ids = train_test_split(self.tile_ids, test_size=val_split, random_state=random_state)
        self.tile_ids = train_ids if is_training else val_ids
        print(f"{'Training' if is_training else 'Validation'} dataset with {len(self.tile_ids)} tiles.")

        self.image_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5]*5, std=[0.5]*5), # Basic normalization
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
        self.affine_transform = T.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)

    def __len__(self):
        return len(self.tile_ids)

    def __getitem__(self, index):
        tid = self.tile_ids[index]
        
        bands = []
        # Find a filename that contains the tile_id
        fname_pattern = [f for f in os.listdir(self.band_dirs[0]) if tid in f][0]
        
        for i in range(5):
            # Attempt to construct band filenames
            band_fname = fname_pattern.replace("B1", f"B{i+1}").replace("b1", f"b{i+1}")
            fp = os.path.join(self.band_dirs[i], band_fname)
            if not os.path.exists(fp): # Fallback for names like img_01_01.tif
                fp = os.path.join(self.band_dirs[i], fname_pattern)

            img = Image.open(fp)
            bands.append(np.array(img))
        
        x = np.stack(bands, axis=-1).astype(np.float32) # Shape (H, W, 5)

        label_path = _find_label_path(self.label_dir, tid)
        y = Image.open(label_path)

        # Convert to tensors
        x = TF.to_tensor(x)
        y = TF.to_tensor(y).float()

        # Apply synchronized geometric augmentations
        if self.is_training:
            if random.random() > 0.5:
                x, y = TF.hflip(x), TF.hflip(y)
            if random.random() > 0.5:
                x, y = TF.vflip(x), TF.vflip(y)
            
            angle = T.RandomRotation.get_params([-30, 30])
            x, y = TF.rotate(x, angle), TF.rotate(y, angle)

            # Apply non-geometric transforms to image only
            x = T.ColorJitter(brightness=0.3, contrast=0.3)(x)

        # Normalize image
        x = TF.normalize(x, mean=[0.5]*5, std=[0.5]*5)
        
        return x, y

# --- Main Training & Evaluation Loop ---
def main():
    train_dataset = GlacierDataset(DATA_DIR, is_training=True)
    val_dataset = GlacierDataset(DATA_DIR, is_training=False)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=5, out_channels=1).to(device)
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=10, verbose=True)

    best_mcc = -1
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss, val_mcc = 0.0, 0.0
        all_preds, all_masks = [], []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy().flatten()
                all_preds.extend(preds)
                all_masks.extend(masks.cpu().numpy().flatten())

        val_mcc = matthews_corrcoef(all_masks, all_preds)
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val MCC: {val_mcc:.4f}")

        if val_mcc > best_mcc:
            best_mcc = val_mcc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> New best model saved! MCC: {best_mcc:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  -> No improvement for {patience_counter} epochs.")

        scheduler.step(val_mcc)
        if patience_counter >= 20:
            print("Early stopping triggered.")
            break

    print(f"\nTraining complete. Best Val MCC: {best_mcc:.4f}")

# To run this in Colab, you would call main()
# if __name__ == '__main__':
#     main()
