import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelANN(nn.Module):
    """Simple pixel-wise ANN for glacier segmentation."""
    
    def __init__(self, in_channels=5, hidden_dims=[32, 64, 32, 16], dropout_rate=0.2):
        super().__init__()
        
        layers = []
        prev_dim = in_channels
        
        # Build hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    """U-Net model for semantic segmentation of satellite imagery."""
    
    def __init__(self, in_channels=5, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)
        self.enc4 = self._block(256, 512)
        
        # Bottleneck
        self.bottleneck = self._block(512, 1024)
        
        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec1 = self._block(1024, 512)  # 512 + 512 (skip)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = self._block(512, 256)  # 256 + 256 (skip)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._block(256, 128)  # 128 + 128 (skip)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = self._block(128, 64)  # 64 + 64 (skip)
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Pooling layer for encoder
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder with skip connections
        dec1 = self.up1(bottleneck)
        dec1 = torch.cat((dec1, enc4), dim=1)
        dec1 = self.dec1(dec1)
        
        dec2 = self.up2(dec1)
        dec2 = torch.cat((dec2, enc3), dim=1)
        dec2 = self.dec2(dec2)
        
        dec3 = self.up3(dec2)
        dec3 = torch.cat((dec3, enc2), dim=1)
        dec3 = self.dec3(dec3)
        
        dec4 = self.up4(dec3)
        dec4 = torch.cat((dec4, enc1), dim=1)
        dec4 = self.dec4(dec4)
        
        # Final layer
        out = self.final(dec4)
        return torch.sigmoid(out)
    
    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ model for semantic segmentation."""
    
    def __init__(self, in_channels=5, out_channels=1):
        super(DeepLabV3Plus, self).__init__()
        
        # Use a lighter backbone for model size constraints
        self.backbone = nn.Sequential(
            # Initial conv to increase channels
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Downsample blocks
            self._make_layer(64, 128, stride=2),
            self._make_layer(128, 256, stride=2),
            self._make_layer(256, 512, stride=2),
            self._make_layer(512, 512, stride=1, dilation=2),
        )
        
        self.low_level_features = nn.Sequential(
            nn.Conv2d(128, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.aspp = self._build_aspp(512, 256)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, out_channels, 1)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def _make_layer(self, in_channels, out_channels, stride=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, 
                     padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, 
                     dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _build_aspp(self, in_channels, out_channels):
        # 1x1 convolution
        self.aspp_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Atrous convolutions with different rates
        self.aspp_3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.aspp_3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.aspp_3x3_3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Global average pooling
        self.aspp_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Project to reduce channels
        self.aspp_project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
        
        return nn.ModuleList([self.aspp_1x1, self.aspp_3x3_1, self.aspp_3x3_2, self.aspp_3x3_3, self.aspp_pool])
        
    def forward(self, x):
        # Save input size for final upsampling
        input_size = x.shape[-2:]
        
        # Extract features
        x1 = self.backbone[0:3](x)  # Low-level features
        x = self.backbone[3:](x1)   # High-level features
        
        # Apply ASPP modules
        aspp_1x1_out = self.aspp_1x1(x)
        aspp_3x3_1_out = self.aspp_3x3_1(x)
        aspp_3x3_2_out = self.aspp_3x3_2(x)
        aspp_3x3_3_out = self.aspp_3x3_3(x)
        
        # Apply global pooling module separately (needs special handling for interpolation)
        global_pool = self.aspp_pool(x)
        global_pool = F.interpolate(global_pool, size=x.shape[-2:], mode='bilinear', align_corners=False)
        
        # Concatenate all ASPP results
        x = torch.cat([aspp_1x1_out, aspp_3x3_1_out, aspp_3x3_2_out, aspp_3x3_3_out, global_pool], dim=1)
        
        # Apply projection
        x = self.aspp_project(x)
        
        # Process low-level features
        x1 = self.low_level_features(x1)
        
        # Upsample high-level features
        x = F.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        
        # Concatenate low and high level features
        x = torch.cat([x, x1], dim=1)
        
        # Decoder
        x = self.decoder(x)
        
        # Final upsampling to original image size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return self.sigmoid(x)