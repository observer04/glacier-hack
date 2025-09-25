import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# EfficientNet-style building blocks
class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Conv Block with SE attention."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, expand_ratio=6, se_ratio=0.25, drop_rate=0.0):
        super().__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        self.expand_conv = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        ) if expand_ratio != 1 else nn.Identity()
        
        # Depthwise conv
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, stride, 
                     padding=kernel_size//2, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True)
        )
        
        # Squeeze-and-Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, se_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_channels, expanded_channels, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.project_conv = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Stochastic depth
        self.drop_path = nn.Dropout2d(drop_rate) if drop_rate > 0 else nn.Identity()
    
    def forward(self, x):
        identity = x
        
        # Expansion
        x = self.expand_conv(x)
        
        # Depthwise conv
        x = self.depthwise_conv(x)
        
        # SE attention
        se_weights = self.se(x)
        x = x * se_weights
        
        # Output projection
        x = self.project_conv(x)
        
        # Stochastic depth + residual
        if self.use_residual:
            x = self.drop_path(x) + identity
        
        return x

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

class EfficientUNet(nn.Module):
    """Efficient U-Net with MBConv blocks and attention."""
    
    def __init__(self, in_channels=5, out_channels=1, width_mult=1.0):
        super().__init__()
        
        # Calculate channel dimensions
        def make_divisible(v, divisor=8):
            return max(divisor, int(v + divisor / 2) // divisor * divisor)
        
        channels = [make_divisible(c * width_mult) for c in [32, 64, 128, 256, 512]]
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True)
        )
        
        # Encoder - EfficientNet-style blocks
        self.enc1 = nn.Sequential(
            MBConvBlock(channels[0], channels[0], 3, 1, 1, 0.25, 0.1),
            MBConvBlock(channels[0], channels[1], 3, 2, 6, 0.25, 0.1)
        )
        self.enc2 = nn.Sequential(
            MBConvBlock(channels[1], channels[1], 3, 1, 6, 0.25, 0.1),
            MBConvBlock(channels[1], channels[2], 3, 2, 6, 0.25, 0.1)
        )
        self.enc3 = nn.Sequential(
            MBConvBlock(channels[2], channels[2], 5, 1, 6, 0.25, 0.2),
            MBConvBlock(channels[2], channels[3], 5, 2, 6, 0.25, 0.2)
        )
        self.enc4 = nn.Sequential(
            MBConvBlock(channels[3], channels[3], 5, 1, 6, 0.25, 0.2),
            MBConvBlock(channels[3], channels[4], 3, 2, 6, 0.25, 0.3)
        )
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            MBConvBlock(channels[4], channels[4], 3, 1, 6, 0.25, 0.3),
            MBConvBlock(channels[4], channels[4], 3, 1, 6, 0.25, 0.3)
        )
        
        # Decoder with feature fusion
        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], 2, 2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(channels[3] * 2, channels[3], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.SiLU(inplace=True),
            MBConvBlock(channels[3], channels[3], 3, 1, 4, 0.25, 0.2)
        )
        
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, 2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(channels[2] * 2, channels[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.SiLU(inplace=True),
            MBConvBlock(channels[2], channels[2], 3, 1, 4, 0.25, 0.1)
        )
        
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, 2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(channels[1] * 2, channels[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.SiLU(inplace=True),
            MBConvBlock(channels[1], channels[1], 3, 1, 4, 0.25, 0.1)
        )
        
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, 2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(channels[0] * 2, channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0], channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.SiLU(inplace=True)
        )
        
        # Final prediction head with deep supervision
        self.final = nn.Sequential(
            nn.Conv2d(channels[0], channels[0] // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0] // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels[0] // 2, out_channels, 1)
        )
        
        # Deep supervision heads
        self.aux_head4 = nn.Conv2d(channels[3], out_channels, 1)
        self.aux_head3 = nn.Conv2d(channels[2], out_channels, 1)
        
    def forward(self, x, deep_supervision=False):
        input_size = x.shape[-2:]
        
        # Encoder
        x0 = self.stem(x)
        x1 = self.enc1(x0)  # 1/2
        x2 = self.enc2(x1)  # 1/4  
        x3 = self.enc3(x2)  # 1/8
        x4 = self.enc4(x3)  # 1/16
        
        # Bottleneck
        x = self.bottleneck(x4)
        
        # Decoder
        x = self.up4(x)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)
        
        aux4 = None
        if deep_supervision:
            aux4 = torch.sigmoid(F.interpolate(self.aux_head4(x), input_size, mode='bilinear', align_corners=False))
        
        x = self.up3(x)
        x = torch.cat([x, x3], dim=1) 
        x = self.dec3(x)
        
        aux3 = None
        if deep_supervision:
            aux3 = torch.sigmoid(F.interpolate(self.aux_head3(x), input_size, mode='bilinear', align_corners=False))
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        
        # Final prediction
        out = torch.sigmoid(self.final(x))
        
        if deep_supervision:
            return out, aux4, aux3
        return out

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
        # Split into stem + layers to tap low-level (128-ch) features cleanly
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(64, 128, stride=2)   # low-level features (1/2 res)
        self.layer2 = self._make_layer(128, 256, stride=2)  # (1/4 res)
        self.layer3 = self._make_layer(256, 512, stride=2)  # (1/8 res)
        self.layer4 = self._make_layer(512, 512, stride=1, dilation=2)

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
        inp_size = x.shape[-2:]

        # Extract features
        x0 = self.stem(x)               # (N,64,H,W)
        x1 = self.layer1(x0)            # (N,128,H/2,W/2) low-level
        x2 = self.layer2(x1)            # (N,256,H/4,W/4)
        x3 = self.layer3(x2)            # (N,512,H/8,W/8)
        x = self.layer4(x3)             # (N,512,H/8,W/8) high-level

        # ASPP on high-level features
        aspp_1x1_out = self.aspp_1x1(x)
        aspp_3x3_1_out = self.aspp_3x3_1(x)
        aspp_3x3_2_out = self.aspp_3x3_2(x)
        aspp_3x3_3_out = self.aspp_3x3_3(x)

        global_pool = self.aspp_pool(x)
        global_pool = F.interpolate(global_pool, size=x.shape[-2:], mode='bilinear', align_corners=False)

        # Concatenate ASPP results and project
        x = torch.cat([aspp_1x1_out, aspp_3x3_1_out, aspp_3x3_2_out, aspp_3x3_3_out, global_pool], dim=1)
        x = self.aspp_project(x)

        # Process low-level features
        low = self.low_level_features(x1)

        # Upsample high-level to low-level resolution and fuse
        x = F.interpolate(x, size=low.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, low], dim=1)

        # Decoder
        x = self.decoder(x)

        # Final upsampling to original image size
        x = F.interpolate(x, size=inp_size, mode='bilinear', align_corners=False)

        return self.sigmoid(x)