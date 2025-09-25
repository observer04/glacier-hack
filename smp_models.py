import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from .models import PixelANN

class SMPModel(nn.Module):
    """Wrapper for segmentation_models_pytorch models with 5-channel input."""
    
    def __init__(self, architecture='Unet', encoder_name='efficientnet-b3', 
                 encoder_weights='imagenet', in_channels=5, classes=1, activation='sigmoid'):
        super().__init__()
        
        # Create SMP model
        model_class = getattr(smp, architecture)
        self.model = model_class(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation
        )
        
        # If using pretrained weights on 3-channel, adapt first layer for 5 channels
        if encoder_weights is not None and in_channels != 3:
            self._adapt_first_layer(in_channels)
    
    def _adapt_first_layer(self, in_channels):
        """Adapt first convolutional layer for different input channels."""
        first_conv = None
        
        # Find first conv layer in encoder
        for name, module in self.model.encoder.named_modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                break
        
        if first_conv is not None and first_conv.in_channels != in_channels:
            # Create new conv layer with correct input channels
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            
            # Initialize new channels by averaging existing weights
            with torch.no_grad():
                if in_channels > 3:
                    # Replicate RGB weights for extra channels
                    new_conv.weight[:, :3] = first_conv.weight
                    new_conv.weight[:, 3:] = first_conv.weight[:, :in_channels-3]
                else:
                    # Average RGB weights for fewer channels
                    new_conv.weight = first_conv.weight[:, :in_channels]
                
                if first_conv.bias is not None:
                    new_conv.bias = first_conv.bias.clone()
            
            # Replace first conv layer
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = dict(self.model.encoder.named_modules())[parent_name]
            setattr(parent, child_name, new_conv)
    
    def forward(self, x):
        return self.model(x)

def create_smp_model(model_type='unet_efficientb3', **kwargs):
    """Factory function to create SMP models."""
    
    configs = {
        'unet_efficientb3': {
            'architecture': 'Unet',
            'encoder_name': 'efficientnet-b3',
            'encoder_weights': 'imagenet'
        },
        'unet_efficientb4': {
            'architecture': 'Unet', 
            'encoder_name': 'efficientnet-b4',
            'encoder_weights': 'imagenet'
        },
        'unetpp_efficientb3': {
            'architecture': 'UnetPlusPlus',
            'encoder_name': 'efficientnet-b3', 
            'encoder_weights': 'imagenet'
        },
        'deeplabv3p_efficientb3': {
            'architecture': 'DeepLabV3Plus',
            'encoder_name': 'efficientnet-b3',
            'encoder_weights': 'imagenet'
        },
        'fpn_efficientb3': {
            'architecture': 'FPN',
            'encoder_name': 'efficientnet-b3',
            'encoder_weights': 'imagenet'  
        }
    }
    
    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}")
    
    config = configs[model_type]
    config.update(kwargs)
    
    return SMPModel(**config)