import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class ImageEncoder(nn.Module):
    def __init__(self, latent_size, pretrained):
        super(ImageEncoder, self).__init__()

        if pretrained:
            self.encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.encoder = efficientnet_b0(weights=None)
        
        self.linear = nn.Sequential(nn.Linear(1000, latent_size),
                                    nn.LayerNorm(latent_size), nn.Tanh())
                                
    def forward(self, x):
                
        x = self.encoder(x)
        
        out = self.linear(x)
            
        return out


