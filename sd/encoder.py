import torch
from torch import nn
from torch.nn import functional as f
import torch
from decoder import VAE_AttentionBlock,VAE_ResidualBlock
class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3,128,kernel_size=3,padding=1),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            nn.Conv2d(3,128,kernel_size=3,padding=0,stride=2),
            VAE_ResidualBlock(128,256),
            VAE_ResidualBlock(128,256),
            
            
            
            

        )
