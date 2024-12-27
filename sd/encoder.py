import torch
from torch import nn
from torch.nn import functional as f
import torch
from decoder import VAE_AttentionBlock,VAE_ResidualBlock
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3,128,kernel_size=3,padding=1),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            nn.Conv2d(3,128,kernel_size=3,padding=0,stride=2),
            VAE_ResidualBlock(128,256),
            VAE_ResidualBlock(256,256),
            nn.Conv2d(256,256,kernel_size=3,padding=0,stride=2),
            VAE_ResidualBlock(256,512),
            VAE_ResidualBlock(512,512),
            nn.Conv2d(512,512,kernel_size=3,padding=0,stride=2),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512),
            nn.GroupNorm(32,512),
            nn.SiLU(32,512),
            nn.Conv2d(512,8,kernel_size=3,padding=1),
            nn.Conv2d(512,8,kernel_size=3,padding=0)

            
            
            
            
            

        )
        self.to(device)
        
