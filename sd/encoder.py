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
            nn.Conv2d(128,128,kernel_size=3,padding=0,stride=2),
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
            nn.SiLU(),
            nn.Conv2d(512,8,kernel_size=3,padding=1),
            nn.Conv2d(8,8,kernel_size=3,padding=0)

            
            
            
            
            

        )
        self.to(device)

        def forward(self,x:torch.Tensor,noise:torch.Tensor)->torch.Tensor:
            x = x.to(self.device)
            noise = noise.to(self.device)
            for module in self:
                if getattr(module,'stride',None)==(2,2):
                    x=F.padding(x,(0,1,0,1))
                x=module(x)
            mean,log_variance=torch.chunk(x,2,dim=1)
            log_variance=torch.clamp(log_variance,-30,20)
            variance=log_variance.exp()
            std=variance.sqrt()
            x=mean+std*noise
            x*=0.18215
            return x





        
