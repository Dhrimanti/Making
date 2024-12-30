import torch 
from torch import nn
from troch.nn import functional as F
from attention import SelfAttention,CrossAttention


class TimeEmbedding(nn.Module):
    def __init__(self,d_model:int):
        super().__init__()
        self.linear_1=nn.Linear(n_embed,4*n_embed)
        self.linear_2=nn.Linear(4*n_embed,4*n_embed)
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=self.linear_1(x)
        x=F.silu(x)
        x=self.linear_2(x)
        return x
       


class Diffusion(nn.Module):
    self.time_embedding=TimeEmbedding(320)
    self.unet=UNET()
    self.final=UNET_OutputLayer()
    def forward(self,latent:torch.Tensor,context:torch.Tensor,time:torch.Tensor):
        time=self.time_embedding(time)
        output=self.unet.(latent,context,time)
        output=self.final(output)
        return output

