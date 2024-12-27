import torch
from torch import nn
from torch.nn import functional as f
import torch
from decoder import VAE_AttentionBlock,VAR_ResidualBlock
class VAE_Encoder(nn.Sequential):
    