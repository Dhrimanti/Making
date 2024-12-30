import torch
from torch import nn 
from torch.nn import functional as F 
from attention import SelfAttention





class CLIP(nn.Module):
    def __init__(self):
        self.embedding=CLIPEmbedding(49408,768,77)
        self.layers=nn.Module([
            CLIPLayer(12,768) for i in range(12)
         ])
        self.layernorm=nn.LayerNorm(768)
    def forward(slef,tokens:torch.LongTensor)->torch.FloatTensor:
        tokens=tokens.type(torch.long)
        state=self.embedding(tokens)
        for layer in self.layers:
            state=layer(state)

        output=self.layernorm(state)
        return output
