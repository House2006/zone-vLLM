from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

class SiluAndMul(nn.Module):
    def forward(self,gate:torch.Tensor,value:torch.Tensor) -> torch.Tensor:
        return F.silu(gate)*value
    
