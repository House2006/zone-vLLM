from __future__ import annotations
import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self,hidden_size:int,eps:float):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.hidden_size = hidden_size
            self.eps = eps

    def _rms_normalize(self,x:torch.Tensor)->torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight.to(x.dtype)

class BasicRMSNorm(RMSNorm):   
    def forward(self,x:torch.Tensor)->torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        out32 = self._rms_normalize(x32)
        return out32.to(orig_dtype)

class FusedRMSNorm(RMSNorm):
    def forward(self,x:torch.Tensor,res:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        orig_dtype = x.dtype
        x32 = x.float()
        res32 = res.float()
        new_res32 = res32 + x32
        out32 = self._rms_normalize(new_res32)
        return (out32.to(orig_dtype),new_res32.to(orig_dtype))



