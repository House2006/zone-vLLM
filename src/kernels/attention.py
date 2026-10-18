from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import nn

class Attention(nn.Module):
    def __init__(self, head_size:int, head_num:int, kv_head_num:int):
        self.head_num = head_num
        self.kv_head_num = kv_head_num
        self.scaling = head_size ** -0.5
    def forward(self, q:torch.Tensor,k:torch.Tensor,v:torch.Tensor)->torch.Tensor:
       # q[B,S,Hq,d]->q[B,Hq,S,d] 
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)      
        # Hq个头独立计算S个token的注意力分数
        out = F.scaled_dot_product_attention(q,k,v,is_causal=True,scale=self.scaling,enable_gqa=self.kv_head_num!=self.head_num)
        
        return out.transpose(1, 2).contiguous()
        