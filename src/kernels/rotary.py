from __future__ import annotations
import torch
from torch import nn
from typing import cast

def _rotate(x:torch.Tensor,sin:torch.Tensor,cos:torch.Tensor)->torch.Tensor:
    x1,x2 = x.chunk(2,dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin),dim=-1)  

class RotaryEmbedding(nn.Module):
    def __init__ (self, head_size:int, rotary_dim:int, max_pos:int, base:float):
        # q, k, v 维度
        # rotary_dim 参与旋转编码的维度
        # max_pos 查找表支持的最大序列长度
        # base 角速度计算，理论max_pos = 2*π*base，与rotary_dim无关
        self.head_size = head_size
        self.max_pos = max_pos
        # 计算查找表
        ang_freq = 1/(base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
        pos_code = torch.arange(0, max_pos)
        angle = torch.outer(pos_code, ang_freq)
        self.register_buffer(
            "sin_cos",
            torch.cat(
                (angle.sin(),
                angle.cos()),
                dim = -1
            )
        )  

    def forward(self, query:torch.Tensor, key:torch.Tensor, position:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        pos = position.reshape(-1)
        orig_q_shape = query.shape
        orig_k_shape = key.shape
        # q[B*S, Hq, d]
        q = query.flatten(0, 1)
        k = key.flatten(0, 1)

        # 按需查找
        cache = cast(torch.Tensor, self.sin_cos)    # 告知 Pylance，cache是Tensor
        sin_cos = torch.index_select(cache, 0, pos).to(
            device=query.device,dtype=query.dtype
        )# 查找表始终默认float32，需要类型转换
        sin, cos = sin_cos.chunk(2, dim=-1)
        sin = sin.unsqueeze(-2) # 为q,v的Head维度广播
        cos = cos.unsqueeze(-2)
        q_rot = _rotate(q, sin, cos)
        k_rot = _rotate(k, sin, cos)
        return q_rot.reshape(orig_q_shape),k_rot.reshape(orig_k_shape)
        

            

