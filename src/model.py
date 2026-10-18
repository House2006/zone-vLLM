from __future__ import annotations
from collections.abc import Iterable
from typing import cast

import torch
from torch import nn
from transformers.models.qwen3 import Qwen3Config

from .kernels.norm import BasicRMSNorm, FusedRMSNorm
from .kernels.rotary import RotaryEmbedding
from .kernels.attention import Attention
from .kernels.activation import SiluAndMul

class Qwen3MLP(nn.Module):
    def __init__(self, config:Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size,config.intermediate,bias=False)
        self.up_proj = nn.Linear(config.hidden_size,config.intermediate,bias=False)
        self.down_proj = nn.Linear(config.config.intermediate,config.hidden_size,bias=False)
        self.activate_func = SiluAndMul()
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activate_func.forward(self.gate_proj(x),self.up_proj(x)))

class Qwen3Attention(nn.Module):
    def __init__(self, config:Qwen3Config) -> None:
        super().__init__()
        self.q_head_num = config.num_attention_heads
        self.kv_head_num = (config.num_key_value_heads if config.num_key_value_heads is not None 
                            else self.q_head_num)
        
        configured_head_dim = getattr(config,"head_dim",None)
        self.head_dim = (configured_head_dim if configured_head_dim is not None
                        else config.hidden_dim)
        self.q_total_dim = self.q_head_num * self.head_dim
        self.kv_total_dim = self.kv_head_num * self.head_dim

        
        self.qkv_proj = nn.Linear(config.hidden_size, self.q_total_dim + 2 * self.kv_total_dim, bias=False)
        self.o_proj = nn.Linear(self.q_total_dim, config.hidden_size,bias=False)
        self.rotary_embed = RotaryEmbedding(
            self.head_dim,
            self.head_dim,
            config.max_position_embeddings,
            float(getattr(config,"rope_theta",1000000))
        )
        self.q_norm = BasicRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = BasicRMSNorm(self.head_dim, config.rms_norm_eps)
        self.attention = Attention(self.head_dim,
                                   self.q_head_num,
                                   self.kv_head_num
                                   )
        
    def forward(self,hidden_states:torch.Tensor,positions:torch.Tensor) -> torch.Tensor:        
        # hidden_states[B,S,D] --qkv_proj--> Q[B,S,Hq*d] | K[B,S,Hk*d] | V[B,S,Hv*d]
        q_total, k_total, v_total = self.qkv_proj(hidden_states).split(
            [self.q_total_dim,self.kv_total_dim, self.kv_total_dim],    # 一次生成 Q|K|V 后切割
            dim=-1
        )
        # Q[B,S,Hq*d] -> q[B,S,Hq,d] -- q_norm -- rotary_embedding -- attention --> o[B,S,Hq,d] -> O[B,S,Hq*d] 
        q = q_total.unflatten(-1,(self.q_head_num,self.head_dim))
        k = k_total.unflatten(-1,(self.kv_head_num,self.head_dim))
        v = v_total.unflatten(-1,(self.kv_head_num,self.head_dim))
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_embed(q, k, positions)        
        # O[B,S,Hq*d] --o_proj--> hidden_states[B,S,D]
        return self.o_proj(self.attention(q,k,v).flatten(-2,-1))

class Qwen3DecoderLayer(nn.Module):
    def __init__(self,config:Qwen3Config) -> None:
        super().__init__()
        self.self_attention = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = FusedRMSNorm(config.hidden_size,config.rms_norm_eps)
        self.post_attention_layernorm = FusedRMSNorm(config.hidden_size,config.rms_norm_eps)

    # 层间传递 hidden_states,residual: float[B,S,D]
    def forward(self,hidden_states:torch.Tensor,residual:torch.Tensor,positions:torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        x, res = self.input_layernorm(hidden_states, residual)
        x = self.self_attention(x, positions)
        x, res = self.post_attention_layernorm(x, res)
        x = self.mlp(x)
        return x, res

class Qwen3Model(nn.Module):
    def __init__(self,config:Qwen3Config) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size,config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.last_norm = FusedRMSNorm(config.hidden_size,config.rms_norm_eps)
    
    # input_ids: int[B,S]  
    def forward(self,input_ids:torch.Tensor,positions:torch.Tensor) -> torch.Tensor:
        # 独热编码 + 嵌入矩阵降维
        hidden_states = self.embed_tokens(input_ids) 
        # 首层残差传入0张量
        hidden_states, residual = self.layers(hidden_states,torch.zeros(hidden_states.shape))
        # 最终残差相加和归一化，丢弃残差
        hidden_states, _ = self.last_norm(hidden_states,residual)
        return hidden_states

class Qwen3ForCausalLM(nn.Module):
    def __init__(self,config:Qwen3Config) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size,config.vocab_size,bias=False)

    # 默认Backbone，需要时再算logits
    def forward(self,input_ids:torch.Tensor,positions:torch.Tensor) -> torch.Tensor:
        return self.model(input_ids,positions)
    
    def compute_logits(self,hidden_states:torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)