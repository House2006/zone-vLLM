from __future__ import annotations

from collections.abc import Callable, Iterator

import torch
from transformers import AutoTokenizer

from .config import EngineConfig 
from .model import Qwen3ForCausalLM
from .model_loader import init_context, load_model
from .sampler import Sampler

class Engine:
    def __init__(self,config:EngineConfig):
        # self.model_path = config.model_path
        # self.model_config = config.model_config
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype =  self._get_dtype()
        
        # 初始化分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path,trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id # 结束字符
        
        # 开辟空参数模型
        with init_context(self.dtype, self.device):
            self.model = Qwen3ForCausalLM(self.config.model_config)
        # 加载权重，自动转化精度
        load_model(self.model, self.config.model_path)
        self.model.eval()
        # 开启 CUDA Graph
        self._forward_model = self.model
        if not self.config.enforce_eager and self.device == torch.device("cuda:0"):
            self._forward_model = torch.compile(self._forward_model,mode="reduce-overhead")
        
        # 初始化采样器
        self.sampler = Sampler()
        
    
    def _get_dtype(self)->torch.dtype:
            # 如果 EngineConfig 指定精度
            if self.config.dtype != "auto":
                return getattr(torch, self.config.dtype)
            # 如果模型有默认精度
            model_dtype = getattr(self.config.model_config, "dtype", None)
            if isinstance(model_dtype, str):
                model_dtype = getattr(torch, model_dtype)
            # 否则使用 bfloat32
            return model_dtype if isinstance(model_dtype, torch.dtype) else torch.bfloat16


    @torch.inference_mode()
    def generate(
        self,         
        prompt,
        temperature,
        top_p,
        top_k,
        max_new_tokens):

        # 普通输入包装为1维张量
        def _pack(v,dtype): 
            return None if v is None else torch.tensor([v],device=self.device,dtype=dtype)
        temp_t = _pack(temperature,torch.float)
        top_p_t = _pack(top_p,torch.float)
        top_k_t = _pack(top_k,torch.long)
        
        ids = list(prompt) if isinstance(prompt,list) else self.tokenizer.encode(prompt)
        context_len = self.config.context_len
        # 截断到 input_ids的长度，塞进去
        if len(ids) >= context_len:
            ids = ids[-context_len:]
        input_ids = torch.empty((1,context_len),dtype=torch.long,device=self.device)
        input_ids[0,:len(ids)] = torch.tensor(ids,dtype=torch.long,device=self.device)
        positions = torch.arange(context_len,dtype=torch.long,device=self.device).unsqueeze(0)
        cur_len = len(ids)
        pending: list[int] = []

        # 循环退出条件：生成EOS；撑暴input_ids；输出token数达到上限
        for _ in range(max_new_tokens):
            output = self._forward_model(input_ids[:,:cur_len],positions[:,:cur_len])
            logits = self.model.compute_logits(output[:,-1:]).squeeze(1)    # output[B,S,D]->logits[B,V]
            next_id = int(self.sampler(logits,temp_t,top_p_t,top_k_t).item())
            if next_id == self.eos_token_id:
                break
            ids.append(next_id)
            pending.append(next_id)

            # 流式输出文本
            text  = self.tokenizer.decode(pending,skip_special_tokens=False)
            if text and not text.endswith("\ufffd"):
                count = len(pending)
                pending.clear()
                yield text, count
            
            # 超出上下文窗口，input_ids 无法容纳
            if cur_len >= context_len:
                break
            input_ids[0,cur_len] = next_id
            cur_len += 1            

        # flush 剩余 token
        if pending:
            yield self.tokenizer.decode(pending, skip_special_tokens=False), len(pending)



        
         
