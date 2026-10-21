from __future__ import annotations
import os.path as path
from dataclasses import dataclass,field
# dataclasses Python数据类，自动生成_init_等样板代码
from transformers import AutoConfig
# AutoConfig HuggingFace工具，自动加载模型config.json, 填充Qwen3Config
from transformers.models.qwen3 import Qwen3Config
# PretrainedConfig 所有HF模型配置的基类

@dataclass
class EngineConfig:
    model_path: str
    model_config:Qwen3Config = field(init=False) # 实例化时不传入该参数
    # 引擎基本参数
    dtype:str = "auto"
    context_len: int = 2048
    enforce_eager: bool = True

    def _post__init_(self) -> None:
        self.model_path = path.expanduser(self.model_path)
        model_config = AutoConfig.from_pretrained(self.model_path,trust_remote_code=True)
        if not isinstance(model_config,Qwen3Config):
            raise TypeError(f"expected a Qwen3 config, got {type(model_config).__name__}")
        self.model_config = model_config
        # 初始化模型时传入 Qwen3ForCausalLM

@dataclass
class ForwardConfig:
    prompt: str
    max_new_tokens: int = 200
    temperature: float = 0.9
    top_p: float = 0.95
    top_k: int = 20



