from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, TypeVar

import torch
from safetensors import safe_open
from torch import nn
from tqdm.auto import tqdm

__all__ = [
    "init_context",
    "load_model",
    "safetensors_weights_iterator",
]

# ----------------模型侧：参数张量初始化----------------

def _skip_reset_parameters(self: nn.Module) -> None:
    """用于替换 Linear/Embedding 的随机初始化"""
    pass

@contextmanager
def init_context(
    dtype: torch.dtype,
    device: torch.device | str = "cpu",
) -> Iterator[None]:
    """建模型时的临时设置全局 torch 环境"""
    previous_dtype = torch.get_default_dtype()
    linear_reset = nn.Linear.reset_parameters
    embedding_reset = nn.Embedding.reset_parameters

    # 张量数据类型设置
    # 关闭随机数初始化，写入权重时会直接覆盖
    # 参数张量存在哪里：“cpu”全部写入系统内存；“meta”只分配空壳不占内存
    torch.set_default_dtype(dtype)
    nn.Linear.reset_parameters = _skip_reset_parameters
    nn.Embedding.reset_parameters = _skip_reset_parameters
    try:
        with torch.device(device):
            yield
    finally:
        torch.set_default_dtype(previous_dtype)
        nn.Linear.reset_parameters = linear_reset
        nn.Embedding.reset_parameters = embedding_reset


# ----------------文件侧：加载权重到内存----------------

def _weight_files(folder: Path) -> list[Path]:
    """统计所有权重文件的路径"""
    index = folder / "model.safetensors.index.json"
    
    # 大模型分片存储权重
    if index.is_file():
        index_data = json.loads(index.read_text(encoding="utf-8"))
        weight_map = index_data["weight_map"]
        # 同一分片可能被多个权重名引用，先去重再排序
        return sorted({folder / name for name in weight_map.values()})
    
    # 小模型单文件存储权重
    files = sorted(folder.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no safetensors weights found in {folder}")
    return files

def safetensors_weights_iterator(
    hf_folder: str | Path,
    *,
    desc: str | None = "Loading weights",
) -> Iterator[tuple[str, torch.Tensor]]:
    """依次从每个权重文件流式加载权重张量，内存缓冲区仅需一个Tensor大小"""
    files = _weight_files(Path(hf_folder))
    total_bytes = sum(path.stat().st_size for path in files)

    with tqdm(
        total=total_bytes,
        desc=desc,
        unit="B",
        unit_scale=True,
        disable=desc is None,
    ) as bar:
        for path in files:  
            # 遍历路径下所有模型文件
            with safe_open(path, framework="pt", device="cpu") as shard:
            # 使用 safetensors 提供的 safe_open 打开并读取模型文件，以迭代器形式输出
                for name in shard.keys():
                    tensor = shard.get_tensor(name)
                    bar.update(tensor.numel() * tensor.element_size())
                    yield name, tensor

        # safetensors 文件头也计入文件大小，最后把进度条补到 100%。
        if bar.n < total_bytes:
            bar.update(total_bytes - bar.n)




# -------对接：调用 model.load_weights 将缓冲区的 safetensors 权重写入模型内部--------- 

class WeightLoadable(Protocol):
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None: ...

# 泛型设置：必须有 load_weights 方法
ModelT = TypeVar("ModelT", bound=WeightLoadable)

def load_model(model: ModelT, hf_folder: str | Path) -> ModelT:
    """将目录中的 checkpoint 写入模型，名称映射、形状校验、类型转换由 load_weights 完成。"""
    model.load_weights(safetensors_weights_iterator(hf_folder))
    return model