from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# 默认下载路径
DEFAULT_CACHE_DIR = Path.home() / "huggingface"

__all__ = [
    "DEFAULT_CACHE_DIR",
    "ModelSpec",
    "QWEN3_REGISTRY",
    "download_model",
    "download_models",
    "is_local_model",
    "resolve_model_path",
]

# 模型条目
@dataclass(frozen=True)
class ModelSpec:
    local_name: str
    repo_id: str

# 模型仓库
QWEN3_REGISTRY = (
    ModelSpec("Qwen3-0.6B", "Qwen/Qwen3-0.6B"),
    ModelSpec("Qwen3-1.7B", "Qwen/Qwen3-1.7B"),
    ModelSpec("Qwen3-4B", "Qwen/Qwen3-4B"),
)


def is_local_model(path: str | Path) -> bool:
    path = Path(path).expanduser()
    return path.is_dir() and (path / "config.json").is_file()




def resolve_model_path(model: str | Path, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path: 
    path = Path(model).expanduser()
    # 传入路径是否是本地模型文件的路径，是则直接返回该路径的Path对象，否则返回默认缓存位置
    return path if is_local_model(path) else cache_dir / path.name



def _is_complete_download(path: Path) -> bool:
    """逐个检查张量参数，判断该模型是否已经完整下载"""
    if not is_local_model(path) or not (path / "tokenizer.json").is_file():
        return False

    index = path / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8")).get(
                "weight_map", {}
            )
            shards = set(weight_map.values())
            return bool(shards) and all(
                isinstance(shard, str) and (path / shard).is_file()
                for shard in shards
            )
        except (OSError, json.JSONDecodeError, AttributeError, TypeError):
            return False

    return any(path.glob("*.safetensors"))



def download_model(spec: ModelSpec, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    target = cache_dir / spec.local_name
    # 该模型完整下载则直接返回
    if _is_complete_download(target): 
        return target

    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        spec.repo_id,
        local_dir=str(target),
        token=os.environ.get("HF_TOKEN"),
        endpoint=os.environ.get("HF_ENDPOINT") or None,  
    )
    return target


def download_models(specs: tuple[ModelSpec, ...] = QWEN3_REGISTRY) -> list[Path]:
    return [download_model(spec) for spec in specs]
