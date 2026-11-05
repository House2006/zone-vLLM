from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.run_utils import load_config, merge_cli  # noqa: E402

from src.config import EngineConfig  # noqa: E402
from src.engine import Engine  # noqa: E402
from src.model_hub import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    QWEN3_REGISTRY,
    download_models,
    resolve_model_path,
)

DEFAULTS = {
    "model": "Qwen3-0.6B",
    "prompt": "Hello, my name is",
    "max_new_tokens": 200,
    "temperature": 0.9,
    "top_p": 0.95,
    "top_k": 20,
    "enforce_eager": True,
    "dtype": "auto",
}


def run(cfg: dict) -> None:
    model_path = resolve_model_path(cfg["model"]) # 根据名字找到缓存文件的位置:当前目录下\默认缓存路径
    print(f"Loading {model_path} ...")
    engine = Engine(
        EngineConfig(
            model_path=str(model_path),
            enforce_eager=cfg["enforce_eager"],
            dtype=cfg["dtype"],
        ) # 这部分推理参数初始化时即传入。还会自动从model_path获取模型参数hf_config
    )

    prompt = cfg["prompt"]
    print(f"Prompt: {prompt}\n{prompt}", end="", flush=True)
    start = time.perf_counter()
    n = 0
    for text, count in engine.generate(
        prompt,
        max_new_tokens=cfg["max_new_tokens"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
    ): # 另一部分推理参数在运行时传入
        print(text, end="", flush=True)
        n += count
    elapsed = time.perf_counter() - start
    print(f"\n\n{n} tokens in {elapsed:.2f}s ({n / elapsed:.1f} tok/s)")


def main() -> None:
    p = argparse.ArgumentParser(description="vllm-v0 基础推理")
    p.add_argument("--config", default=None, help="YAML 配置文件路径")
    p.add_argument("--model", default=None)
    p.add_argument("--prompt", default=None)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--download", action="store_true")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    # 直接返回
    if args.list:
        print(f"Cache: {DEFAULT_CACHE_DIR}")
        print(f"Models: {', '.join(s.local_name for s in QWEN3_REGISTRY)}")
        return
    if args.download:
        print(f"Downloading to {DEFAULT_CACHE_DIR} ...")
        download_models()
        print("Done.")
        return

    # 多途径获取config配置信息：CLI指定值 > CLI 指定YAML文件 > 默认YAML文件 > 默认值
    config_path = Path(args.config) if args.config else HERE / "config.yaml"
    cfg = load_config(config_path, DEFAULTS) 
    merge_cli(cfg, args, ("model", "prompt", "max_new_tokens", "temperature", "top_p", "top_k")) # 根据 CLI 覆盖配置文件
    run(cfg)


if __name__ == "__main__":
    main()
