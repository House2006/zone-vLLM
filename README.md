# zone-vLLM

一个基于 PyTorch 的 LLM 推理引擎 

## 计划

- 完成单请求生成和 KV cache。
- 实现 PagedAttention 和 KV block 的分配、回收。
- 加入 continuous batching 和 chunked prefill。
- 加入 Prefill/Decode 分离和 MoE。
- 补充正确性测试和性能对比。

## 仓库结构

```text
src/
  config.py             引擎和生成配置
  model.py              Qwen3 模型结构
  model_loader.py       Safetensors 权重读取和加载
  model_hub.py          Qwen3 模型下载和路径处理
  sampler.py            greedy、top-k、top-p 采样
  kernels/
    activation.py       SwiGLU
    attention.py        基于 PyTorch SDPA 的 attention
    norm.py             RMSNorm
    rotary.py           RoPE
docs/
  project_roadmap.md   开发路线与评测计划
  rotary_embedding.md  RoPE 笔记
  model_architecture.md Qwen3 模型结构
  posts/               结构图
```

## 模型结构

见 [Qwen3 模型结构](docs/model_architecture.md)。

## 参考

- [vLLM](https://github.com/vllm-project/vllm)：PagedAttention、KV cache 和请求调度的实现参考。
- [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)：轻量推理引擎参考。
- [dzyy-vLLM](https://dlog.com.cn/posts/offer01/vllm%E4%BB%8B%E7%BB%8D/)：从零实现 vLLM 的系列。
- [Transformers Qwen3](https://huggingface.co/docs/transformers/model_doc/qwen3)：Qwen3 模型结构及配置参考。
- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：当前 attention 使用的 PyTorch 接口。
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)：PagedAttention 的设计背景。
