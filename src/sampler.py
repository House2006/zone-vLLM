from __future__ import annotations

import torch

class Sampler:
    def __call__(
        self,
        logits:torch.Tensor,
        temperatures: torch.Tensor | None = None,
        top_ps: torch.Tensor | None = None,
        top_ks: torch.Tensor | None = None
    ) -> torch.Tensor:
        # 没有 top-k,top-p 则为贪心
        if  top_ps is None and top_ks is None:
            return torch.argmax(logits, dim=-1)
        
        # temperature=0 视为贪心
        if temperatures is not None and torch.all(temperatures == 0):
            return torch.argmax(logits, dim=-1)

        logits = logits.float()
        # 按照 temperature 放缩，调整 logits 分布的方差
        if temperatures is not None:
            temperatures = torch.where(temperatures <= 0, 1.0, temperatures)
            logits = logits / temperatures.unsqueeze(1)
        
        # 将不满足条件的分数置为 -inf，其 softmax 概率为 0
        logits = _top_k_top_p(logits, top_ps, top_ks)
        # probs[B,V]
        probs = torch.softmax(logits, dim=-1)
        # 逐个Batch按概率分布采样，返回 token ID，即下标 index[B]
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

def _top_k_top_p(
    logits: torch.Tensor,
    top_p: torch.Tensor | None = None,
    top_k: torch.Tensor | None = None,
) -> torch.Tensor:
    vocab_size = logits.size(-1)
    # 降序排序，保留原顺序的索引
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    if top_k is not None:
        sorted_logits = _apply_top_k(sorted_logits, top_k, vocab_size)
    if top_p is not None:
        sorted_logits = _apply_top_p(sorted_logits, top_p)
    # 按索引还原顺序，保证 token 在概率分布中的位置对应 token ID
    return torch.empty_like(sorted_logits).scatter_(1, sorted_idx, sorted_logits)

def _apply_top_k(sorted_logits: torch.Tensor, top_k: torch.Tensor,vocab_size: int) -> torch.Tensor:
    # 处理异常条件 k<=0 和 k>vocab_size 
    k = torch.where(top_k <= 0, vocab_size, top_k)
    k = torch.clamp(k, max=vocab_size).to(torch.long)

    # 得到第 k 大的元素的值
    cutoff = sorted_logits.gather(1, (k - 1).unsqueeze(1))
    # 保留数量可能大于 k ,重复值会被保留
    mask = sorted_logits < cutoff
    min_val = torch.finfo(sorted_logits.dtype).min
    return sorted_logits.masked_fill(mask, min_val)


def _apply_top_p(sorted_logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    # 处理异常条件 top_p < 0  
    p = torch.where(top_p <= 0, 1.0, top_p)
    # softmax 得到概率分布
    probs = sorted_logits.softmax(dim=-1)
    
    #  只保留前缀和小于等于 p 的部分
    mask = probs.cumsum(dim=-1) > p.unsqueeze(1)
    mask[:, 0] = False  # 至少保留一个元素
    # 将超出 p 的部分置为 -inf
    min_val = torch.finfo(sorted_logits.dtype).min
    return sorted_logits.masked_fill(mask, min_val)