# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import Optional
import torch
import triton
import triton.language as tl


@triton.jit
def group_topk_score_kernel(input_ptr,
                            bias_ptr,
                            topk_weight_ptr,
                            topk_ids_ptr,
                            scale,
                            eps,
                            N: tl.constexpr,
                            K: tl.constexpr,
                            G: tl.constexpr,
                            GK: tl.constexpr,
                            DIV: tl.constexpr,
                            ):
    pid = tl.program_id(axis=0)
    GS: tl.constexpr = N // G
    k: tl.constexpr = K // GK

    logit = tl.load(input_ptr + pid * N + tl.arange(0, N))
    x = tl.sigmoid(logit)
    if bias_ptr is not None:
        b = tl.load(bias_ptr + tl.arange(0, N))
        m = tl.reshape(x + b, (G, GS))
    else:
        m = tl.reshape(x, (G, GS))
    o = tl.reshape(x, (G, GS))

    gt = tl.topk(m, k, dim=1)

    gts = tl.sum(gt, 1)
    gtst = tl.topk(gts, GK, dim=0)
    sum_min_value = tl.min(gtst)

    filling = tl.where( (gts[:, None] >= sum_min_value), o, -1.0)
    filling = tl.reshape(filling, [N])
    t = tl.min(tl.topk(filling, K, dim=0))
    filling = tl.where(filling >= t, filling, -1.0)

    score = filling / (tl.sum(tl.maximum(filling, 0.0)) + eps)
    mask = filling >= 0
    binary_mask = tl.where(mask, 1, 0)
    acc = tl.cumsum(binary_mask, 0) - 1

    if DIV:
        tl.store(topk_weight_ptr + pid * (K + 1) + K, 1.0/scale)
    else:
        tl.store(topk_weight_ptr + pid * (K + 1) + K, 1.0)
    tl.store(topk_ids_ptr + pid * (K + 1) + K, N)

    if tl.sum(binary_mask) > K:
        fillings = filling.to(tl.float64) + tl.arange(0, N).to(tl.float64) * 1e-12
        ts = tl.min(tl.topk(fillings, K, dim=0))
        filling = tl.where(fillings >= ts, filling, -1.0)

        score = filling / (tl.sum(tl.maximum(filling, 0.0)) + eps)
        mask = filling >= 0
        binary_mask = tl.where(mask, 1, 0)
        acc = tl.cumsum(binary_mask, 0) - 1
        if DIV:
            tl.store(topk_weight_ptr + pid * (K + 1) + acc, score, mask=mask)
        else:
            tl.store(topk_weight_ptr + pid * (K + 1) + acc, score * scale , mask=mask)
        tl.store(topk_ids_ptr + pid * (K + 1) + acc, tl.arange(0, N), mask=mask)
    else:
        if DIV:
            tl.store(topk_weight_ptr + pid * (K + 1) + acc, score, mask=mask)
        else:
            tl.store(topk_weight_ptr + pid * (K + 1) + acc, score * scale, mask=mask)
        tl.store(topk_ids_ptr + pid * (K + 1) + acc, tl.arange(0, N), mask=mask)


def triton_group_topk_score(x: torch.Tensor,
                            k: int,
                            expert_bias: Optional[torch.Tensor]=None,
                            num_groups=32,
                            group_topk=4,
                            scaling_factor=1.0,
                            score_function='sigmoid',
                            num_shared_experts=1,
                            div=True,
                            eps=1e-20):
    """
    calculate topk.
    Args:
        x: input tensor.
        expert_bias: expert bias
        k: topk
    Returns:
        topk_weights:  
        topk_ids: 
    """
    device = x.device
    M, N = x.shape
    assert x.is_contiguous() and score_function == 'sigmoid' and num_shared_experts == 1

    topk_weights = torch.empty((M, k + 1), device=device, dtype=torch.float32)
    topk_ids = torch.empty((M, k + 1), device=device, dtype=torch.int32)
    grid = (M, )
    group_topk_score_kernel[grid](
        x,
        expert_bias,
        topk_weights,
        topk_ids,
        scaling_factor,
        eps,
        N,
        k,
        num_groups,
        group_topk,
        div,
        num_stages=2,
        num_warps=2)
    return topk_weights, topk_ids
