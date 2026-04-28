# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import Optional
import torch
import triton
import triton.language as tl


@triton.jit
def group_topk_score_kernel(
    input_ptr,
    bias_ptr,
    topk_weight_ptr,
    topk_ids_ptr,
    scale,
    eps,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    GK: tl.constexpr,
    SHARE_EXPERTS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    GS: tl.constexpr = N // G
    k: tl.constexpr = K // GK

    logit = tl.load(input_ptr + pid * N + tl.arange(0, N)).to(tl.float32)
    x = tl.sigmoid(logit)
    b = tl.load(bias_ptr + tl.arange(0, N)).to(tl.float32)
    m = tl.reshape(x + b, (G, GS))

    gt = tl.topk(m, k, dim=1)

    gts = tl.sum(gt, 1)
    gtst = tl.topk(gts, GK, dim=0)
    sum_min_value = tl.min(gtst)

    group_filling = tl.where((gts[:, None] >= sum_min_value), m, -1.0)
    group_filling = tl.reshape(group_filling, [N])
    t = tl.min(tl.topk(group_filling, K, dim=0))
    mask = group_filling >= t

    hit_count = tl.sum(tl.where(mask, 1, 0))

    if hit_count > K:
        group_fillings = (
            group_filling.to(tl.float64) - tl.arange(0, N).to(tl.float64) * 1e-12
        )
        ts = tl.min(tl.topk(group_fillings, K, dim=0))
        masks = group_fillings >= ts
    else:
        masks = mask

    hitmap = tl.where(masks, 1, 0)
    filling = tl.where(masks, x, 0.0)
    score = filling / (tl.sum(filling) + eps)
    acc = tl.cumsum(hitmap, 0) - 1

    if SHARE_EXPERTS == 1:
        tl.store(topk_weight_ptr + pid * (K + 1) + acc, score, mask=masks)
        tl.store(topk_ids_ptr + pid * (K + 1) + acc, tl.arange(0, N), mask=masks)
        tl.store(topk_weight_ptr + pid * (K + 1) + K, 1.0 / scale)
        tl.store(topk_ids_ptr + pid * (K + 1) + K, N)
    else:
        tl.store(topk_weight_ptr + pid * K + acc, score, mask=masks)
        tl.store(topk_ids_ptr + pid * K + acc, tl.arange(0, N), mask=masks)


def triton_group_topk_score(
    x: torch.Tensor,
    k: int,
    expert_bias: torch.Tensor,
    num_groups=8,
    group_topk=4,
    scaling_factor=1.0,
    score_function="sigmoid",
    num_shared_experts=0,
    eps=1e-20,
):
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
    assert x.is_contiguous() and score_function == "sigmoid"
    assert num_shared_experts in (0, 1)
    topk_weights = torch.empty(
        (M, k + num_shared_experts), device=device, dtype=torch.float32
    )
    topk_ids = torch.empty(
        (M, k + num_shared_experts), device=device, dtype=torch.int32
    )

    grid = (M,)
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
        num_shared_experts,
        num_stages=1,
        num_warps=1,
    )

    return topk_weights, topk_ids
