# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def topk_forward_kernel(input_ptr, value_ptr, index_ptr,
                        N: tl.constexpr,
                        K: tl.constexpr):
    pid = tl.program_id(axis=0)

    xo = tl.load(input_ptr + pid * N + tl.arange(0, N))

    x = xo
    for i in range(K):
        val = tl.max(x, 0)
        idx = tl.argmax(x, 0)
        tl.store(value_ptr + pid * K + i, val)
        tl.store(index_ptr + pid * K + i, idx)
        x = tl.where(x == val, -2e38, x)

    if tl.sum(tl.where(x < -1e38, 1, 0)) > K:
        y = xo.to(tl.float64) - tl.arange(0, N).to(tl.float64) * 1e-12
        for i in range(K):
            val = tl.max(y, 0)
            idx = tl.argmax(y, 0)
            tl.store(value_ptr + pid * K + i, val)
            tl.store(index_ptr + pid * K + i, idx)
            y = tl.where(y == val, -2e38, y)


def triton_topk_forward(x, k, dim=-1):
    """
    calculate topk.
    Args:
        x: input tensor.
        k: topk
    Returns:
        values: topk values 
        indices: topk indices
    """
    device = x.device
    shape = x.shape
    assert dim == -1 and len(shape) <= 3
    assert x.is_contiguous()
    if len(shape) == 3:
        M, B, N = shape
        g = M * B
        values = torch.empty((M, B, k), device=device, dtype=x.dtype)
        indices = torch.empty((M, B, k), device=device, dtype=torch.int64)
    else:
        M, N = shape
        g = M
        values = torch.empty((M, k), device=device, dtype=x.dtype)
        indices = torch.empty((M, k), device=device, dtype=torch.int64)
    grid = (g,)
    topk_forward_kernel[grid](
        x,
        values,
        indices,
        N,
        k,
        num_stages=2,
        num_warps=2
    )
    return values, indices


@triton.jit
def topk_backward_kernel(grad_ptr, index_ptr, dx_ptr,
                         N: tl.constexpr,
                         K: tl.constexpr):
    pid = tl.program_id(axis=0)

    grad = tl.load(grad_ptr + pid * K + tl.arange(0, K))
    index = tl.load(index_ptr + pid * K + tl.arange(0, K))
    tl.store(dx_ptr + pid * N + index, grad)


def triton_topk_backward(grad_output, indices, N, dim=-1):
    """
    topk backward.
    Args:
        grad_output: grad tensor of values.
        indices: topk indices
        N: dim
    Returns:
        dx
    """
    device = grad_output.device
    shape = grad_output.shape
    assert dim == -1 and len(shape) <= 3
    assert grad_output.is_contiguous()
    if len(shape) == 3:
        M, B, k = shape
        g = M * B
        dx = torch.zeros((M, B, N), device=device, dtype=grad_output.dtype)
    else:
        M, k = shape
        g = M
        dx = torch.zeros((M, N), device=device, dtype=grad_output.dtype)
    grid = (g,)
    topk_backward_kernel[grid](
        grad_output,
        indices,
        dx,
        N,
        k,
        num_stages=2,
        num_warps=2
    )
    return dx


@triton.jit
def group_topk_score_forward_kernel(input_ptr, bias_ptr, prob_ptr, map_ptr,
                                    scale,
                                    eps,
                                    N: tl.constexpr,
                                    K: tl.constexpr,
                                    G: tl.constexpr,
                                    GK: tl.constexpr,
                                    BIAS: tl.constexpr
                                    ):
    pid = tl.program_id(axis=0)
    GS: tl.constexpr = N // G
    k: tl.constexpr = K // GK

    logit = tl.load(input_ptr + pid * N + tl.arange(0, N))
    x = tl.sigmoid(logit)
    if BIAS:
        b = tl.load(bias_ptr + tl.arange(0, N))
    else:
        b = 0.0
    xb = tl.reshape(x + b, (G, GS))
    xbsort = tl.sort(xb, dim=1, descending=True)
    array = tl.arange(0, GS)
    xbsum = tl.sum(tl.where(array < k, xbsort, 0.0), 1)
    xbsumsort = tl.sort(xbsum, dim=0, descending=True)

    arr = tl.arange(0, G)
    group_min_value = tl.min(tl.where(arr < GK, xbsumsort, 2e38))

    xb_group_mask = tl.where(xbsum[:, None] >= group_min_value, xb, -1e38)
    xb_group_mask = tl.reshape(xb_group_mask, (N,))
    x_group_mask_sort = tl.sort(xb_group_mask, dim=0, descending=True)
    expert_array = tl.arange(0, N)
    min_value = tl.min(tl.where(expert_array < K, x_group_mask_sort, 1e38))
    score = tl.where(xb_group_mask >= min_value, x, 0)

    score = score / (tl.sum(score) + eps) * scale
    map_idx = tl.where(xb_group_mask >= min_value, 1, 0)

    if tl.sum(map_idx) > K:
        y = x.to(tl.float64) + b.to(tl.float64) - tl.arange(0, N).to(
            tl.float64) * 1e-12
        yb = tl.reshape(y, (G, GS))
        ybsort = tl.sort(yb, dim=1, descending=True)
        ysortmask = tl.where(array < k, ybsort, 0)

        ybsum = tl.sum(ysortmask, 1)
        ybsumsort = tl.sort(ybsum, dim=0, descending=True)

        yb_group_min_value = tl.min(tl.where(arr < GK, ybsumsort, 2e38))

        y_group_mask = tl.where(ybsum[:, None] >= yb_group_min_value, yb, -1e38)
        y_group_mask = tl.reshape(y_group_mask, (N,))
        y_group_mask_sort = tl.sort(y_group_mask, dim=0, descending=True)
        y_min_value = tl.min(
            tl.where(expert_array < K, y_group_mask_sort, 1e38))
        double_score = tl.where(y_group_mask >= y_min_value, y, 0)

        double_score = double_score / (tl.sum(double_score) + eps) * scale

        tl.store(prob_ptr + pid * N + tl.arange(0, N), double_score)
        tl.store(map_ptr + pid * N + tl.arange(0, N),
                 tl.where(y_group_mask >= y_min_value, 1, 0))
    else:
        tl.store(prob_ptr + pid * N + tl.arange(0, N), score)
        tl.store(map_ptr + pid * N + tl.arange(0, N), map_idx)


def triton_group_topk_score_forward(x, k,
                                    expert_bias=None,
                                    num_groups=32,
                                    group_topk=4,
                                    scaling_factor=1.0,
                                    score_function='sigmoid',
                                    eps=1e-20):
    """
    calculate topk.
    Args:
        x: input tensor.
        expert_bias: expert bias
        k: topk
    Returns:
        probs:  
        routing_map: 
        tokens_per_expert: 
    """
    device = x.device
    shape = x.shape
    assert len(shape) <= 3 and x.is_contiguous() and score_function == 'sigmoid'
    if len(shape) == 3:
        M, B, N = shape
        g = M * B
        probs = torch.empty((M, B, N), device=device, dtype=x.dtype)
        routing_map = torch.empty((M, B, N), device=device, dtype=torch.bool)
    else:
        M, N = shape
        g = M
        probs = torch.empty((M, N), device=device, dtype=x.dtype)
        routing_map = torch.empty((M, N), device=device, dtype=torch.bool)
    BIAS = expert_bias is not None
    grid = (g,)
    group_topk_score_forward_kernel[grid](
        x,
        expert_bias,
        probs,
        routing_map,
        scaling_factor,
        eps,
        N,
        k,
        num_groups,
        group_topk,
        BIAS,
        num_stages=1,
        num_warps=1
    )
    return probs, routing_map, routing_map.sum(0)


@triton.jit
def group_topk_score_backward_kernel(grad_ptr, input_ptr, map_ptr, dx_ptr,
                                     scale,
                                     eps,
                                     N: tl.constexpr):
    pid = tl.program_id(axis=0)
    grad = tl.load(grad_ptr + pid * N + tl.arange(0, N))
    logit = tl.load(input_ptr + pid * N + tl.arange(0, N))
    mask = tl.load(map_ptr + pid * N + tl.arange(0, N)).to(tl.float32)

    s = tl.sigmoid(logit)
    z = tl.sum(s * mask) + eps
    dx = scale * mask * s * (1 - s) / z * (grad - tl.sum(s * grad * mask) / z)
    tl.store(dx_ptr + pid * N + tl.arange(0, N), dx)


def triton_group_topk_score_backward(grad_output, input, routing_map,
                                     scaling_factor=1.0, eps=1e-20):
    """
    topk backward.
    Args:
        grad_output: grad tensor of prob.
        routing_map: topk indices
    Returns:
        dx: grad of logits
    """
    device = grad_output.device
    shape = grad_output.shape
    assert len(
        shape) <= 3 and grad_output.is_contiguous() and routing_map.is_contiguous()
    if len(shape) == 3:
        M, B, N = shape
        g = M * B
        dx = torch.empty((M, B, N), device=device, dtype=grad_output.dtype)
    else:
        M, N = shape
        g = M
        dx = torch.empty((M, N), device=device, dtype=grad_output.dtype)
    grid = (g,)
    group_topk_score_backward_kernel[grid](
        grad_output,
        input,
        routing_map,
        dx,
        scaling_factor,
        eps,
        N,
        num_stages=2,
        num_warps=1
    )
    return dx
