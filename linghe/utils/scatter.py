# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def aligned_scatter_add_kernel(
    x_ptr,
    o_ptr,
    indices_ptr,
    weights_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, N)

    sums = tl.zeros((N,), dtype=tl.float32)
    for i in range(K):
        idx = tl.load(indices_ptr + pid * K + i)
        x = tl.load(x_ptr + idx * N + offs)
        if SCALE == 1:
            weight = tl.load(weights_ptr + idx)
            sums += x * weight
        else:
            sums += x

    tl.store(o_ptr + pid * N + offs, sums)


def triton_aligned_scatter_add(
    x: torch.Tensor,
    outputs: torch.Tensor,
    indices: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
):
    """
    scatter_add for megatron 0.11
    Args:
        x: input tensor
        outputs:  output tensor
        indices:  gather indices
        weights:  rowwise weight, it is router prob in MoE router

    Returns:
        output tensor
    """
    assert x.is_contiguous() and outputs.is_contiguous() and indices.is_contiguous()
    M, N = x.shape
    m = outputs.size(0)

    indices = torch.argsort(indices)
    K = M // m
    assert K * m == M and triton.next_power_of_2(N) == N
    SCALE = 1 if weights is not None else 0

    num_stages = 5
    num_warps = 8

    grid = (m,)
    aligned_scatter_add_kernel[grid](
        x,
        outputs,
        indices,
        weights,
        M,
        N,
        K,
        SCALE,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return outputs


# for deepep scatter_add
# atomic_add supports fp16 and fp32, but not bf16


@triton.jit
def scatter_add_kernel(x_ptr, o_ptr, indices_ptr, M, T, N: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, N)

    for i in range(T):
        src_idx = pid * T + i
        dst_idx = tl.load(indices_ptr + src_idx, mask=src_idx < M)
        x = tl.load(x_ptr + src_idx * N + offs, mask=src_idx < M).to(tl.float32)
        tl.atomic_add(o_ptr + dst_idx * N + offs, x, sem="relaxed")


@triton.jit
def fp32_to_bf16_kernel(x_ptr, o_ptr, M, T, N: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = tl.arange(0, N)

    for i in range(T):
        idx = pid * T + i
        x = tl.load(x_ptr + idx * N + offs, mask=idx < M)
        tl.store(o_ptr + idx * N + offs, x, mask=idx < M)


def triton_scatter_add(x, outputs, indices):
    """
    naive version of scatter add, very slow
    Args:
        x: input tensor
        outputs: output tensor
        indices: indices

    Returns:
        output tensor
    """
    assert x.is_contiguous() and outputs.is_contiguous() and indices.is_contiguous()
    M, N = x.shape
    assert triton.next_power_of_2(N) == N

    float_outputs = torch.zeros(
        outputs.shape, dtype=torch.float32, device=outputs.device
    )

    sm = 512
    T = triton.cdiv(M, sm)

    num_stages = 5
    num_warps = 8

    grid = (sm,)
    scatter_add_kernel[grid](
        x, float_outputs, indices, M, T, N, num_stages=num_stages, num_warps=num_warps
    )

    m = outputs.shape[0]
    T = triton.cdiv(m, sm)
    grid = (sm,)
    fp32_to_bf16_kernel[grid](
        float_outputs, outputs, m, T, N, num_stages=num_stages, num_warps=num_warps
    )

    return outputs


@triton.jit
def unpermute_with_mask_map_kernel(
    grads_ptr,
    probs_ptr,
    mask_map_ptr,
    output_ptr,
    output_probs_ptr,
    n,
    N: tl.constexpr,
    num_experts: tl.constexpr,
    PROB: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    n = n.to(tl.int64)
    # sums = tl.zeros((N,), dtype=tl.float32)
    sums = tl.zeros((N,), dtype=tl.float32)

    indices = tl.load(mask_map_ptr + pid * num_experts + tl.arange(0, num_experts))
    count = tl.sum(tl.where(indices >= 0, 1, 0))
    mask_indices = tl.where(indices < 0, 2**24, indices)
    idx = tl.argmin(mask_indices, 0)
    index = tl.min(mask_indices)

    for i in range(count):
        load_mask = (index >= 0) & (tl.arange(0, N) < n)
        sums += tl.load(grads_ptr + index * n + tl.arange(0, N), mask=load_mask).to(
            tl.float32
        )

        if PROB:
            mask = index >= 0
            prob = tl.load(probs_ptr + index, mask=mask)
            tl.store(output_probs_ptr + pid * num_experts + idx, prob, mask=mask)

        mask_indices = tl.where(indices <= index, 2**24, indices)
        idx = tl.argmin(mask_indices, 0)
        index = tl.min(mask_indices)

    tl.store(output_ptr + pid * n + tl.arange(0, N), sums, mask=tl.arange(0, N) < n)


def triton_unpermute_with_mask_map(
    grad: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
):
    """
    scatter add with row id map
    Args:
        grad: gradient tensor, [num_out_tokens, hidden_size]
        row_id_map: row id map, [n_experts, num_tokens]
        probs: [num_out_tokens]

    Returns:
        - output: [num_tokens, hidden_size]
        - restore_probs: [num_tokens, num_experts]
    """
    assert grad.is_contiguous() and row_id_map.is_contiguous()
    n = grad.shape[1]
    N = triton.next_power_of_2(n)
    num_tokens, num_experts = row_id_map.shape  # not transposed

    output = torch.empty((num_tokens, n), dtype=grad.dtype, device="cuda")

    PROB = probs is not None
    if PROB:
        assert probs.is_contiguous()
        restore_probs = torch.zeros(
            (num_tokens, num_experts), dtype=probs.dtype, device="cuda"
        )
    else:
        restore_probs = None

    if num_tokens == 0:
        return output, restore_probs

    grid = (num_tokens,)
    unpermute_with_mask_map_kernel[grid](
        grad,
        probs,
        row_id_map,
        output,
        restore_probs,
        n,
        N,
        num_experts,
        PROB,
        num_stages=4,
        num_warps=4,
    )
    return output, restore_probs
