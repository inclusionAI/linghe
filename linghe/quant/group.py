# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def group_quant_kernel(
        x_ptr,
        y_ptr,
        s_ptr,
        N,
        BLOCK_SIZE: tl.constexpr,
        K: tl.constexpr,
        ROUND: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offs = pid * N + tl.arange(0, K * BLOCK_SIZE)
    n = tl.cdiv(N, K * BLOCK_SIZE)
    soffs = pid * (N // BLOCK_SIZE) + tl.arange(0, K)
    for i in tl.range(n, flatten=True):
        x = tl.load(x_ptr + offs).to(tl.float32)
        x = tl.reshape(x, (K, BLOCK_SIZE), can_reorder=False)
        s = tl.maximum(tl.max(tl.abs(x), 1) / 448.0, 1e-30)
        if ROUND:
            s = tl.exp2(tl.ceil(tl.log2(s)))
        y = x / s[:, None]
        y = y.to(y_ptr.dtype.element_ty)
        y = tl.reshape(y, (K * BLOCK_SIZE,), can_reorder=False)
        tl.store(y_ptr + offs, y)
        tl.store(s_ptr + soffs, s)
        offs += K * BLOCK_SIZE
        soffs += K


def triton_group_quant(x, dtype=torch.float8_e4m3fn, group_size=128,
                       round_scale=False):
    """
    groupwise quantize x, group is in under rowwise format
    Args:
        x: input tensor
        group_size: group wise
        round_scale: whether round scale to power of 2

    Returns:
        - y: quantized tensor, float8_e4m3fn
        - s: quantization scale, float32
    """
    M, N = x.shape
    K = 16 if N > 1024 else 8
    assert N % group_size == 0
    assert x.is_contiguous()

    y = torch.empty((M, N), device=x.device, dtype=dtype)
    s = torch.empty(M, N // group_size, device=x.device, dtype=torch.float32)
    grid = (M,)  # noqa
    group_quant_kernel[grid](
        x, y, s, N, group_size, K, round_scale, num_stages=5, num_warps=4
    )
    return y, s
