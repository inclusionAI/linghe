# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def silu_and_block_quant_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        scale_ptr,
        routing_scale,
        M,
        n: tl.constexpr,
        K: tl.constexpr,
        ROUND: tl.constexpr, ):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)

    offs = (rid * n * 2
            + cid * K * 128
            + tl.arange(0, K)[:, None] * 128
            + tl.arange(0, 128)[None, :])

    x1 = tl.load(x_ptr + offs).to(tl.float32)
    x2 = tl.load(x_ptr + n + offs).to(tl.float32)
    x = x1 * tl.sigmoid(x1) * x2
    if weight_ptr is not None: 
        weight = tl.load(weight_ptr + rid).to(tl.float32) * routing_scale
        x = x * weight

    scale = tl.maximum(tl.max(x.abs(), 1) / 448, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    tl.store(scale_ptr + rid * n // 128 + cid * K + tl.arange(0, K), scale)

    xq = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr
        + rid * n
        + cid * K * 128
        + tl.arange(0, K)[:, None] * 128
        + tl.arange(0, 128)[None, :],
        xq,
        )


def triton_silu_and_block_quant(
        x, weight=None, out=None, scale=None, routing_scale=1.0, round_scale=False):
    """
    fused silu and blockwise quantization in mlp/moe kernel
    scale is not transposed, which is different from deepgemm kernel
    Args:
        x: input tensor
        weight: router weight
        round_scale: whether round scale to power of 2
    Returns:
        - out: quantized tensor
        - scale: quantization scale
    """
    M, N = x.shape
    n = N // 2
    assert n % 128 == 0
    device = x.device
    if out is None:
        out = torch.empty((M, n), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.empty((M, n // 128), device=device,
                            dtype=torch.float32)
    
    B = n // 128
    if M >= 256:
        if B % 4 == 0:
            B = B // 4
        elif B % 2 == 0:
            B = B // 2
    elif M >= 64:
        if B % 2 == 0:
            B = B // 2

    K = n // B // 128
    grid = (M, B)
    silu_and_block_quant_kernel[grid](
            x,
            weight,
            out,
            scale,
            routing_scale,
            M,
            n,
            K,
            round_scale,
            num_stages=2,
            num_warps=4)

    return out, scale
