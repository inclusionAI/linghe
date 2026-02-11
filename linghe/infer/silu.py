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
        out_ptr,
        scale_ptr,
        M,
        n: tl.constexpr,
        ROUND: tl.constexpr, ):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)

    offs = (rid * 128 * n * 2
            + cid * 128
            + tl.arange(0, 128)[:, None] * n * 2
            + tl.arange(0, 128)[None, :])
    indices = rid * 128 + tl.arange(0, 128)
    mask = indices[:, None] < M

    x1 = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    x2 = tl.load(x_ptr + n + offs, mask=mask).to(tl.float32)
    x = x1 * tl.sigmoid(x1) * x2
    # x1 = tl.load(x_ptr + offs, mask=mask)
    # x2 = tl.load(x_ptr + n + offs, mask=mask)
    # x = tl.sigmoid(x1.to(tl.float32)) * x1 * x2

    scale = tl.maximum(tl.max(x.abs(), 1) / 448, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    PM = (M + 3) // 4 * 4
    scale_offs = cid * PM + indices
    tl.store(scale_ptr + scale_offs, scale, mask=indices < M)

    xq = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr
        + rid * 128 * n
        + cid * 128
        + tl.arange(0, 128)[:, None] * n
        + tl.arange(0, 128)[None, :],
        xq,
        mask=mask, )


def triton_silu_and_block_quant(
        x, out=None, scale=None, round_scale=False):
    """
    fused silu and blockwise quantization, used in shared expert
    Args:
        x: input tensor
        round_scale: whether round scale to power of 2
    Returns:
        - out: quantized tensor
        - scale: quantization scale
    """
    M, N = x.shape
    n = N // 2
    device = x.device
    if out is None:
        out = torch.empty((M, n), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.zeros((n // 128, (M + 3) // 4 * 4), device=device,
                            dtype=torch.float32)

    grid = (triton.cdiv(M, 128), n // 128)
    silu_and_block_quant_kernel[grid](
        x,
        out,
        scale,
        M,
        n,
        round_scale,
        num_stages=2,
        num_warps=8, )

    return out, scale[:, :M].t()
