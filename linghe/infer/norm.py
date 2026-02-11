# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_and_block_quant_kernel(
        x_ptr,
        weight_ptr,
        residual_ptr,
        out_ptr,
        scale_ptr,
        eps,
        M,
        T: tl.constexpr,
        N: tl.constexpr,
        nb: tl.constexpr,
        W: tl.constexpr,
        ROUND: tl.constexpr
        ):
    pid = tl.program_id(axis=0)

    # row-wise read, row-wise write
    weight = tl.load(weight_ptr + tl.arange(0, N)).to(tl.float32)[None, :]
    offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, N)[
                                                            None, :]
    PM = (M + 3) // 4 * 4
    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32)

        if residual_ptr is not None:
            r = tl.load(residual_ptr + offs, mask=indices[:, None] < M).to(tl.float32)
            x = x + r
            tl.debug_barrier()
            tl.store(residual_ptr + offs, x, mask=indices[:, None] < M)

        rms = tl.rsqrt(tl.sum(x * x, axis=1) / N + eps)
        x = x * rms[:, None] * weight
        x = tl.reshape(x, [W, nb, 128])

        scale = tl.max(tl.abs(x), 2) / 448.0
        scale = tl.where(scale == 0.0, 1.0, scale)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(scale_ptr + tl.arange(0, nb)[:, None] * PM + indices[None, :], tl.trans(scale),
                 mask=indices[None, :] < M
                 )

        x = x / scale[:, :, None]
        x = tl.reshape(x, [W, N])

        tl.store(out_ptr + offs, x.to(out_ptr.dtype.element_ty), mask=indices[:, None] < M)

        offs += N * W


def triton_rms_norm_and_block_quant(
        x: torch.Tensor,
        weight: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        out: Optional[torch.Tensor] = None,
        scale: Optional[torch.Tensor] = None,
        rms: Optional[torch.Tensor] = None,
        round_scale: bool = False,
        ):
    """
    Fused RMSNorm forward and block quantization.
    Args:
        x: Input tensor, shape [M, N]
        weight: RMSNorm weight,  shape [N]
        residual: Residual tensor, shape [M, N]
        eps: epsilon value for L2 normalization.
        out: output of quantization data
        scale: output of quantization scale.
        rms: output of rms
        round_scale: Set whether to force power of 2 scales.
    Returns:
        - out: quantization data.
        - scale: quantization scale.
        - residual: residual tensor.
    """
    assert x.is_contiguous() and weight.is_contiguous() and (residual is None or residual.is_contiguous())
    M, N = x.shape
    assert N <= 8192 and 8192 % N == 0
    device = x.device

    if out is None:
        out = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)

    if scale is None:
        scale = torch.zeros((N // 128, (M + 3) // 4 * 4), device=device, dtype=torch.float32)

    W = 8192 // N
    T = 4 // W
    grid = (triton.cdiv(M, 4),)

    rms_norm_and_block_quant_kernel[grid](
        x,
        weight,
        residual,
        out,
        scale,
        eps,
        M,
        T,
        N,
        N // 128,
        W,
        round_scale,
        num_stages=3,
        num_warps=8
        )

    return out, scale[:, :M].t(), residual


@triton.jit
def residual_rms_norm_and_block_quant_kernel(
        x_ptr,
        weight_ptr,
        residual_ptr,
        out_ptr,
        scale_ptr,
        eps,
        M,
        PM,
        N: tl.constexpr,
        nb: tl.constexpr,
        W: tl.constexpr,
        ROUND: tl.constexpr
        ):
    pid = tl.program_id(axis=0)

    # row-wise read, row-wise write
    weight = tl.load(weight_ptr + tl.arange(0, N)).to(tl.float32)[None, :]
    offs = pid * W * N + tl.arange(0, W)[:, None] * N + tl.arange(0, N)[
                                                        None, :]
    # PM = (M + 3) // 4 * 4
    indices = pid * W + tl.arange(0, W)
    x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32)

    r = tl.load(residual_ptr + offs, mask=indices[:, None] < M).to(tl.float32)
    x = x + r
    tl.debug_barrier()
    tl.store(residual_ptr + offs, x, mask=indices[:, None] < M)

    rms = tl.rsqrt(tl.sum(x * x, axis=1) / N + eps)
    x = x * rms[:, None] * weight
    tl.store(x_ptr + offs, x, mask=indices[:, None] < M)

    x = tl.reshape(x, [W, nb, 128], can_reorder=False)

    scale = tl.max(tl.abs(x), 2) / 448.0
    scale = tl.where(scale == 0.0, 1.0, scale)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(scale_ptr + tl.arange(0, nb)[:, None] * PM + indices[None, :], tl.trans(scale), mask=indices[None, :] < M)

    x = x / scale[:, :, None]
    x = tl.reshape(x, [W, N], can_reorder=False)

    tl.store(out_ptr + offs, x, mask=indices[:, None] < M)


def triton_residual_rms_norm_and_block_quant(
        x: torch.Tensor,
        weight: torch.Tensor,
        residual: torch.Tensor,
        eps: float = 1e-6,
        round_scale: bool = False,
        ):
    """
    Fused RMSNorm forward and block quantization.
    Args:
        x: Input tensor, shape [M, N]
        weight: RMSNorm weight,  shape [N]
        residual: Residual tensor, shape [M, N]
        eps: epsilon value for L2 normalization.
        out: output of quantization data
        scale: output of quantization scale.
        rms: output of rms
        round_scale: Set whether to force power of 2 scales.
    Returns:
        - x: rmsnorm data.
        - out: quantization data.
        - scale: quantization scale.
        - residual: residual tensor.
    """
    assert x.is_contiguous() and weight.is_contiguous() and residual.is_contiguous()
    M, N = x.shape
    assert N <= 8192 and 8192 % N == 0 and N >= 2048
    device = x.device

    out = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)

    PM = (M + 3) // 4 * 4
    scale = torch.zeros((N // 128, PM), device=device, dtype=torch.float32)

    W = 8192 // N
    grid = (triton.cdiv(M, W),)

    residual_rms_norm_and_block_quant_kernel[grid](
        x,
        weight,
        residual,
        out,
        scale,
        eps,
        M,
        PM,
        N,
        N // 128,
        W,
        round_scale,
        num_stages=3,
        num_warps=2
        )

    return x, out, scale[:, :M].t(), residual
    # return x, None, residual
