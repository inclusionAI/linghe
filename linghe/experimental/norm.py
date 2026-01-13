# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

"""
the code is used to reproduce the barrier bug in cross-block reduce.
"""


@triton.jit
def rms_norm_forward_kernel(x_ptr,
                            weight_ptr,
                            out_ptr,
                            cache_ptr,
                            signal_ptr,
                            rms_ptr,
                            eps,
                            M,
                            n: tl.constexpr,
                            H: tl.constexpr,
                            B: tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    CB = tl.num_programs(1)

    indices = rid * H + tl.arange(0, H)
    offs = rid * H * n + cid * B + tl.arange(0, H)[:, None] * n + tl.arange(0,
                                                                            B)[
                                                                  None, :]

    x = tl.load(x_ptr + offs).to(tl.float32)
    weight = tl.load(weight_ptr + cid * B + tl.arange(0, B)).to(tl.float32)

    s = tl.sum(x * x, axis=1)
    tl.atomic_add(cache_ptr + indices, s, sem='acq_rel', scope='sys')
    # tl.debug_barrier()
    tl.atomic_add(signal_ptr + rid, 1, sem='acq_rel', scope='sys')
    tl.debug_barrier()
    # tl.inline_asm_elementwise(
    #         "membar.gl;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    #     )
    # tl.inline_asm_elementwise(
    #         "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    #     )
    count = tl.load(signal_ptr + rid, cache_modifier='.cv')
    while count < CB:
        count = tl.load(signal_ptr + rid, cache_modifier='.cv')
    # if cid + rid == 0:
    #     tl.device_print('count', count)
    tl.debug_barrier()

    sums = tl.load(cache_ptr + indices, cache_modifier='.cv', volatile=True)
    # sums = tl.atomic_add(cache_ptr + indices, 0.0, sem='acq_rel', scope='gpu')

    rms = tl.rsqrt(sums / n + eps)
    if cid == 0:
        tl.store(rms_ptr + indices, rms)

    x = x * rms[:, None] * weight[None, :]

    tl.store(out_ptr + offs, x)


def triton_rms_norm_forward(x: torch.Tensor,
                            weight: torch.Tensor,
                            eps: float = 1e-6):
    """
    Fused RMSNorm forward.
    Args:
        x: Input tensor, shape [M, N]
        weight: RMSNorm weight,  shape [N]
        eps: epsilon value for L2 normalization.
    Returns:
        - out: output.
        - rms: Reciprocal of the root mean square of the
            input calculated over the last dimension.
    """
    assert x.is_contiguous() and weight.is_contiguous()
    M, n = x.shape
    device = x.device

    out = torch.empty((M, n), dtype=x.dtype, device=device)
    rms = torch.empty((M,), dtype=torch.float32, device=device)
    H = 8
    B = 128
    assert M % H == 0 and n % B == 0
    CB = n // 128  # column block
    RB = M // H  # row block
    cache = torch.zeros((M,), dtype=torch.float32, device=device)
    signals = torch.zeros((RB,), dtype=torch.int32, device=device)
    grid = (RB, CB)
    rms_norm_forward_kernel[grid](
        x,
        weight,
        out,
        cache,
        signals,
        rms,
        eps,
        M,
        n,
        H,
        B,
        num_stages=1,
        num_warps=1
    )
    return out, rms


@triton.jit
def _parallel_rms_norm_and_block_quant_forward_kernel(x_ptr,
                                                      weight_ptr,
                                                      out_ptr,
                                                      scale_ptr,
                                                      transpose_output_ptr,
                                                      transpose_scale_ptr,
                                                      cache_ptr,
                                                      rms_ptr,
                                                      eps,
                                                      M,
                                                      n: tl.constexpr,
                                                      H: tl.constexpr,
                                                      B: tl.constexpr,
                                                      K: tl.constexpr,
                                                      ROUND: tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    CB = tl.num_programs(1)

    indices = rid * H + tl.arange(0, H)
    masks = indices[:, None] < M
    offs = rid * H * n + cid * B + tl.arange(0, H)[:, None] * n + tl.arange(0,
                                                                            B)[
                                                                  None, :]

    x = tl.load(x_ptr + offs, mask=masks).to(tl.float32)
    s = tl.sum(x * x, axis=1)
    # tl.debug_barrier()
    tl.atomic_add(cache_ptr + indices, s, scope='sys')
    # tl.debug_barrier()
    tl.atomic_add(cache_ptr + M + rid, 1.0, scope='sys')
    # tl.inline_asm_elementwise(
    #         "bar.sync 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    #     )
    tl.debug_barrier()
    # count = tl.atomic_add(cache_ptr + M + rid, 0.0)
    # tl.inline_asm_elementwise(
    #         "membar.gl;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    #     )
    for i in range(3):
        count = tl.load(cache_ptr + M + rid)
        # tl.debug_barrier()
        while count < CB:
            count = tl.load(cache_ptr + M + rid)
        tl.debug_barrier()
    # tl.inline_asm_elementwise(
    #         "membar.gl;", "=r", [], dtype=tl.int32, is_pure=False, pack=1
    #     )
    sums = tl.load(cache_ptr + indices, mask=indices < M)

    rms = tl.rsqrt(sums / n + eps)
    if cid == CB - 1:
        tl.store(rms_ptr + indices, rms, mask=indices < M)

    toffs = cid * M * B + rid * H + tl.arange(0, B)[:, None] * M + tl.arange(0,
                                                                             H)[
                                                                   None, :]
    weight = tl.load(weight_ptr + cid * B + tl.arange(0, B)).to(tl.float32)
    x = x * rms[:, None] * weight[None, :]
    scale = tl.maximum(tl.max(tl.abs(x), 1) / 448.0, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    q = (x / scale[:, None]).to(out_ptr.dtype.element_ty)

    tl.store(scale_ptr + cid * M + indices, scale, mask=indices < M)
    tl.store(out_ptr + offs, q, mask=masks)

    scale = tl.maximum(tl.max(x.abs(), 0) / 448.0, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(transpose_scale_ptr + rid * n + cid * B + tl.arange(0, B), scale)

    q = (tl.trans(x / scale)).to(transpose_output_ptr.dtype.element_ty)
    tl.store(transpose_output_ptr + toffs, q, mask=indices[None, :] < M)


@triton.jit
def parallel_rms_norm_and_block_quant_forward_kernel(x_ptr,
                                                     weight_ptr,
                                                     out_ptr,
                                                     scale_ptr,
                                                     transpose_output_ptr,
                                                     transpose_scale_ptr,
                                                     cache_ptr,
                                                     rms_ptr,
                                                     eps,
                                                     M,
                                                     n: tl.constexpr,
                                                     H: tl.constexpr,
                                                     B: tl.constexpr,
                                                     K: tl.constexpr,
                                                     ROUND: tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    CB = tl.num_programs(1)

    indices = rid * H + tl.arange(0, H)
    masks = indices[:, None] < M
    offs = rid * H * n + cid * K * B + tl.arange(0, H)[:, None] * n + tl.arange(
        0, B)[
                                                                      None, :]

    s = tl.zeros((H,), dtype=tl.float32)
    for i in range(K):
        x = tl.load(x_ptr + i * B + offs, mask=masks).to(tl.float32)
        s += tl.sum(x * x, axis=1)

    tl.atomic_add(cache_ptr + indices, s)
    tl.atomic_add(cache_ptr + M + rid, 1.0)
    tl.debug_barrier()

    count = tl.atomic_add(cache_ptr + M + rid, 0.0)
    # count = tl.load(cache_ptr + M + rid)
    tl.debug_barrier()
    while count < 1.0 * CB:
        count = tl.load(cache_ptr + M + rid)
        for i in range(1024):
            pass
    tl.debug_barrier()
    sums = tl.load(cache_ptr + indices, mask=indices < M)

    rms = tl.rsqrt(sums / n + eps)

    if cid == CB - 1:
        tl.store(rms_ptr + indices, rms, mask=indices < M)

    toffs = cid * M * B * K + rid * H + tl.arange(0, B)[:,
                                        None] * M + tl.arange(0, H)[
                                                    None, :]
    for i in range(K):
        weight = tl.load(weight_ptr + cid * K * B + i * B + tl.arange(0, B)).to(
            tl.float32)
        x = tl.load(x_ptr + i * B + offs, mask=masks).to(tl.float32)
        x = x * rms[:, None] * weight[None, :]
        scale = tl.maximum(tl.max(tl.abs(x), 1) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        q = (x / scale[:, None]).to(out_ptr.dtype.element_ty)

        tl.store(scale_ptr + cid * K * M + i * M + indices, scale,
                 mask=indices < M)
        tl.store(out_ptr + i * B + offs, q, mask=masks)

        scale = tl.maximum(tl.max(x.abs(), 0) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(
            transpose_scale_ptr + rid * n + cid * B * K + i * B + tl.arange(0,
                                                                            B),
            scale)

        q = (tl.trans(x / scale)).to(transpose_output_ptr.dtype.element_ty)
        tl.store(transpose_output_ptr + i * B * M + toffs, q,
                 mask=indices[None, :] < M)


def triton_parallel_rms_norm_and_block_quant_forward(x: torch.Tensor,
                                                     weight: torch.Tensor,
                                                     eps: float = 1e-6,
                                                     out: Optional[
                                                         torch.Tensor] = None,
                                                     scale: Optional[
                                                         torch.Tensor] = None,
                                                     rms: Optional[
                                                         torch.Tensor] = None,
                                                     round_scale: bool = False,
                                                     output_mode: int = 2):
    """
    Fused RMSNorm forward and block quantization.
    Args:
        x: Input tensor, shape [M, N]
        weight: RMSNorm weight,  shape [N]
        eps: epsilon value for L2 normalization.
        out: output of quantization data
        scale: output of quantization scale.
        rms: output of rms
        round_scale: Set whether to force power of 2 scales.
        output_mode: one of {0, 1, 2}.
            0: only output non-transpose tensor
            1: only output transposed tensor
            2: return both
    Returns:
        - out: quantization data.
        - scale: quantization scale.
        - rms: Reciprocal of the root mean square of the
            input calculated over the last dimension.
        - transpose_output: quantization data of transposed gradient.
        - transpose_scale: quantization scale of transposed gradient.
    """
    # row-wise read, row-wise write
    assert x.is_contiguous() and weight.is_contiguous()
    M, n = x.shape
    device = x.device

    if out is None and output_mode in (0, 2):
        out = torch.empty((M, n), device=device, dtype=torch.float8_e4m3fn)

    if scale is None and output_mode in (0, 2):
        scale = torch.empty((n // 128, M), device=device, dtype=torch.float32)

    transpose_output = torch.empty((n, M), device=device,
                                   dtype=torch.float8_e4m3fn)
    transpose_scale = torch.empty(((M + 127) // 128, n), device=device,
                                  dtype=torch.float32)

    assert rms is None
    rms = torch.empty((M,), dtype=torch.float32, device=device)
    H = 128
    B = 128
    CB = n // B
    K = n // (B * CB)
    assert K >= 1
    RB = triton.cdiv(M, H)
    cache = torch.zeros((M + RB,), dtype=torch.float32, device=device)
    grid = (RB, CB)
    _parallel_rms_norm_and_block_quant_forward_kernel[grid](
        x,
        weight,
        out,
        scale,
        transpose_output,
        transpose_scale,
        cache,
        rms,
        eps,
        M,
        n,
        H,
        B,
        K,
        round_scale,
        num_stages=5,
        num_warps=4
    )
    return out, scale, rms, transpose_output, transpose_scale
