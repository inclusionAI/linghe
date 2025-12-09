# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def mxfp8_quant_kernel(
    x_ptr,
    x_q_ptr,
    x_s_ptr,
    xt_q_ptr,
    xt_s_ptr,
    m,
    N: tl.constexpr,
    B: tl.constexpr,
    OUTPUT_MODE: tl.constexpr,
):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)

    offs = (
        rid * 32 * N
        + cid * B
        + tl.arange(0, 32)[:, None] * N
        + tl.arange(0, B)[None, :]
    )
    indices = rid * 32 + tl.arange(0, 32)
    mask = indices[:, None] < m
    b = N // 32
    sb: tl.constexpr = B // 32

    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)

    if OUTPUT_MODE % 2 == 0:
        xr = tl.reshape(x, [32, sb, 32])
        scale = tl.maximum(tl.max(xr.abs(), 2) / 448, 1e-30)
        log_scale = tl.ceil(tl.log2(scale))
        scale = tl.exp2(log_scale)
        tl.store(
            x_s_ptr
            + rid * 32 * b
            + cid * B // 32
            + tl.arange(0, 32)[:, None] * b
            + tl.arange(0, sb),
            log_scale + 127,
        )
        xq = tl.reshape(xr / scale[:, :, None], (32, B)).to(x_q_ptr.dtype.element_ty)
        tl.store(
            x_q_ptr
            + rid * 32 * N
            + cid * B
            + tl.arange(0, 32)[:, None] * N
            + tl.arange(0, B)[None, :],
            xq,
            mask=mask,
        )

    if OUTPUT_MODE > 0:
        scale = tl.maximum(tl.max(x.abs(), 0) / 448, 1e-30)
        log_scale = tl.ceil(tl.log2(scale))
        scale = tl.exp2(log_scale)
        tl.store(xt_s_ptr + rid * N + cid * B + tl.arange(0, B), log_scale + 127)
        xq = (x / scale).to(xt_q_ptr.dtype.element_ty)
        tl.store(
            xt_q_ptr
            + rid * 32 * N
            + cid * B
            + tl.arange(0, 32)[:, None] * N
            + tl.arange(0, B)[None, :],
            xq,
            mask=mask,
        )


def triton_mxfp8_quant(x, output_mode=2):
    """
    fused silu and mxfp8 quantization, used in shared expert
    Args:
        x: input tensor
        output_mode: one of {0, 1, 2}
            0: only output non-transposed quantized tensor
            1: only output transposed quantized tensor
            2: output both

    Returns:
        - x_q: quantized tensor
        - x_scale: quantization scale
        - xt_q: quantized tensor of transposed output
        - xt_scale: quantization scale of transposed output
    """
    m, N = x.shape
    M = (m + 127) // 128 * 128
    assert N % 128 == 0  # transposed scaled should be multiplier of 128
    assert x.is_contiguous()
    device = x.device
    x_q = torch.empty((m, N), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((M, N // 32), device=device, dtype=torch.uint8)

    xt_q = torch.empty((m, N), device=device, dtype=torch.float8_e4m3fn)
    xt_scale = torch.empty((M // 32, N), device=device, dtype=torch.uint8)
    B = 128
    grid = (M // 32, N // B)
    mxfp8_quant_kernel[grid](
        x, x_q, x_scale, xt_q, xt_scale, m, N, B, output_mode, num_stages=3, num_warps=2
    )

    return x_q, x_scale, xt_q, xt_scale


@triton.jit
def batch_mxfp8_quant_kernel(
    x_ptr,
    count_ptr,
    xq_ptr,
    xs_ptr,
    xtq_ptr,
    xts_ptr,
    N: tl.constexpr,
    B: tl.constexpr,
    E: tl.constexpr,
    OUTPUT_MODE: tl.constexpr,
):
    eid = tl.program_id(axis=0)
    rid = tl.program_id(axis=1)
    cid = tl.program_id(axis=2)

    count = tl.load(count_ptr + eid)
    counts = tl.load(count_ptr + tl.arange(0, E))

    if rid >= tl.cdiv(count, 128) * 4:
        return

    m_block = tl.sum(tl.where(tl.arange(0, E) < eid, tl.cdiv(counts, 128), 0)) * 4
    si = tl.sum(tl.where(tl.arange(0, E) < eid, counts, 0))

    offs = (
        si * N
        + rid * 32 * N
        + cid * B
        + tl.arange(0, 32)[:, None] * N
        + tl.arange(0, B)[None, :]
    )
    indices = rid * 32 + tl.arange(0, 32)
    mask = indices[:, None] < count
    b = N // 32
    sb: tl.constexpr = B // 32

    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)

    if OUTPUT_MODE % 2 == 0:
        xr = tl.reshape(x, [32, sb, 32])
        scale = tl.maximum(tl.max(xr.abs(), 2) / 448, 1e-30)
        log_scale = tl.ceil(tl.log2(scale))
        scale = tl.exp2(log_scale)
        tl.store(
            xs_ptr
            + m_block * N
            + rid * 32 * b
            + cid * B // 32
            + tl.arange(0, 32)[:, None] * b
            + tl.arange(0, sb),
            log_scale + 127,
        )
        xq = tl.reshape(xr / scale[:, :, None], (32, B)).to(xq_ptr.dtype.element_ty)
        tl.store(xq_ptr + offs, xq, mask=mask)

    if OUTPUT_MODE > 0:
        scale = tl.maximum(tl.max(x.abs(), 0) / 448, 1e-30)
        log_scale = tl.ceil(tl.log2(scale))
        scale = tl.exp2(log_scale)
        tl.store(
            xts_ptr + m_block * N + rid * N + cid * B + tl.arange(0, B), log_scale + 127
        )
        xq = (x / scale).to(xtq_ptr.dtype.element_ty)
        tl.store(xtq_ptr + offs, xq, mask=mask)

def triton_batch_mxfp8_quant(xs, token_count_per_expert, splits, output_mode=2):
    """
    select and quant, used in megatron 0.12 flex moe
    Args:
        xs: [bs, dim]
        token_count_per_expert: [n_experts]
        splits: python int list of token_count_per_expert
        output_mode: one of {0, 1, 2}
            0: only output non-transposed quantized tensor
            1: only output transposed quantized tensor
            2: output both

    Returns:
        - x_q:
        - x_scale:
        - xt_q:
        - xt_scale:

    """
    assert xs.is_contiguous()
    m, N = xs.shape
    assert N % 128 == 0
    n_experts = token_count_per_expert.size(0)
    device = xs.device
    M = sum([(x + 127) // 128 for x in splits]) * 128

    x_q = torch.empty((m, N), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((M, N // 32), device=device, dtype=torch.uint8)
    xt_q = torch.empty((m, N), device=device, dtype=torch.float8_e4m3fn)
    xt_scale = torch.empty((M // 32, N), device=device, dtype=torch.uint8)

    if m == 0:
        return x_q, x_scale, xt_q, xt_scale

    B = 256 if N % 256 == 0 else 128
    grid = (n_experts, triton.cdiv(max(splits), 128) * 4, N // B)
    batch_mxfp8_quant_kernel[grid](
        xs,
        token_count_per_expert,
        x_q,
        x_scale,
        xt_q,
        xt_scale,
        N,
        B,
        n_experts,
        output_mode,
        num_stages=3,
        num_warps=4,
    )

    return x_q, x_scale, xt_q, xt_scale
