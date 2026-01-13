# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def block_quant_kernel(x_ptr, y_ptr, s_ptr, M, N, BLOCK_SIZE: tl.constexpr,
                       ROUND: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x)) / 448.0, 1e-30)
    if ROUND:
        s = tl.exp2(tl.ceil(tl.log2(s)))
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y, mask=mask)
    tl.store(s_ptr + pid_m * n + pid_n, s)


def triton_block_quant(x,
                       block_size=128,
                       round_scale=False):
    """
    blockwise quantize x, used for blockwise recipe for weight in megatron
    Args:
        x: input tensor
        block_size: block wise
        round_scale: whether round scale to power of 2

    Returns:
        - y: quantized tensor, float8_e4m3fn
        - s: quantization scale, float32
    """
    assert x.is_contiguous()
    M, N = x.size()
    y = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(M // block_size, N // block_size,
                    dtype=torch.float32, device=x.device)
    grid = (triton.cdiv(M, block_size), triton.cdiv(N, block_size))
    block_quant_kernel[grid](x,
                             y,
                             s,
                             M,
                             N,
                             BLOCK_SIZE=block_size,
                             ROUND=round_scale,
                             num_stages=6,
                             num_warps=8)
    return y, s


@triton.jit
def blockwise_quant_kernel(x_ptr,
                           x_q_ptr,
                           x_scale_ptr,
                           xt_q_ptr,
                           xt_scale_ptr,
                           M,
                           N: tl.constexpr,
                           ROUND: tl.constexpr,
                           OUTPUT_MODE: tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)

    offs = rid * 128 * N + cid * 128 + tl.arange(0, 128)[:,
                                       None] * N + tl.arange(0, 128)[
                                                   None, :]
    indices = rid * 128 + tl.arange(0, 128)
    mask = indices[:, None] < M

    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)

    if OUTPUT_MODE % 2 == 0:
        scale = tl.maximum(tl.max(x.abs(), 1) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))

        tl.store(x_scale_ptr + rid * 128 + cid * M + tl.arange(0, 128), scale,
                 mask=indices < M)
        xq = (x / scale[:, None]).to(x_q_ptr.dtype.element_ty)
        tl.store(x_q_ptr + rid * 128 * N + cid * 128 + tl.arange(0, 128)[:,
                                                       None] * N + tl.arange(0,
                                                                             128)[
                                                                   None, :], xq,
                 mask=mask)

    if OUTPUT_MODE > 0:
        scale = tl.maximum(tl.max(x.abs(), 0) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(xt_scale_ptr + rid * N + cid * 128 + tl.arange(0, 128),
                 scale)
        xq = (x / scale).to(xt_q_ptr.dtype.element_ty)
        tl.store(xt_q_ptr + rid * 128 + cid * 128 * M + tl.arange(0,
                                                                  128)[
                                                        :,
                                                        None] * M + tl.arange(
            0, 128)[
                                                                    None,
                                                                    :],
                 tl.trans(xq), mask=indices[None, :] < M)


def triton_blockwise_quant(x,
                           round_scale=False,
                           output_mode=2):
    """
    blockwise quantization, used in blockwise recipt in megatron
    Args:
        x: input tensor
        round_scale: whether round scale to power of 2
        output_mode: one of {0, 1, 2}
            0: only output non-transposed quantized tensor
            1: only output transposed quantized tensor
            2: output both

    Returns:
        x_q: 
        x_scale: 
        xt_q: 
        xt_scale: 
    """
    M, N = x.shape
    assert M % 16 == 0 and x.is_contiguous()
    device = x.device
    x_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((N // 128, M), device=device,
                          dtype=torch.float32)

    xt_q = torch.empty((N, M), device=device,
                       dtype=torch.float8_e4m3fn)
    xt_scale = torch.empty((triton.cdiv(M, 128), N), device=device,
                           dtype=torch.float32)

    grid = (triton.cdiv(M, 128), N // 128)
    blockwise_quant_kernel[grid](
        x,
        x_q,
        x_scale,
        xt_q,
        xt_scale,
        M,
        N,
        round_scale,
        output_mode,
        num_stages=2,
        num_warps=2
    )

    return x_q, x_scale, xt_q, xt_scale


@triton.jit
def batch_blockwise_quant_kernel(x_ptr,
                                 count_ptr,
                                 xq_ptr,
                                 xs_ptr,
                                 xtq_ptr,
                                 xts_ptr,
                                 N: tl.constexpr,
                                 E: tl.constexpr,
                                 ROUND: tl.constexpr,
                                 ):
    eid = tl.program_id(axis=0)
    rid = tl.program_id(axis=1)
    cid = tl.program_id(axis=2)

    count = tl.load(count_ptr + eid)
    counts = tl.load(count_ptr + tl.arange(0, E))

    if rid >= tl.cdiv(count, 128):
        return

    nb = N // 128

    m_block = tl.sum(tl.where(tl.arange(0, E) < eid, tl.cdiv(counts, 128), 0))
    si = tl.sum(tl.where(tl.arange(0, E) < eid, counts, 0))

    rids = rid * 128 + tl.arange(0, 128)

    x = tl.load(
        x_ptr + si * N + rid * 128 * N + cid * 128 + tl.arange(0, 128)[:,
                                                     None] * N + tl.arange(
            0, 128)[None, :], mask=rids[:, None] < count).to(tl.float32)

    scale = tl.maximum(tl.max(tl.abs(x), 1) / 448.0, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    xq = x / scale[:, None]

    tl.store(xs_ptr + si * nb + cid * count + rid * 128 + tl.arange(0, 128),
             scale,
             mask=rids < count)
    tl.store(xq_ptr + si * N + rid * 128 * N + cid * 128 + tl.arange(0, 128)[:,
                                                           None] * N + tl.arange(
        0, 128)[None, :],
             xq,
             mask=rids[:, None] < count)

    scale = tl.maximum(tl.max(tl.abs(x), 0) / 448.0, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    xq = x / scale[None, :]

    tl.store(xts_ptr + m_block * N + rid * N + cid * 128 + tl.arange(0, 128),
             scale)
    tl.store(
        xtq_ptr + si * N + rid * 128 + cid * 128 * count + tl.arange(0, 128)[:,
                                                           None] * count + tl.arange(
            0, 128)[None, :], tl.trans(xq), mask=rids[None, :] < count)


def triton_batch_blockwise_quant(xs,
                                 token_count_per_expert,
                                 splits,
                                 round_scale=False):
    """
    select and quant, used in megatron 0.12 flex moe
    Args:
        xs: [bs, dim]
        token_count_per_expert: [n_experts]
        splits: python int list of token_count_per_expert
        round_scale: whether round scale to power of 2

    Returns:
        - x_q: 
        - x_scale: 
        - xt_q: 
        - xt_scale: 

    """
    assert xs.is_contiguous()
    M, N = xs.shape
    n_experts = token_count_per_expert.size(0)
    device = xs.device
    x_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    # intra layout and inner layput are not consist,
    # tensors will be viewed after splitting
    x_scale = torch.empty((M * N // 128,), device=device, dtype=torch.float32)
    xt_q = torch.empty((M * N,), device=device,
                       dtype=torch.float8_e4m3fn)
    blocks = sum([(x + 127) // 128 for x in splits])
    xt_scale = torch.empty((blocks * N,), device=device,
                           dtype=torch.float32)

    if M == 0:
        return x_q, x_scale, xt_q, xt_scale

    grid = (n_experts, triton.cdiv(max(splits), 128), N // 128)
    batch_blockwise_quant_kernel[grid](
        xs,
        token_count_per_expert,
        x_q,
        x_scale,
        xt_q,
        xt_scale,
        N,
        n_experts,
        round_scale,
        num_stages=2,
        num_warps=4
    )

    return x_q, x_scale, xt_q, xt_scale
