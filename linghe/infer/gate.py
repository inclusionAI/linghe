# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def group_rms_norm_gate_kernel(x_ptr,
                               gate_ptr,
                               weight_ptr,
                               out_ptr,
                               eps,
                               DIM: tl.constexpr,
                               D: tl.constexpr,
                               GROUP_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    weight = tl.load(weight_ptr + tl.arange(0, DIM))
    weight = tl.reshape(weight, [GROUP_SIZE, D]).to(tl.float32)
    x_offs = pid * DIM + tl.arange(0, GROUP_SIZE)[:, None] * D + tl.arange(0,
                                                                           D)[
                                                                 None, :]
    x = tl.load(x_ptr + x_offs).to(tl.float32)
    offs = pid * DIM + tl.arange(0, GROUP_SIZE)[:, None] * D + tl.arange(0, D)[
                                                               None, :]
    g = tl.load(gate_ptr + offs).to(tl.float32)
    rms = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    x = (x * rms[:, None]) * weight * tl.sigmoid(g)
    tl.store(out_ptr + offs, x)


def triton_group_rms_norm_gate(x: torch.Tensor,
                               gate: torch.Tensor,
                               weight: torch.Tensor,
                               eps=1e-6,
                               group_size=4,
                               dtype=torch.bfloat16):
    """
    norm and gate in linear attention
    Args:
        x: output of attn, [tokens, n_heads, head_dim]
        gate: gate tensor, [tokens, dim]
        weight: rms norm weight, [dim]
        eps: epsilon of rms norm
        group_size: group size of group rms norm
    Returns:
        output tensor, [tokens, dim]
    """
    # row-wise read, row-wise write
    tokens, dim = gate.shape
    assert (dim <= 8192
            and triton.next_power_of_2(dim) == dim
            and triton.next_power_of_2(group_size) == group_size)
    d = dim // group_size
    device = x.device
    out = torch.empty((tokens, dim), device=device, dtype=dtype)
    grid = (tokens,)
    group_rms_norm_gate_kernel[grid](
        x,
        gate,
        weight.data,
        out,
        eps,
        dim,
        d,
        group_size,
        num_stages=3,
        num_warps=4)
    return out



@triton.jit
def block_group_rms_norm_gate_kernel(x_ptr,
                               gate_ptr,
                               weight_ptr,
                               out_ptr,
                               scale_ptr,
                               eps,
                               D: tl.constexpr,
                               d: tl.constexpr,
                               GROUP_SIZE: tl.constexpr,
                               ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    M = tl.num_programs(axis=0)
    weight = tl.load(weight_ptr + tl.arange(0, D))
    weight = tl.reshape(weight, [GROUP_SIZE, d])
    x_offs = pid * D + tl.arange(0, GROUP_SIZE)[:, None] * d + tl.arange(0,
                                                                           d)[
                                                                 None, :]
    x = tl.load(x_ptr + x_offs).to(tl.float32)
    offs = pid * D + tl.arange(0, GROUP_SIZE)[:, None] * d + tl.arange(0, d)[
                                                               None, :]
    g = tl.load(gate_ptr + offs).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x, axis=1) / d + eps)
    x = (x / rms[:, None]) * weight * tl.sigmoid(g)

    b : tl.constexpr = d // 128 
    B : tl.constexpr = D // 128 
    x = tl.reshape(x, (GROUP_SIZE, b, 128))

    scale = tl.max(tl.abs(x), 2) / 448.0
    scale = tl.where(scale == 0.0, 1.0, scale)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    PM = ( M + 3 ) // 4 * 4

    tl.store(scale_ptr + tl.arange(0, B) * PM + pid,
             tl.reshape(scale, (B, )))

    x = x / scale[:, :, None]
    x = tl.reshape(x, [GROUP_SIZE, d], can_reorder=False)

    tl.store(out_ptr + offs, x)


def triton_block_group_rms_norm_gate(x: torch.Tensor,
                               gate: torch.Tensor,
                               weight: torch.Tensor,
                               eps=1e-6,
                               group_size=4,
                               round_scale=False):
    """
    norm and gate in linear attention
    Args:
        x: output of attn, [tokens, n_heads, head_dim]
        gate: gate tensor, [tokens, dim]
        weight: rms norm weight, [dim]
        eps: epsilon of rms norm
        group_size: group size of group rms norm
    Returns:
        output tensor, [tokens, dim]
    """
    # row-wise read, row-wise write
    M, D = gate.shape
    assert (D <= 8192
            and triton.next_power_of_2(D) == D
            and triton.next_power_of_2(group_size) == group_size)
    d = D // group_size
    device = x.device
    out = torch.empty((M, D), device=device, dtype=torch.float8_e4m3fn)
    PM = (M + 3) // 4 * 4
    scale = torch.empty((D // 128, PM), device=device, dtype=torch.float32)
    grid = (M, )
    block_group_rms_norm_gate_kernel[grid](
        x,
        gate,
        weight.data,
        out,
        scale,
        eps,
        D,
        d,
        group_size,
        ROUND=round_scale,
        num_stages=3,
        num_warps=4)
    return out, scale[:, :M].t()

