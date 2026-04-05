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
    weight = tl.load(weight_ptr + tl.arange(0, DIM)).to(tl.float32)
    weight = tl.reshape(weight, [GROUP_SIZE, D])
    x_offs = pid * DIM + tl.arange(0, GROUP_SIZE)[:, None] * D + tl.arange(0,
                                                                           D)[
                                                                 None, :]
    x = tl.load(x_ptr + x_offs).to(tl.float32)
    offs = pid * DIM + tl.arange(0, GROUP_SIZE)[:, None] * D + tl.arange(0, D)[
                                                               None, :]
    g = tl.load(gate_ptr + offs).to(tl.float32)
    rms = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    x = x * rms[:, None] * weight * tl.sigmoid(g)
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
    assert x.is_contiguous() and gate.is_contiguous() and weight.is_contiguous()
    tokens, dim = gate.shape
    assert dim <= 8192 and triton.next_power_of_2(dim) == dim
    d = dim // group_size
    device = x.device
    out = torch.empty((tokens, dim), device=device, dtype=dtype)
    grid = (tokens,)
    group_rms_norm_gate_kernel[grid](
        x,
        gate,
        weight,
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
                               M,
                               PM,
                               D: tl.constexpr,
                               d: tl.constexpr,
                               GROUP_SIZE: tl.constexpr,
                               ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    mask = pid < M
    b : tl.constexpr = d // 128
    B : tl.constexpr = D // 128

    weight = tl.load(weight_ptr + tl.arange(0, D)).to(tl.float32)
    weight = tl.reshape(weight, [GROUP_SIZE, d])
    offs = pid * D + tl.arange(0, GROUP_SIZE)[:, None] * d + tl.arange(0,
                                                                           d)[
                                                                 None, :]
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    g = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    rms = tl.rsqrt(tl.sum(x * x, axis=1) / d + eps)
    x = x * tl.sigmoid(g) * rms[:, None] * weight[None, :]

    x = tl.reshape(x, (GROUP_SIZE, b, 128))

    scale = tl.maximum(tl.max(tl.abs(x), 2) / 448.0, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    tl.store(scale_ptr + pid + tl.arange(0, B) * PM,
             tl.reshape(scale, (B, )))

    x = x / scale[:, :, None]
    x = tl.reshape(x, [GROUP_SIZE, d])

    tl.store(out_ptr + offs, x, mask=mask)


def triton_block_group_rms_norm_gate(x: torch.Tensor,
                               gate: torch.Tensor,
                               weight: torch.Tensor,
                               eps=1e-6,
                               group_size=4,
                               round_scale=False):
    """
    norm and gate in linear attention
    Args:
        x: output of attn, [tokens, n_heads * head_dim]
        gate: gate tensor, [tokens, dim]
        weight: rms norm weight, [dim]
        eps: epsilon of rms norm
        group_size: group size of group rms norm
    Returns:
        output tensor, [tokens, dim]
    """
    assert x.is_contiguous() and gate.is_contiguous() and weight.is_contiguous()
    # print(f'{x.shape=} {gate.shape=} {weight.shape=} {group_size=}')
    M, D = gate.shape
    assert D <= 8192 and triton.next_power_of_2(D) == D
    d = D // group_size
    PM = triton.cdiv(M, 4) * 4
    device = x.device
    out = torch.empty((M, D), device=device, dtype=torch.float8_e4m3fn)
    scale = torch.empty((D // 128, PM), device=device, dtype=torch.float32)
    grid = (PM, )
    block_group_rms_norm_gate_kernel[grid](
        x,
        gate,
        weight,
        out,
        scale,
        eps,
        M,
        PM,
        D,
        d,
        group_size,
        round_scale,
        num_stages=5,
        num_warps=4)
    return out, scale[:, :M].t()

