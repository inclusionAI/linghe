# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import Optional

import torch
import triton
import triton.language as tl

@triton.jit
def group_rms_norm_gate_infer_forward_kernel(x_ptr, gate_ptr, weight_ptr, out_ptr, eps,
                            DIM: tl.constexpr, 
                            D: tl.constexpr, 
                            GROUP_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    weight = tl.load(weight_ptr + tl.arange(0, DIM))
    weight = tl.reshape(weight, [GROUP_SIZE, D])
    x_offs = pid * DIM + tl.arange(0, GROUP_SIZE)[:, None] * D + tl.arange(0, D)[
                                                            None, :]
    x = tl.load(x_ptr + x_offs).to(tl.float32)
    offs = pid * DIM + tl.arange(0, GROUP_SIZE)[:, None] * D + tl.arange(0, D)[
                                                            None, :]
    g = tl.load(gate_ptr + offs).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x, axis=1) / D + eps)
    x = (x / rms[:, None]) * weight * tl.sigmoid(g)
    tl.store(out_ptr + offs, x)


def triton_group_rms_norm_gate_infer_forward(x: torch.Tensor, 
                                       gate: torch.Tensor, 
                                       weight: torch.Tensor, 
                                       eps=1e-6, 
                                       group_size=4
                                       ):
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
    assert dim <= 8192 and triton.next_power_of_2(dim) == dim and triton.next_power_of_2(group_size) == group_size
    d = dim // group_size
    device = x.device
    out = torch.empty((tokens, dim), device=device, dtype=x.dtype)
    grid = (tokens,)
    group_rms_norm_gate_infer_forward_kernel[grid](
        x,
        gate,
        weight.data,
        out,
        eps,
        dim, 
        d,
        group_size,
        num_stages=3,
        num_warps=4
    )
    return out