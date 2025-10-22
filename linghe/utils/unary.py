# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def calculate_smooth_scale_kernel(x_ptr, y_ptr, min_value, smooth_coef, 
                                  N,
                                  B: tl.constexpr,
                                  EVEN: tl.constexpr,
                                  ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * B + tl.arange(0, B)
    if EVEN:
        x = tl.load(x_ptr+offs).to(tl.float32)
    else:
        x = tl.load(x_ptr+offs, mask=offs<N).to(tl.float32)
    x = tl.exp(-smooth_coef * tl.log(tl.maximum(x, min_value)))
    if ROUND:
        x = tl.exp2(tl.ceil(tl.log2(x)))
    if EVEN:
        tl.store(y_ptr + offs, x)
    else:
        tl.store(y_ptr + offs, x, mask=offs<N)


def triton_calculate_smooth_scale(x, min_value=1.0, smooth_coef=0.5, inplace=False, round_scale=False):
    N = x.shape[0]
    B = 4096
    if inplace:
        output = x 
    else:
        output = torch.empty((N,), dtype=x.dtype, device=x.device)

    min_value = max(min_value, 1e-30)

    EVEN = N % B == 0
    num_stages = 3
    num_warps = 4
    grid = (triton.cdiv(N, B), )
    calculate_smooth_scale_kernel[grid](
        x, output, 
        min_value,
        smooth_coef,
        N,
        B,
        EVEN,
        round_scale,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return output