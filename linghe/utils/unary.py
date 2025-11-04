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





@triton.jit
def batch_clip_kernel(input_ptrs, size_ptr, clip_value, 
                              B: tl.constexpr):
    tid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    sm = tl.num_programs(axis=1)

    size = tl.load(size_ptr + tid)
    input_ptr = tl.load(input_ptrs + tid).to(tl.pointer_type(tl.float32))
    t = tl.cdiv(size, B * sm)
    offs = bid * t * B + tl.arange(0, B)
    for i in range(t):
        x = tl.load(input_ptr + offs, mask=offs < size, other=0).to(tl.float32)
        # x = tl.minimum(tl.maximum(x, -clip_value), clip_value)
        # tl.store(input_ptr + offs, x, mask=offs < size)

        max_val = tl.max(tl.abs(x))
        if max_val > clip_value:
            x = tl.minimum(tl.maximum(x, -clip_value), clip_value)
            tl.store(input_ptr + offs, x, mask=offs < size)
        offs += B


def triton_batch_clip(xs, clip_value=100.0):
    """
    return [clip(x, -clip_value, clip_value) for x in xs],
    used to clip gradient.
    Args:
        xs: Tensor lists.
        clip_value: a python float scale
    Returns:
        updated xs
    """
    assert all([x.is_contiguous() for x in xs])
    device = xs[0].device
    sizes = torch.tensor([x.numel() for x in xs], dtype=torch.int64,
                         device=device)
    ptrs = torch.tensor([x.data_ptr() for x in xs], dtype=torch.int64,
                        device=device)

    sm = 64
    tensor_count = len(xs)
    B = 512
    grid = (tensor_count, sm)
    batch_clip_kernel[grid](
        ptrs,
        sizes,
        clip_value,
        B,
        num_stages=2,
        num_warps=2
    )
    return xs