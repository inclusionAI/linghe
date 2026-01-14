# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def dot_kernel(x_ptr, y_ptr, sum_ptr, M, N, H: tl.constexpr, W: tl.constexpr):
    # rowwise read, rowwise write
    pid = tl.program_id(axis=0)
    offs = pid * W * N + tl.arange(0, W)[:, None] * N + tl.arange(0, H)[None, :]

    n = tl.cdiv(N, H)
    sums = tl.zeros((W,), dtype=tl.float32)
    for i in range(n):
        x = tl.load(x_ptr + offs).to(tl.float32)
        y = tl.load(y_ptr + offs).to(tl.float32)
        sums += tl.sum(x * y, axis=1)
        offs += H

    tl.store(sum_ptr + pid * W + tl.arange(0, W), sums)


def triton_dot(x, y):
    """
    vector dot multiply, output = sum(x*y, 1),
    it is used to calculate gradient of router weight
    Args:
        x:
        y:

    Returns:
        output of sum(x*y, 1)
    """
    assert x.is_contiguous() and y.is_contiguous()
    M, N = x.shape
    H = 128
    W = 16
    assert M % W == 0

    num_stages = 5
    num_warps = 8
    device = x.device
    s = torch.empty((M,), device=device, dtype=x.dtype)
    grid = (triton.cdiv(M, W),)
    dot_kernel[grid](
        x, y, s,
        M, N,
        H, W,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return s


@triton.jit
def inplace_scale_kernel(input_ptr, scale, m, B: tl.constexpr):
    pid = tl.program_id(axis=0).to(tl.int64)

    offs = pid * B + tl.arange(0, B)
    x = tl.load(input_ptr + offs, mask=offs < m)
    x = x * scale
    tl.store(input_ptr + offs, x, mask=offs < m)


def triton_inplace_scale(x, scale):
    """
    inplace scale a tensor.
    Args:
        x: Tensor.
        scale: a python float scale
    Returns:
        x
    """
    assert x.is_contiguous()
    B = 512
    m = x.numel()
    grid = (triton.cdiv(m, B),)
    inplace_scale_kernel[grid](
        x,
        scale,
        m,
        B,
        num_stages=2,
        num_warps=2
    )
    return x


@triton.jit
def batch_scale_kernel(input_ptrs, size_ptr, scale,
                       DT: tl.constexpr,
                       B: tl.constexpr,
                       ZERO: tl.constexpr, ):
    tid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    T = tl.num_programs(axis=1)

    size = tl.load(size_ptr + tid)
    if DT == 0:
        input_ptr = tl.load(input_ptrs + tid).to(tl.pointer_type(tl.float32))
    else:
        input_ptr = tl.load(input_ptrs + tid).to(tl.pointer_type(tl.bfloat16))
    t = tl.cdiv(size, B * T)
    offs = bid.to(tl.int64) * t * B + tl.arange(0, B)
    for i in range(t):
        x = tl.load(input_ptr + offs, mask=offs < size, other=0).to(tl.float32)
        if ZERO:
            x = tl.where(tl.abs(x) == float('inf'), 1.0, 0.0)
        else:
            x = x * scale
        tl.store(input_ptr + offs, x, mask=offs < size)
        offs += B


def triton_batch_scale(xs, scale):
    """
    return [x*scale for x in xs],
    used to scale gradient.
    Args:
        xs: Tensor lists.
        scale: a python float scale
    Returns:
        xs
    """
    if len(xs) == 0:
        return
    dtype = xs[0].dtype
    assert dtype in (torch.float32, torch.bfloat16)
    assert all([x.is_contiguous() and x.dtype == dtype for x in xs])

    device = xs[0].device
    sizes = torch.tensor([x.numel() for x in xs],
                         dtype=torch.int64).cuda(device, non_blocking=True)
    ptrs = torch.tensor([x.data_ptr() for x in xs],
                        dtype=torch.int64).cuda(device, non_blocking=True)

    DT = 0 if dtype == torch.float32 else 1
    T = 256
    tensor_count = len(xs)
    B = 512
    ZERO = scale == 0.0
    grid = (tensor_count, T)
    batch_scale_kernel[grid](
        ptrs,
        sizes,
        scale,
        DT,
        B,
        ZERO,
        num_stages=2,
        num_warps=2
    )
    return xs
