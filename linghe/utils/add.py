# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import List
import torch
import triton
import triton.language as tl


@triton.jit
def inplace_add_kernel(x_ptr,
                       y_ptr,
                       N,
                       B: tl.constexpr,
                       EVEN: tl.constexpr,
                       ACCUM: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid * B + tl.arange(0, B)
    if ACCUM:
        if EVEN:
            x = tl.load(x_ptr + offs).to(tl.float32)
            y = tl.load(y_ptr + offs).to(tl.float32)
            tl.store(x_ptr + offs, x + y)
        else:
            mask = offs < N
            x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
            y = tl.load(y_ptr + offs, mask=mask).to(tl.float32)
            tl.store(x_ptr + offs, x + y, mask=mask)
    else:
        if EVEN:
            y = tl.load(y_ptr + offs)
            tl.store(x_ptr + offs, y)
        else:
            mask = offs < N
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(x_ptr + offs, y, mask=mask)


def triton_inplace_add(x: torch.Tensor, y: torch.Tensor, accum: bool = True):
    """
    inplace add y to x
    Args:
        x: Tensor
        y: Tensor
        accum: x += y if accum=True else x.copy_(y)

    Returns:
        updated x
    """
    assert x.is_contiguous() and y.is_contiguous()
    N = x.numel()
    B = 512
    EVEN = N % B == 0
    num_stages = 2
    num_warps = 4

    grid = (triton.cdiv(N, B), )
    inplace_add_kernel[grid](
        x,
        y,
        N,
        B,
        EVEN,
        accum,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return x


@triton.jit
def batch_inplace_add_kernel(x_ptrs,
                             y_ptrs,
                             size_ptr,
                             T, 
                             B: tl.constexpr, 
                             ACCUM: tl.constexpr,
                             XT: tl.constexpr,
                             YT: tl.constexpr):
    tid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    size = tl.load(size_ptr + tid)

    if XT == 0:
        x_ptr = tl.load(x_ptrs + tid).to(tl.pointer_type(tl.float32))
    elif XT == 1:
        x_ptr = tl.load(x_ptrs + tid).to(tl.pointer_type(tl.bfloat16))
    else:
        x_ptr = tl.load(x_ptrs + tid).to(tl.pointer_type(tl.float16))


    if YT == 0:
        y_ptr = tl.load(y_ptrs + tid).to(tl.pointer_type(tl.float32))
    elif YT == 1:
        y_ptr = tl.load(y_ptrs + tid).to(tl.pointer_type(tl.bfloat16))
    else:
        y_ptr = tl.load(y_ptrs + tid).to(tl.pointer_type(tl.float16))

    t = tl.cdiv(size, B * T)
    offs = bid * t * B + tl.arange(0, B)

    if ACCUM:
        for i in range(t):
            x = tl.load(x_ptr + offs, mask=offs < size).to(tl.float32)
            y = tl.load(y_ptr + offs, mask=offs < size).to(tl.float32)
            tl.store(x_ptr + offs, x + y, mask=offs < size)
            offs += B
    else:
        for i in range(t):
            y = tl.load(y_ptr + offs, mask=offs < size)
            tl.store(x_ptr + offs, y, mask=offs < size)
            offs += B

def triton_batch_inplace_add(xs: List[torch.Tensor], ys: List[torch.Tensor], accum: bool = True):
    """
    inplace add y to x
    Args:
        xs: a list of Tensor
        ys: a list of Tensor
        accum: x += y if accum=True else x.copy_(y)

    Returns:
        updated xs
    """
    # assert all([x.is_contiguous() for x in xs]) 
    # assert all([y.is_contiguous() for y in ys])

    device = xs[0].device
    sizes = torch.tensor([x.numel() for x in xs],
                         dtype=torch.int64).cuda(device, non_blocking=True)
    x_ptrs = torch.tensor([x.data_ptr() for x in xs],
                        dtype=torch.int64).cuda(device, non_blocking=True)
    y_ptrs = torch.tensor([y.data_ptr() for y in ys],
                        dtype=torch.int64).cuda(device, non_blocking=True)
    x_dtype = xs[0].dtype
    assert x_dtype in (torch.bfloat16, torch.float32, torch.float16)
    if x_dtype == torch.float32:
        XT = 0
    elif x_dtype == torch.bfloat16:
        XT = 1
    else:
        XT = 2
    y_dtype = ys[0].dtype
    assert y_dtype in (torch.bfloat16, torch.float32, torch.float16)
    if y_dtype == torch.float32:
        YT = 0
    elif y_dtype == torch.bfloat16:
        YT = 1
    else:
        YT = 2

    T = 512
    B = 1024
    num_stages = 3
    num_warps = 4

    grid = (len(xs), T)
    batch_inplace_add_kernel[grid](
        x_ptrs,
        y_ptrs,
        sizes,
        T,
        B,
        accum,
        XT,
        YT,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return xs
