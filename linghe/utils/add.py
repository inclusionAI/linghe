# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def inplace_add_kernel(
    x_ptr,
    y_ptr,
    M,
    N,
    H: tl.constexpr,
    W: tl.constexpr,
    EVEN: tl.constexpr,
    ACCUM: tl.constexpr,
):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    offs = (
        rid * H * N + cid * W + tl.arange(0, H)[:, None] * N + tl.arange(0, W)[None, :]
    )
    if ACCUM:
        if EVEN:
            x = tl.load(x_ptr + offs)
            y = tl.load(y_ptr + offs).to(tl.float32)
            tl.store(x_ptr + offs, x + y)
        else:
            x = tl.load(
                x_ptr + offs,
                mask=(cid * W + tl.arange(0, W)[None, :] < N)
                & (rid * H + tl.arange(0, H)[:, None] < M),
            )
            y = tl.load(
                y_ptr + offs,
                mask=(cid * W + tl.arange(0, W)[None, :] < N)
                & (rid * H + tl.arange(0, H)[:, None] < M),
            )
            tl.store(
                x_ptr + offs,
                x + y,
                mask=(cid * W + tl.arange(0, W)[:, None] < N)
                & (rid * H + tl.arange(0, H)[None, :] < M),
            )
    else:
        if EVEN:
            y = tl.load(y_ptr + offs).to(tl.float32)
            tl.store(x_ptr + offs, y)
        else:
            y = tl.load(
                y_ptr + offs,
                mask=(cid * W + tl.arange(0, W)[None, :] < N)
                & (rid * H + tl.arange(0, H)[:, None] < M),
            )
            tl.store(
                x_ptr + offs,
                y,
                mask=(cid * W + tl.arange(0, W)[:, None] < N)
                & (rid * H + tl.arange(0, H)[None, :] < M),
            )


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
    N = x.shape[-1]
    M = x.numel() // N
    # M, N = x.shape
    H = 128
    W = 128
    EVEN = M % H == 0 and N % W == 0
    num_stages = 2
    num_warps = 8

    grid = (triton.cdiv(M, H), triton.cdiv(N, W))
    inplace_add_kernel[grid](
        x, y, M, N, H, W, EVEN, accum, num_stages=num_stages, num_warps=num_warps
    )
    return x
