# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx


@triton.jit
def inplace_add_warp_specialized_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_TILE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE * BLOCK_TILE
    loop = BLOCK_TILE // 2
    with tlx.async_tasks():
        with tlx.async_task("default"):
            for i in tl.range(loop):
                offsets = block_start + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                x = tl.load(x_ptr + offsets, mask=mask)
                y = tl.load(y_ptr + offsets, mask=mask)
                tl.store(x_ptr + offsets, x + y, mask=mask)
                # offsets += BLOCK_SIZE
        with tlx.async_task(num_warps=4):
            for i in tl.range(loop):
                offsets = block_start + i * BLOCK_SIZE + BLOCK_SIZE * loop + tl.arange(0, BLOCK_SIZE)
                mask = offsets < n_elements
                a = tl.load(x_ptr + offsets, mask=mask)
                b = tl.load(y_ptr + offsets, mask=mask)
                tl.store(x_ptr + offsets, a + b, mask=mask)
                # offsets += BLOCK_SIZE


def triton_inplace_add_warp_specialized(x: torch.Tensor, y: torch.Tensor):
    assert x.is_contiguous() and y.is_contiguous()
    n_elements = x.numel()
    BLOCK_TILE = 64
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"] * BLOCK_TILE), )
    inplace_add_warp_specialized_kernel[grid](x, y, n_elements, BLOCK_SIZE=1024, BLOCK_TILE=BLOCK_TILE)
    return x

