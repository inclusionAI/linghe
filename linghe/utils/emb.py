

# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
from typing import Iterable, Optional, Tuple
import triton
import triton.language as tl


@triton.jit
def embedding_forward_kernel(x_ptr,
                             y_ptr,
                             w_ptr,
                             dim,
                             DIM: tl.constexpr):
    pid = tl.program_id(axis=0).to(tl.int64)
    index = tl.load(x_ptr + pid)
    weight_ptr = w_ptr.to(tl.pointer_type(tl.bfloat16))

    x = tl.load(weight_ptr + index * dim + tl.arange(0, DIM), mask=tl.arange(0, DIM) < dim)
    tl.store(y_ptr + pid * dim + tl.arange(0, DIM), x, mask=tl.arange(0, DIM) < dim)


def triton_embedding_forward(x, w_ptr, dim=4096, dtype=torch.bfloat16):
    """
    inplace add y to x
    Args:
        x: input ids Tensor
        t_ptr: data_ptr of embedding weight
    Returns:
        embedding output
    """
    assert x.is_contiguous()
    assert dtype == torch.bfloat16
    M = x.numel()
    y = torch.empty((x.shape + (dim,)), device=x.device, dtype=dtype)
    DIM = triton.next_power_of_2(dim)
    num_stages = 2
    num_warps = 8

    grid = (M, )
    embedding_forward_kernel[grid](
        x,
        y,
        w_ptr,
        dim,
        DIM,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return y


@triton.jit
def embedding_backward_kernel(y_ptr,
                             x_ptr,
                             g_ptr,
                             dim,
                             DIM: tl.constexpr):
    pid = tl.program_id(axis=0).to(tl.int64)
    index = tl.load(x_ptr + pid)

    grad_ptr = g_ptr.to(tl.pointer_type(tl.bfloat16))

    x = tl.load(y_ptr + pid * dim + tl.arange(0, DIM), mask=tl.arange(0, DIM) < dim)
    tl.atomic_add(grad_ptr + index * dim + tl.arange(0, DIM), x, mask=tl.arange(0, DIM) < dim)


def triton_embedding_backward(y, x, g_ptr, dtype=torch.bfloat16):
    """
    inplace update embedding weight gradient
    Args:
        y: gradient of output
        x: input ids Tensor
        g_ptr: data_ptr of embedding weight gradient
    Returns:
        None
    """
    assert y.is_contiguous()
    assert dtype == torch.bfloat16
    M = x.numel()
    dim = y.size(-1)

    DIM = triton.next_power_of_2(dim)
    num_stages = 4
    num_warps = 16

    grid = (M, )
    embedding_backward_kernel[grid](
        y,
        x,
        g_ptr,
        dim,
        DIM,
        num_stages=num_stages,
        num_warps=num_warps
    )
