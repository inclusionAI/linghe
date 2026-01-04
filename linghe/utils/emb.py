

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

    w = tl.load(weight_ptr + index * dim + tl.arange(0, DIM), mask=tl.arange(0, DIM) < dim)
    tl.store(y_ptr + pid * dim + tl.arange(0, DIM), w, mask=tl.arange(0, DIM) < dim)


def triton_embedding_forward(x, w_ptr, dim=4096, dtype=torch.bfloat16):
    """
    inplace add y to x
    Args:
        x: input ids Tensor
        w_ptr: data_ptr of embedding weight
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
def embedding_backward_kernel(grad_output_ptr,
                             unique_ids_ptr,
                             sorted_indices_ptr,
                             accum_counts_ptr,
                             g_ptr,
                             stride_0,
                             stride_1,
                             dim,
                             B,
                             L,
                             DIM: tl.constexpr,
                             T: tl.constexpr,
                             ):
    pid = tl.program_id(axis=0).to(tl.int64)

    if pid == 0:
        c0 = 0
        c0 = c0.to(tl.int64)
        c1 = tl.load(accum_counts_ptr)
    else:
        c01 = tl.load(accum_counts_ptr + pid - 1 + tl.arange(0, 2))
        c0, c1= tl.split(c01)
    count = c1 - c0
    input_id = tl.load(unique_ids_ptr + pid).to(tl.int64)

    if T == 0:
        grad_ptr = g_ptr.to(tl.pointer_type(tl.float32))
    else:
        grad_ptr = g_ptr.to(tl.pointer_type(tl.bfloat16))

    outputs = tl.zeros((DIM,), dtype=tl.float32)

    for i in range(count):
        pos = tl.load(sorted_indices_ptr + c0 + i)
        bid = pos // L
        lid = pos % L
        g = tl.load(grad_output_ptr + bid * stride_0 + lid * stride_1 + tl.arange(0, DIM), mask=tl.arange(0, DIM) < dim).to(tl.float32)
        outputs += g
    tl.store(grad_ptr + input_id * dim + tl.arange(0, DIM), outputs, mask=tl.arange(0, DIM) < dim)


def triton_embedding_backward(grad_output, x, g_ptr, dtype=torch.bfloat16):
    """
    inplace update embedding weight gradient
    Args:
        y: gradient of output
        x: input ids Tensor
        g_ptr: data_ptr of embedding weight gradient
    Returns:
        None
    """
    assert dtype in (torch.bfloat16, torch.float32)
    T = 0 if dtype == torch.float32 else 1
    shape = x.shape
    assert len(shape) == 2
    B, L, dim = grad_output.shape
    stride_0 = grad_output.stride(0)
    stride_1 = grad_output.stride(1)

    sorted_ids, sorted_indices = torch.sort(x.view(-1), stable=False)
    unique_ids, unique_counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    accum_counts = torch.cumsum(unique_counts, 0)
    DIM = triton.next_power_of_2(dim)
    num_stages = 3
    num_warps = 4

    grid = (unique_ids.size(0), )
    embedding_backward_kernel[grid](
        grad_output,
        unique_ids,
        sorted_indices,
        accum_counts,
        g_ptr,
        stride_0,
        stride_1,
        dim,
        B,
        L,
        DIM,
        T,
        num_stages=num_stages,
        num_warps=num_warps
    )



@triton.jit
def deprecated_embedding_backward_kernel(
    y_ptr, 
    x_ptr, 
    g_ptr, 
    stride_0,
    stride_1,
    dim, 
    DIM: tl.constexpr,
    T: tl.constexpr
):
    bid = tl.program_id(axis=0).to(tl.int64)
    lid = tl.program_id(axis=1)
    B = tl.num_programs(0)
    L = tl.num_programs(1)
    index = tl.load(x_ptr + bid * L + lid)

    if T == 0:
        grad_ptr = g_ptr.to(tl.pointer_type(tl.float32))
    else:
        grad_ptr = g_ptr.to(tl.pointer_type(tl.bfloat16))

    y = tl.load(y_ptr + bid * stride_0 + lid * stride_1 + tl.arange(0, DIM), mask=tl.arange(0, DIM) < dim)
    tl.atomic_add(grad_ptr + index * dim + tl.arange(0, DIM), y, mask=tl.arange(0, DIM) < dim)


def triton_deprecated_embedding_backward(y, x, g_ptr, dtype=torch.bfloat16):
    """
    inplace update embedding weight gradient
    Args:
        y: gradient of output
        x: input ids Tensor
        g_ptr: data_ptr of embedding weight gradient
    Returns:
        None
    """
    assert dtype in (torch.bfloat16, torch.float32)
    shape = x.shape
    assert len(shape) == 2
    T = 0 if dtype == torch.float32 else 1
    B, L, dim = y.shape
    stride_0 = y.stride(0)
    stride_1 = y.stride(1)

    DIM = triton.next_power_of_2(dim)
    num_stages = 2
    num_warps = 16

    grid = (B, L)
    deprecated_embedding_backward_kernel[grid](
        y,
        x,
        g_ptr,
        stride_0,
        stride_1,
        dim,
        DIM,
        T,
        num_stages=num_stages,
        num_warps=num_warps
    )


