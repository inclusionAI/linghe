

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
def embedding_backward_kernel(
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
    embedding_backward_kernel[grid](
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




# @triton.jit
# def embedding_backward_kernel(y_ptr,
#                              x_ptr,
#                              xs_ptr,
#                              g_ptr,
#                              dim,
#                              DIM: tl.constexpr,
#                              T: tl.constexpr,
#                              B: tl.constexpr):
#     pid = tl.program_id(axis=0).to(tl.int64)

#     if T == 0:
#         grad_ptr = g_ptr.to(tl.pointer_type(tl.float32))
#     else:
#         grad_ptr = g_ptr.to(tl.pointer_type(tl.bfloat16))

#     idx = -1
#     outputs = tl.zeros((DIM,), dtype=tl.float32)
#     for i in range(B):
#         pos = tl.load(xs_ptr + pid * B + i)
#         index = tl.load(x_ptr + pos)
#         y = tl.load(y_ptr + pos * dim + tl.arange(0, DIM), mask=tl.arange(0, DIM) < dim)
#         if (idx == -1) | (index == idx):
#             outputs += y.to(tl.float32)
#         else:
#             tl.atomic_add(grad_ptr + idx * dim + tl.arange(0, DIM), outputs, mask=tl.arange(0, DIM) < dim)
#             outputs = y.to(tl.float32)
#         idx = index
#     tl.atomic_add(grad_ptr + idx * dim + tl.arange(0, DIM), outputs, mask=tl.arange(0, DIM) < dim)



# def triton_embedding_backward(y, x, g_ptr, dtype=torch.bfloat16):
#     """
#     inplace update embedding weight gradient
#     Args:
#         y: gradient of output
#         x: input ids Tensor
#         g_ptr: data_ptr of embedding weight gradient
#     Returns:
#         None
#     """
#     assert y.is_contiguous()
#     assert dtype in (torch.bfloat16, torch.float32)
#     T = 0 if dtype == torch.float32 else 1
#     M = x.numel()
#     dim = y.size(-1)

#     xs = torch.argsort(x.view(-1), stable=False)

#     DIM = triton.next_power_of_2(dim)
#     num_stages = 4
#     num_warps = 32

#     B = 2
#     assert M % B == 0
#     grid = (M//B, )
#     embedding_backward_kernel[grid](
#         y,
#         x,
#         xs,
#         g_ptr,
#         dim,
#         DIM,
#         T,
#         B,
#         num_stages=num_stages,
#         num_warps=num_warps
#     )



# @triton.jit
# def embedding_backward_kernel(y_ptr,
#                              x_ptr,
#                              xs_ptr,
#                              g_ptr,
#                              dim,
#                              DIM: tl.constexpr,
#                              T: tl.constexpr,
#                              B: tl.constexpr):
#     pid = tl.program_id(axis=0).to(tl.int64)

#     if T == 0:
#         grad_ptr = g_ptr.to(tl.pointer_type(tl.float32))
#     else:
#         grad_ptr = g_ptr.to(tl.pointer_type(tl.bfloat16))

#     pos = tl.load(xs_ptr + pid * B + tl.arange(0, B))
#     index = tl.load(x_ptr + pos)
#     if tl.max(index) == tl.min(index):

#         # y = tl.load(y_ptr + pos[:,None] * dim + tl.arange(0, DIM), mask=tl.arange(0, DIM)[None, :] < dim)
#         # ys = tl.sum(y, 0)
#         # tl.atomic_add(grad_ptr + tl.max(index) * dim + tl.arange(0, DIM), ys, mask=tl.arange(0, DIM) < dim)
#         outputs = tl.zeros((DIM,), dtype=tl.float32)
#         idx = 0
#         for i in range(B):
#             p = tl.load(xs_ptr + pid * B + i)
#             idx = tl.load(x_ptr + p)
#             o = tl.load(y_ptr + p * dim + tl.arange(0, DIM), mask=tl.arange(0, DIM) < dim)
#             outputs += o.to(tl.float32)
#         tl.atomic_add(grad_ptr + idx * dim + tl.arange(0, DIM), outputs, mask=tl.arange(0, DIM) < dim)
#     else:
#         for i in range(B):
#             p = tl.load(xs_ptr + pid * B + i)
#             idx = tl.load(x_ptr + p)
#             o = tl.load(y_ptr + p * dim + tl.arange(0, DIM), mask=tl.arange(0, DIM) < dim)
#             tl.atomic_add(grad_ptr + idx * dim + tl.arange(0, DIM), o, mask=tl.arange(0, DIM) < dim)


# def triton_embedding_backward(y, x, g_ptr, dtype=torch.bfloat16):
#     """
#     inplace update embedding weight gradient
#     Args:
#         y: gradient of output
#         x: input ids Tensor
#         g_ptr: data_ptr of embedding weight gradient
#     Returns:
#         None
#     """
#     assert y.is_contiguous()
#     assert dtype in (torch.bfloat16, torch.float32)
#     T = 0 if dtype == torch.float32 else 1
#     M = x.numel()
#     dim = y.size(-1)

#     xs = torch.argsort(x.view(-1), stable=False)

#     DIM = triton.next_power_of_2(dim)
#     num_stages = 4
#     num_warps = 32

#     B = 2
#     assert M % B == 0
#     grid = (M//B, )
#     embedding_backward_kernel[grid](
#         y,
#         x,
#         xs,
#         g_ptr,
#         dim,
#         DIM,
#         T,
#         B,
#         num_stages=num_stages,
#         num_warps=num_warps
#     )
