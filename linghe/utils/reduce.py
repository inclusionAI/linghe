# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def abs_max_kernel(x_ptr,
                   scale_ptr,
                   smooth_scale_ptr,
                   output_ptr,
                   min_value,
                   M, N,
                   H: tl.constexpr,
                   W: tl.constexpr,
                   EVEN: tl.constexpr,
                   QUANTIZED: tl.constexpr):
    pid = tl.program_id(axis=0)
    # col-wise read, col-wise write
    x_max = tl.zeros((W,), dtype=tl.float32)
    m = tl.cdiv(M, H)
    offs = pid * W + tl.arange(0, H)[:, None] * N + tl.arange(0, W)
    if QUANTIZED:
        smooth_scale = tl.load(smooth_scale_ptr + pid * W + tl.arange(0, W))[
                       None, :]
    for i in range(m):
        if EVEN:
            x = tl.load(x_ptr + offs).to(tl.float32)
        else:
            x = tl.load(x_ptr + offs,
                        mask=i * H + tl.arange(0, H)[:, None] < M).to(
                tl.float32)
        if QUANTIZED:
            scale = tl.load(scale_ptr + i * H + tl.arange(0, H),
                            mask=i * H + tl.arange(0, H) < M)
            x = x * scale[:, None] * smooth_scale
        x_max = tl.maximum(x_max, tl.max(tl.abs(x), axis=0))
        offs += H * N

    scale = tl.maximum(x_max, min_value)

    tl.store(output_ptr + pid * W + tl.arange(0, W), scale)


def triton_abs_max(x, scale=None, smooth_scale=None, min_value=1e-30, axis=0):
    """
    columnwise abs max of x, it is used in smooth quantization
    Args:
        x: input tensor, may be quantized tensor
        scale: quantization scale if x is quantized
        smooth_scale: optional smooth scale
        min_value: output = max(max(abs(x,0)), min_value)
        axis: reduce axis

    Returns:
        max tensor
    """
    assert x.is_contiguous()
    assert axis == 0
    N = x.size(-1)
    M = x.numel() // N
    device = x.device
    maxs = torch.empty((N,), device=device, dtype=torch.float32)
    quantized = scale is not None
    H = 512
    W = 16
    assert N % W == 0
    EVEN = M % H == 0
    grid = (triton.cdiv(N, W),)
    abs_max_kernel[grid](
        x,
        scale,
        smooth_scale,
        maxs,
        min_value,
        M, N,
        H, W,
        EVEN,
        quantized,
        num_stages=2,
        num_warps=4
    )
    return maxs


@triton.jit
def batch_count_zero_kernel(input_ptrs, size_ptr, count_ptr, B: tl.constexpr):
    tid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    sm = tl.num_programs(axis=1)
    count = 0

    size = tl.load(size_ptr + tid)
    input_ptr = tl.load(input_ptrs + tid).to(tl.pointer_type(tl.float32))
    t = tl.cdiv(size, B * sm)
    offs = bid * t * B + tl.arange(0, B)
    for i in tl.range(t, flatten=True):
        x = tl.load(input_ptr + offs, mask=offs < size, other=1).to(tl.float32)
        count += tl.sum(tl.where(x == 0, 1, 0))
        offs += B

    tl.store(count_ptr + tid * sm + bid, count)


def triton_batch_count_zero(xs):
    """
    count zero in tensor list, it is used to monitor zeros in gradient tensor
    Args:
        xs: input tensors

    Returns:
        a single-value int64 tensor
    """
    assert all([x.is_contiguous() for x in xs])
    device = xs[0].device
    sizes = torch.tensor([x.numel() for x in xs],
                         dtype=torch.int64).cuda(device, non_blocking=True)
    ptrs = torch.tensor([x.data_ptr() for x in xs],
                        dtype=torch.int64).cuda(device, non_blocking=True)

    block = 2048
    tensor_count = len(xs)
    counts = torch.empty((tensor_count, block), device=device,
                         dtype=torch.int64)
    B = 1024
    grid = (tensor_count, block)
    batch_count_zero_kernel[grid](
        ptrs,
        sizes,
        counts,
        B,
        num_stages=2,
        num_warps=2
    )
    count = counts.sum()
    return count


@triton.jit
def norm_kernel(input_ptr, tmp_ptr, m,
                B: tl.constexpr,
                ORD: tl.constexpr):
    pid = tl.program_id(axis=0).to(tl.int64)

    offs = pid * B + tl.arange(0, B)
    x = tl.load(input_ptr + offs, mask=offs < m, other=0).to(tl.float32)
    if ORD == 2:
        sums = tl.sum(x * x)
    elif ORD == 1:
        sums = tl.sum(tl.abs(x))
    elif ORD == -1:
        sums = tl.max(tl.abs(x))

    tl.store(tmp_ptr + pid, sums)


def triton_norm(x, ord=2, norm=True, scalar=True):
    """
    calculate norm.
    Args:
        x: input tensor.
        ord: the order of tensor. -1 means 'inf' ord.
        norm:
            only used with ord in (1, 2)
            True: (sum(sum(abs(x)**ord) x for x in xs))**(1/ord) 
            False: sum(sum(abs(x)**ord) x for x in xs))

    Returns:
        a scalar if scalar=True else a single-value fp32 tensor
    """
    assert x.is_contiguous()
    assert ord in (1, 2, -1)
    # assert all([x.is_contiguous() for x in xs])
    device = x.device
    m = x.numel()
    B = 512
    T = triton.cdiv(m, B)
    tmp = torch.empty((T,), device=device, dtype=torch.float32)
    grid = (T,)
    norm_kernel[grid](
        x,
        tmp,
        m,
        B,
        ord,
        num_stages=2,
        num_warps=2
    )
    if ord == -1:
        output = tmp.max()
    else:
        output = tmp.sum()
        if ord == 2 and norm:
            output = torch.sqrt(output)
    if not scalar:
        output = output.unsqueeze(0)
    return output


@triton.jit
def batch_norm_kernel(input_ptrs, size_ptr, tmp_ptr,
                      DT: tl.constexpr,
                      B: tl.constexpr,
                      ORD: tl.constexpr,
                      HP: tl.constexpr):
    tid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1).to(tl.int64)
    sm = tl.num_programs(axis=1)
    if HP:
        sums = tl.zeros((B,), dtype=tl.float64)
    else:
        sums = tl.zeros((B,), dtype=tl.float32)

    size = tl.load(size_ptr + tid)
    if DT == 0:
        input_ptr = tl.load(input_ptrs + tid).to(tl.pointer_type(tl.float32))
    else:
        input_ptr = tl.load(input_ptrs + tid).to(tl.pointer_type(tl.bfloat16))
    t = tl.cdiv(size, B * sm)
    offs = bid * t * B + tl.arange(0, B)
    for i in range(t):
        x = tl.load(input_ptr + offs, mask=offs < size, other=0)
        if HP:
            x = x.to(tl.float64)
        else:
            x = x.to(tl.float32)
        if ORD == 2:
            sums += x * x
        elif ORD == 1:
            sums += tl.abs(x)
        elif ORD == -1:
            sums = tl.maximum(sums, tl.abs(x))
        offs += B

    if ORD == -1:
        sums = tl.max(sums)
    else:
        sums = tl.sum(sums)
    tl.store(tmp_ptr + tid * sm + bid, sums)


def triton_batch_norm(xs, ord=2, norm=True, scalar=True, high_precision=True):
    """
    treat multiple tensors as a single tensor and calculate norm.
    Args:
        xs: Tensor lists.
        ord: the order of tensor. -1 means 'inf' ord.
        norm:
            only used with ord in (1, 2)
            True: (sum(sum(abs(x)**ord) x for x in xs))**(1/ord) 
            False: sum(sum(abs(x)**ord) x for x in xs))

    Returns:
        a scalar if scalar=True else a single-value fp32 tensor
    """
    if len(xs) == 0:
        return torch.zeros(() if scalar else (1,), device='cuda',
                           dtype=torch.float32)
    dtype = xs[0].dtype
    assert dtype in (torch.float32, torch.bfloat16)
    assert all([x.is_contiguous() and x.dtype == dtype for x in xs])
    assert ord in (1, 2, -1)

    device = xs[0].device
    sizes = torch.tensor([x.numel() for x in xs],
                         dtype=torch.int64).cuda(device, non_blocking=True)
    ptrs = torch.tensor([x.data_ptr() for x in xs],
                        dtype=torch.int64).cuda(device, non_blocking=True)

    DT = 0 if dtype == torch.float32 else 1
    sm = 256
    tensor_count = len(xs)
    tmp = torch.empty((tensor_count, sm), device=device,
                      dtype=torch.float64 if high_precision else torch.float32)
    B = 128
    grid = (tensor_count, sm)
    batch_norm_kernel[grid](
        ptrs,
        sizes,
        tmp,
        DT,
        B,
        ord,
        high_precision,
        num_stages=2,
        num_warps=2
    )
    if ord == -1:
        output = tmp.max()
    elif ord == 1:
        output = tmp.sum()
    else:
        if norm:
            output = torch.sqrt(tmp.sum())
        else:
            output = tmp.sum()
    if not scalar:
        output = output.unsqueeze(0)
    if high_precision:
        output = output.float()
    return output
