# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
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

    w = tl.load(weight_ptr + index * dim + tl.arange(0, DIM),
                mask=tl.arange(0, DIM) < dim)
    tl.store(y_ptr + pid * dim + tl.arange(0, DIM), w,
             mask=tl.arange(0, DIM) < dim)


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

    grid = (M,)
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
def atomic_embedding_backward_kernel(
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

    y = tl.load(y_ptr + bid * stride_0 + lid * stride_1 + tl.arange(0, DIM),
                mask=tl.arange(0, DIM) < dim)
    tl.atomic_add(grad_ptr + index * dim + tl.arange(0, DIM), y,
                  mask=tl.arange(0, DIM) < dim)


def triton_atomic_embedding_backward(y, x, g_ptr, dtype=torch.bfloat16):
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
    atomic_embedding_backward_kernel[grid](
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


@triton.jit
def sync_embedding_backward_kernel(grad_output_ptr,
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
        c0, c1 = tl.split(c01)
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
        g = tl.load(
            grad_output_ptr + bid * stride_0 + lid * stride_1 + tl.arange(0,
                                                                          DIM),
            mask=tl.arange(0, DIM) < dim).to(tl.float32)
        outputs += g
    tl.store(grad_ptr + input_id * dim + tl.arange(0, DIM), outputs,
             mask=tl.arange(0, DIM) < dim)


def triton_sync_embedding_backward(grad_output, x, g_ptr, dtype=torch.bfloat16):
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
    unique_ids, unique_counts = torch.unique_consecutive(sorted_ids,
                                                         return_counts=True)
    accum_counts = torch.cumsum(unique_counts, 0)
    DIM = triton.next_power_of_2(dim)
    num_stages = 3
    num_warps = 4

    grid = (unique_ids.size(0),)
    sync_embedding_backward_kernel[grid](
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
def scan_and_count_split_kernel(id_ptr,
                                counts_ptr,
                                unique_id_ptr,
                                unique_count_ptr,
                                L,
                                B: tl.constexpr):
    bid = tl.program_id(axis=0)
    sid = tl.program_id(axis=1)
    ns = tl.num_programs(1)

    ids = tl.load(id_ptr + bid * L + sid * B + tl.arange(0, B))

    write_index = bid * L + sid * B
    unique_count = 0
    stop = False
    while not stop:
        min_id = tl.min(ids)
        if min_id == 2 ** 30:
            stop = True
        else:
            count = tl.sum(tl.where(ids == min_id, 1, 0))
            ids = tl.where(ids <= min_id, 2 ** 30, ids)
            tl.store(counts_ptr + write_index, count)
            tl.store(unique_id_ptr + write_index, min_id)
            write_index += 1
            unique_count += 1
    tl.store(unique_count_ptr + bid * ns + sid, unique_count)


@triton.jit
def scan_and_count_merge_kernel(
        counts_ptr,
        unique_id_ptr,
        unique_count_ptr,
        accum_counts_ptr,
        L,
        B: tl.constexpr,
        T: tl.constexpr):
    bid = tl.program_id(axis=0)
    write_index = bid * (L + 1) + 1
    tl.store(accum_counts_ptr + bid * (L + 1), 0)
    pre_id = -1
    for i in range(T):
        uc = tl.load(unique_count_ptr + bid * T + i)
        counts = tl.load(counts_ptr + bid * L + i * B + tl.arange(0, B),
                         mask=tl.arange(0, B) < uc)
        uids = tl.load(unique_id_ptr + bid * L + i * B + tl.arange(0, B),
                       mask=tl.arange(0, B) < uc, other=2 ** 30)
        min_id = tl.min(uids)
        offset = tl.where(min_id == pre_id, -1, 0)
        pre_id = tl.max(tl.where(tl.arange(0, B) < uc, uids, -1))
        tl.atomic_add(accum_counts_ptr + write_index + offset + tl.arange(0, B),
                      counts, mask=tl.arange(0, B) < uc)
        write_index += uc + offset


def triton_scan_and_count(ids):
    assert ids.is_contiguous()
    shape = ids.shape
    device = ids.device
    assert len(shape) in (1, 2)
    if len(shape) == 2:
        B, L = ids.shape
        BLOCK = 256
        assert L % BLOCK == 0
        T = L // BLOCK
        counts = torch.empty((B, L,), dtype=torch.int32, device=device)
        unique_ids = torch.empty((B, L), dtype=torch.int32, device=device)
        unique_counts = torch.empty((B, T), dtype=torch.int32, device=device)
        accum_counts = torch.zeros((B, L + 1), dtype=torch.int32, device=device)
    else:
        L = shape[0]
        B = 1
        BLOCK = 256
        assert L % BLOCK == 0
        T = L // BLOCK
        counts = torch.empty((L,), dtype=torch.int32, device=device)
        unique_ids = torch.empty((L,), dtype=torch.int32, device=device)
        unique_counts = torch.empty((T,), dtype=torch.int32, device=device)
        accum_counts = torch.zeros((L + 1,), dtype=torch.int32, device=device)

    num_stages = 3
    num_warps = 1
    grid = (B, T)
    scan_and_count_split_kernel[grid](
        ids,
        counts,
        unique_ids,
        unique_counts,
        L,
        BLOCK,
        num_stages=num_stages,
        num_warps=num_warps
    )

    num_stages = 3
    num_warps = 1
    grid = (B,)
    scan_and_count_merge_kernel[grid](
        counts,
        unique_ids,
        unique_counts,
        accum_counts,
        L,
        BLOCK,
        T,
        num_stages=num_stages,
        num_warps=num_warps
    )
    accum_counts = torch.cumsum(accum_counts, -1)

    return accum_counts


@triton.jit
def deprecated_scan_and_count_kernel(id_ptr,
                                     accum_counts_ptr,
                                     B: tl.constexpr,
                                     T: tl.constexpr):
    accum = 0
    write_index = 0
    last_min_id = -1
    for i in range(T):
        ids = tl.load(id_ptr + i * B + tl.arange(0, B))
        stop = False
        while not stop:
            min_id = tl.min(ids)
            if min_id == 2 ** 30:
                stop = True
            else:
                if min_id != last_min_id:
                    tl.store(accum_counts_ptr + write_index, accum)
                    last_min_id = min_id
                    write_index += 1
                count = tl.sum(tl.where(ids == min_id, 1, 0))
                ids = tl.where(ids <= min_id, 2 ** 30, ids)
                accum += count
    tl.store(accum_counts_ptr + write_index, accum)


def triton_deprecated_scan_and_count(ids):
    M = ids.numel()
    accum_counts = -torch.ones((M + 1,), dtype=torch.int32, device=ids.device)
    num_stages = 3
    num_warps = 1

    B = 64
    assert M % B == 0
    T = M // B
    grid = (1,)
    deprecated_scan_and_count_kernel[grid](
        ids,
        accum_counts,
        B,
        T,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return accum_counts


@triton.jit
def embedding_backward_kernel(grad_output_ptr,
                              sorted_ids_ptr,
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
    c01 = tl.load(accum_counts_ptr + pid + tl.arange(0, 2))
    c0, c1 = tl.split(c01)
    if c0 == c1:
        return

    count = c1 - c0
    input_id = tl.load(sorted_ids_ptr + c0).to(tl.int64)

    if T == 0:
        grad_ptr = g_ptr.to(tl.pointer_type(tl.float32))
    else:
        grad_ptr = g_ptr.to(tl.pointer_type(tl.bfloat16))

    outputs = tl.zeros((DIM,), dtype=tl.float32)

    for i in range(count):
        pos = tl.load(sorted_indices_ptr + c0 + i)
        bid = pos // L
        lid = pos % L
        g = tl.load(
            grad_output_ptr + bid * stride_0 + lid * stride_1 + tl.arange(0,
                                                                          DIM),
            mask=tl.arange(0, DIM) < dim).to(tl.float32)
        outputs += g
    tl.store(grad_ptr + input_id * dim + tl.arange(0, DIM), outputs,
             mask=tl.arange(0, DIM) < dim)


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
    accum_counts = triton_scan_and_count(sorted_ids)
    DIM = triton.next_power_of_2(dim)
    num_stages = 3
    num_warps = 2

    grid = (B * L,)
    embedding_backward_kernel[grid](
        grad_output,
        sorted_ids,
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
