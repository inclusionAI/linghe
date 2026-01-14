# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from linghe.experimental.symm_mem_barrier import symm_mem_sync
from linghe.utils.emb import triton_scan_and_count

"""
distributed embedding with vocab parallel
"""


@triton.jit
def tp_embedding_lookup_forward_kernel(input_ids_ptr,
                                       weights_ptr,
                                       outputs_ptr,
                                       buffer_ptrs,
                                       signal_ptrs,
                                       V,
                                       d,
                                       D: tl.constexpr,
                                       SIZE: tl.constexpr,
                                       RANK: tl.constexpr):
    pid = tl.program_id(0)
    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))

    input_id = tl.load(input_ids_ptr + pid)
    mask = tl.arange(0, D) < d

    if (input_id >= RANK * V) & (input_id < (RANK + 1) * V):

        buffer_ptr = tl.load(buffer_ptrs + RANK).to(
            tl.pointer_type(tl.bfloat16))
        buffer_ptr = tl.multiple_of(buffer_ptr, 16)
        w = tl.load(weights_ptr + input_id % V * D + tl.arange(0, D), mask=mask)
        tl.store(outputs_ptr + pid * D + tl.arange(0, D), w, mask=mask)

        tl.store(buffer_ptr + pid * D + tl.arange(0, D), w, mask=mask)
        symm_mem_sync(
            signal_ptrs,
            None,
            RANK,
            SIZE,
            hasPreviousMemAccess=True,
            hasSubsequentMemAccess=True,
        )
    else:
        symm_mem_sync(
            signal_ptrs,
            None,
            RANK,
            SIZE,
            hasPreviousMemAccess=True,
            hasSubsequentMemAccess=True,
        )
        buffer_ptr = tl.load(buffer_ptrs + input_id // V).to(
            tl.pointer_type(tl.bfloat16))
        buffer_ptr = tl.multiple_of(buffer_ptr, 16)
        w = tl.load(buffer_ptr + pid * D + tl.arange(0, D), mask=mask)
        tl.store(outputs_ptr + pid * D + tl.arange(0, D), w, mask=mask)


"""
input_ids is the same cross ranks
"""


def triton_tp_embedding_lookup_forward(input_ids, weights, hdl, group):
    group_size = hdl.world_size
    group_rank = hdl.rank
    shape = input_ids.shape
    V, d = weights.shape
    D = triton.next_power_of_2(d)

    device = weights.device
    dtype = weights.dtype
    assert len(shape) in (1, 2) and dtype == torch.bfloat16

    if len(shape) == 2:
        M = shape[0] * shape[1]
        outputs = torch.empty((shape[0], shape[1], d), device=device,
                              dtype=dtype)
    else:
        M = shape[0]
        outputs = torch.empty((M, d), device=device, dtype=dtype)

    num_warps = 4
    num_stages = 3
    tp_embedding_lookup_forward_kernel[(M,)](
        input_ids,
        weights,
        outputs,
        hdl.buffer_ptrs_dev,
        hdl.signal_pad_ptrs_dev,
        V,
        d,
        D,
        group_size,
        group_rank,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return outputs


@triton.jit
def tp_embedding_lookup_backward_kernel(grad_output_ptr,
                                        sorted_ids_ptr,
                                        sorted_indices_ptr,
                                        accum_counts_ptr,
                                        g_ptr,
                                        buffer_ptrs,
                                        signal_ptrs,
                                        stride_0,
                                        stride_1,
                                        dim,
                                        V,
                                        B,
                                        L,
                                        DIM: tl.constexpr,
                                        T: tl.constexpr,
                                        SIZE: tl.constexpr,
                                        RANK: tl.constexpr
                                        ):
    pid = tl.program_id(axis=0).to(tl.int64)
    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))

    c01 = tl.load(accum_counts_ptr + pid + tl.arange(0, 2))
    c0, c1 = tl.split(c01)
    if c0 == c1:
        return

    count = c1 - c0
    input_id = tl.load(sorted_ids_ptr + c0).to(tl.int64)
    mask = tl.arange(0, DIM) < dim

    if T == 0:
        grad_ptr = g_ptr.to(tl.pointer_type(tl.float32))
    else:
        grad_ptr = g_ptr.to(tl.pointer_type(tl.bfloat16))

    outputs = tl.zeros((DIM,), dtype=tl.float32)

    if (input_id >= RANK * V) & (input_id < (RANK + 1) * V):
        for i in range(count):
            pos = tl.load(sorted_indices_ptr + c0 + i)
            bid = pos // L
            lid = pos % L
            g = tl.load(
                grad_output_ptr + bid * stride_0 + lid * stride_1 + tl.arange(0,
                                                                              DIM),
                mask=mask).to(tl.float32)
            outputs += g
        symm_mem_sync(
            signal_ptrs,
            None,
            RANK,
            SIZE,
            hasPreviousMemAccess=True,
            hasSubsequentMemAccess=True,
        )

        for j in range(SIZE):
            if j != RANK:
                buffer_ptr = tl.load(buffer_ptrs + j).to(
                    tl.pointer_type(tl.bfloat16))
                buffer_ptr = tl.multiple_of(buffer_ptr, 16)
                g = tl.load(buffer_ptr + pid * DIM + tl.arange(0, DIM),
                            mask=mask)
                outputs += g
        tl.store(grad_ptr + input_id % V * DIM + tl.arange(0, DIM), outputs,
                 mask=mask)

    else:

        for i in range(count):
            pos = tl.load(sorted_indices_ptr + c0 + i)
            bid = pos // L
            lid = pos % L
            g = tl.load(
                grad_output_ptr + bid * stride_0 + lid * stride_1 + tl.arange(0,
                                                                              DIM),
                mask=mask).to(tl.float32)
            outputs += g

        buffer_ptr = tl.load(buffer_ptrs + RANK).to(
            tl.pointer_type(tl.bfloat16))
        buffer_ptr = tl.multiple_of(buffer_ptr, 16)
        tl.store(buffer_ptr + pid * DIM + tl.arange(0, DIM), outputs, mask=mask)
        symm_mem_sync(
            signal_ptrs,
            None,
            RANK,
            SIZE,
            hasPreviousMemAccess=True,
            hasSubsequentMemAccess=True,
        )


def triton_tp_embedding_lookup_backward(grad_output, x, g_ptr, vocab_size, hdl,
                                        group, dtype=torch.bfloat16):
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

    group_size = hdl.world_size
    group_rank = hdl.rank
    B, L, dim = grad_output.shape
    stride_0 = grad_output.stride(0)
    stride_1 = grad_output.stride(1)

    sorted_ids, sorted_indices = torch.sort(x.view(-1), stable=False)
    accum_counts = triton_scan_and_count(sorted_ids)
    DIM = triton.next_power_of_2(dim)
    num_stages = 3
    num_warps = 2

    grid = (B * L,)
    tp_embedding_lookup_backward_kernel[grid](
        grad_output,
        sorted_ids,
        sorted_indices,
        accum_counts,
        g_ptr,
        hdl.buffer_ptrs_dev,
        hdl.signal_pad_ptrs_dev,
        stride_0,
        stride_1,
        dim,
        vocab_size,
        B,
        L,
        DIM,
        T,
        group_size,
        group_rank,
        num_stages=num_stages,
        num_warps=num_warps
    )


@triton.jit
def sp_embedding_lookup_forward_kernel(input_ids_ptr,
                                       weights_ptr,
                                       outputs_ptr,
                                       buffer_ptrs,
                                       signal_ptrs,
                                       M,
                                       V,
                                       d,
                                       D: tl.constexpr,
                                       SIZE: tl.constexpr,
                                       RANK: tl.constexpr):
    pid = tl.program_id(0)
    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    mask = tl.arange(0, D) < d

    for chunk in range(SIZE):
        input_id = tl.load(input_ids_ptr + chunk * M + pid)
        if chunk == RANK:
            if (input_id >= RANK * V) & (input_id < (RANK + 1) * V):
                w = tl.load(weights_ptr + input_id % V * D + tl.arange(0, D),
                            mask=mask)
                tl.store(outputs_ptr + pid * D + tl.arange(0, D), w, mask=mask)
            else:
                symm_mem_sync(
                    signal_ptrs,
                    None,
                    RANK,
                    SIZE,
                    hasPreviousMemAccess=True,
                    hasSubsequentMemAccess=True,
                )
                buffer_ptr = tl.load(buffer_ptrs + input_id // V).to(
                    tl.pointer_type(tl.bfloat16))
                buffer_ptr = tl.multiple_of(buffer_ptr, 16)
                w = tl.load(buffer_ptr + pid * D + tl.arange(0, D), mask=mask)
                tl.store(outputs_ptr + pid % M * D + tl.arange(0, D), w,
                         mask=mask)
        else:
            if (input_id >= RANK * V) & (input_id < (RANK + 1) * V):
                w = tl.load(weights_ptr + input_id % V * D + tl.arange(0, D),
                            mask=mask)
                buffer_ptr = tl.load(buffer_ptrs + RANK).to(
                    tl.pointer_type(tl.bfloat16))
                buffer_ptr = tl.multiple_of(buffer_ptr, 16)
                tl.store(buffer_ptr + pid * D + tl.arange(0, D), w, mask=mask)
                symm_mem_sync(
                    signal_ptrs,
                    None,
                    RANK,
                    SIZE,
                    hasPreviousMemAccess=True,
                    hasSubsequentMemAccess=True,
                )


"""
input_ids is different cross ranks
"""


def triton_sp_embedding_lookup_forward(input_ids, weights, hdl, group):
    group_size = hdl.world_size
    group_rank = hdl.rank
    shape = input_ids.shape
    V, d = weights.shape
    D = triton.next_power_of_2(d)

    device = weights.device
    dtype = weights.dtype
    assert len(shape) in (1, 2) and dtype == torch.bfloat16

    if len(shape) == 2:
        M = shape[0] * shape[1]
        outputs = torch.empty((shape[0], shape[1], d), device=device,
                              dtype=dtype)
        gathered_input_ids = torch.empty((group_size, shape[0], shape[1]),
                                         dtype=torch.long, device=device)
    else:
        M = shape[0]
        outputs = torch.empty((M, d), device=device, dtype=dtype)
        gathered_input_ids = torch.empty((group_size, M), dtype=torch.long,
                                         device=device)
    dist.all_gather_into_tensor(gathered_input_ids, input_ids, group=group)

    num_warps = 4
    num_stages = 3

    sp_embedding_lookup_forward_kernel[(M,)](
        gathered_input_ids,
        weights,
        outputs,
        hdl.buffer_ptrs_dev,
        hdl.signal_pad_ptrs_dev,
        M,
        V,
        d,
        D,
        group_size,
        group_rank,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return outputs


@triton.jit
def sp_embedding_lookup_backward_kernel(grad_output_ptr,
                                        sorted_ids_ptr,
                                        sorted_indices_ptr,
                                        accum_counts_ptr,
                                        g_ptr,
                                        buffer_ptrs,
                                        signal_ptrs,
                                        stride_0,
                                        stride_1,
                                        dim,
                                        M,
                                        V,
                                        B,
                                        L,
                                        DIM: tl.constexpr,
                                        T: tl.constexpr,
                                        SIZE: tl.constexpr,
                                        RANK: tl.constexpr
                                        ):
    pid = tl.program_id(axis=0).to(tl.int64)
    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
    mask = tl.arange(0, DIM) < dim

    for chunk in range(SIZE):
        c01 = tl.load(
            accum_counts_ptr + chunk * (L + 1) + pid + tl.arange(0, 2))
        c0, c1 = tl.split(c01)
        if c0 != c1:

            count = c1 - c0
            input_id = tl.load(sorted_ids_ptr + chunk * M + c0).to(tl.int64)

            if T == 0:
                grad_ptr = g_ptr.to(tl.pointer_type(tl.float32))
            else:
                grad_ptr = g_ptr.to(tl.pointer_type(tl.bfloat16))

            outputs = tl.zeros((DIM,), dtype=tl.float32)

            if chunk == RANK:
                if (input_id >= RANK * V) & (input_id < (RANK + 1) * V):
                    for i in range(count):
                        pos = tl.load(sorted_indices_ptr + chunk * M + c0 + i)
                        bid = pos // L
                        lid = pos % L
                        g = tl.load(
                            grad_output_ptr + bid * stride_0 + lid * stride_1 + tl.arange(
                                0, DIM), mask=mask).to(tl.float32)
                        outputs += g

                    tl.atomic_add(
                        grad_ptr + input_id % V * DIM + tl.arange(0, DIM),
                        outputs, mask=mask, sem='relaxed')

                else:

                    for i in range(count):
                        pos = tl.load(sorted_indices_ptr + chunk * M + c0 + i)
                        bid = pos // L
                        lid = pos % L
                        g = tl.load(
                            grad_output_ptr + bid * stride_0 + lid * stride_1 + tl.arange(
                                0, DIM), mask=mask).to(tl.float32)
                        outputs += g

                    # save to dst addr
                    buffer_ptr = tl.load(buffer_ptrs + input_id // V).to(
                        tl.pointer_type(tl.float32))
                    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
                    tl.store(buffer_ptr + pid * DIM + tl.arange(0, DIM),
                             outputs, mask=mask)
                    symm_mem_sync(
                        signal_ptrs,
                        None,
                        RANK,
                        SIZE,
                        hasPreviousMemAccess=True,
                        hasSubsequentMemAccess=True,
                    )
            else:
                if (input_id >= RANK * V) & (input_id < (RANK + 1) * V):
                    symm_mem_sync(
                        signal_ptrs,
                        None,
                        RANK,
                        SIZE,
                        hasPreviousMemAccess=True,
                        hasSubsequentMemAccess=True,
                    )

                    buffer_ptr = tl.load(buffer_ptrs + RANK).to(
                        tl.pointer_type(tl.float32))
                    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
                    g = tl.load(buffer_ptr + pid * DIM + tl.arange(0, DIM),
                                mask=mask)
                    tl.atomic_add(
                        grad_ptr + input_id % V * DIM + tl.arange(0, DIM), g,
                        mask=mask, sem='relaxed')


def triton_sp_embedding_lookup_backward(grad_output, input_ids, g_ptr,
                                        vocab_size, hdl, group,
                                        dtype=torch.bfloat16,
                                        gathered_input_ids=None):
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
    group_size = hdl.world_size
    group_rank = hdl.rank

    shape = input_ids.shape
    device = input_ids.device
    dim = grad_output.size(-1)

    if len(shape) == 2:
        B, L = shape
        M = B * L
        if gathered_input_ids is None:
            gathered_input_ids = torch.empty((group_size, B, L),
                                             dtype=torch.long, device=device)
            dist.all_gather_into_tensor(gathered_input_ids, input_ids,
                                        group=group)
    else:
        M = shape[0]
        if gathered_input_ids is None:
            gathered_input_ids = torch.empty((group_size, M), dtype=torch.long,
                                             device=device)
            dist.all_gather_into_tensor(gathered_input_ids, input_ids,
                                        group=group)

    stride_0 = grad_output.stride(0)
    stride_1 = grad_output.stride(1)

    sorted_ids, sorted_indices = torch.sort(
        gathered_input_ids.view(group_size, M), stable=False, dim=-1)
    accum_counts = triton_scan_and_count(sorted_ids)

    DIM = triton.next_power_of_2(dim)
    num_stages = 3
    num_warps = 2
    grid = (M,)
    sp_embedding_lookup_backward_kernel[grid](
        grad_output,
        sorted_ids,
        sorted_indices,
        accum_counts,
        g_ptr,
        hdl.buffer_ptrs_dev,
        hdl.signal_pad_ptrs_dev,
        stride_0,
        stride_1,
        dim,
        M,
        vocab_size,
        B,
        L,
        DIM,
        T,
        group_size,
        group_rank,
        num_stages=num_stages,
        num_warps=num_warps
    )
