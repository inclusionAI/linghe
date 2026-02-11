# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from linghe.experimental.symm_mem_barrier import symm_mem_sync


@triton.jit
def split_tp_mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    split_atomic_ptr,
    buffer_ptrs,
    signal_ptrs,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    SIZE: tl.constexpr,
    RANK: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    nbn = tl.num_programs(1)
    pid_k = tl.program_id(axis=2)
    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))

    k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_COUNT)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + pid_k * K // SPLIT_COUNT + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + pid_k * K // SPLIT_COUNT + offs_n[None, :] * K + offs_k[:, None]

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        c = tl.dot(a, b, c)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]

    if SPLIT_COUNT == 1:
        buffer_ptr = tl.load(buffer_ptrs + RANK).to(tl.pointer_type(tl.float32))
        buffer_ptr = tl.multiple_of(buffer_ptr, 16)

        tl.store(buffer_ptr + offs_m[:, None] * N + offs_n[None, :], c)
        symm_mem_sync(
            signal_ptrs,
            None,
            RANK,
            SIZE,
            hasPreviousMemAccess=True,
            hasSubsequentMemAccess=True,
        )

        outputs = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for i in tl.static_range(SIZE):
            buffer_ptr = tl.load(buffer_ptrs + i).to(tl.pointer_type(tl.float32))
            buffer_ptr = tl.multiple_of(buffer_ptr, 16)
            outputs += tl.load(buffer_ptr + offs_m[:, None] * N + offs_n[None, :])
        tl.store(c_ptrs, outputs)

        # for j in range(0, RANK):
        #     buffer_ptr = tl.load(buffer_ptrs + j).to(tl.pointer_type(tl.float32))
        #     buffer_ptr = tl.multiple_of(buffer_ptr, 16)
        #     c += tl.load(buffer_ptr + offs_m[:, None] * N + offs_n[None, :])
        # for j in range(RANK+1, SIZE):
        #     buffer_ptr = tl.load(buffer_ptrs + j).to(tl.pointer_type(tl.float32))
        #     buffer_ptr = tl.multiple_of(buffer_ptr, 16)
        #     c += tl.load(buffer_ptr + offs_m[:, None] * N + offs_n[None, :])
        # tl.store(c_ptrs, c)

    else:
        tl.atomic_add(c_ptrs, c, sem="relaxed")
        # tl.atomic_add(c_ptrs, c)
        atomic_index = pid_m * nbn + pid_n
        tl.atomic_add(split_atomic_ptr + atomic_index, 1)

        if pid_k == SPLIT_COUNT - 1:  # pid_k == 0 will result in error output
            tl.debug_barrier()
            while tl.load(split_atomic_ptr + atomic_index) < SPLIT_COUNT:
                pass

            c = tl.load(c_ptrs)

            buffer_ptr = tl.load(buffer_ptrs + RANK).to(tl.pointer_type(tl.float32))
            buffer_ptr = tl.multiple_of(buffer_ptr, 16)
            tl.store(buffer_ptr + offs_m[:, None] * N + offs_n[None, :], c)

            symm_mem_sync(
                signal_ptrs,
                None,
                RANK,
                SIZE,
                hasPreviousMemAccess=True,
                hasSubsequentMemAccess=True,
            )

            outputs = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for j in tl.static_range(SIZE):
                buffer_ptr = tl.load(buffer_ptrs + j).to(tl.pointer_type(tl.float32))
                buffer_ptr = tl.multiple_of(buffer_ptr, 16)
                outputs += tl.load(buffer_ptr + offs_m[:, None] * N + offs_n[None, :])
            tl.store(c_ptrs, outputs)

            # for j in range(0, RANK):
            #     buffer_ptr = tl.load(buffer_ptrs + j).to(tl.pointer_type(tl.float32))
            #     buffer_ptr = tl.multiple_of(buffer_ptr, 16)
            #     c += tl.load(buffer_ptr + offs_m[:, None] * N + offs_n[None, :])
            # for j in range(RANK+1, SIZE):
            #     buffer_ptr = tl.load(buffer_ptrs + j).to(tl.pointer_type(tl.float32))
            #     buffer_ptr = tl.multiple_of(buffer_ptr, 16)
            #     c += tl.load(buffer_ptr + offs_m[:, None] * N + offs_n[None, :])
            # tl.store(c_ptrs, c)


def triton_split_tp_gemm(
    x: torch.Tensor, w: torch.Tensor, hdl, group: dist.ProcessGroup
):
    """
    tensor-parallel fc2 in the shared expert, use split-k implementation
    y = all_reduce(x @ fc2)
    Args:
        a: left matrix with bf16 precision
        b: right matrix with bf16 precision

    Returns:
        c: all-reduced output
    """
    assert x.is_contiguous() and w.is_contiguous()
    M, K = x.size()
    N, K = w.size()
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = max([x for x in [16, 32, 64, 128] if N % x == 0])
    SPLIT_COUNT = 1  # min(triton.cdiv(K, 2048), 4)
    assert M % BLOCK_SIZE_M == 0 and K % BLOCK_SIZE_K == 0
    assert K % (BLOCK_SIZE_K * SPLIT_COUNT) == 0

    device = x.device
    if SPLIT_COUNT == 1:
        c = torch.empty(M, N, dtype=x.dtype, device=device)
        split_atomic_signal = None
    else:
        c = torch.zeros(M, N, dtype=torch.float32, device=device)
        split_atomic_signal = torch.zeros(
            M // BLOCK_SIZE_M * N // BLOCK_SIZE_N, dtype=torch.int32, device=device
        )
    group_size = hdl.world_size
    group_rank = hdl.rank

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
        SPLIT_COUNT,
    )  # noqa
    num_warps = 4
    num_stages = 3
    split_tp_mm_kernel[grid](
        x,
        w,
        c,
        split_atomic_signal,
        hdl.buffer_ptrs_dev,
        hdl.signal_pad_ptrs_dev,
        M,
        N,
        K,
        BLOCK_SIZE_K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        SPLIT_COUNT,
        group_size,
        group_rank,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if SPLIT_COUNT > 1:
        c = c.to(x.dtype)
    return c
