
# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import os
import torch
import triton
import triton.language as tl
from triton import Config


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def tma_persistent_matmul_kernel(
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        SM: tl.constexpr, ):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tid_c = start_pid - SM
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tid in tl.range(start_pid, num_tiles, SM, flatten=True):
        pid_m, pid_n = _compute_pid(tid, num_pid_in_group, num_pid_m,
                                    GROUP_SIZE_M)
        offs_a = pid_m * BLOCK_SIZE_M
        offs_b = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(k_tiles):
            offs_k = k * BLOCK_SIZE_K
            a = a_desc.load([offs_a, offs_k])
            b = b_desc.load([offs_b, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tid_c += SM
        pid_m, pid_n = _compute_pid(tid_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_a_acc = pid_m * BLOCK_SIZE_M
        offs_b_acc = pid_n * BLOCK_SIZE_N

        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        c_desc.store([offs_a_acc, offs_b_acc], acc0)
        c_desc.store([offs_a_acc, offs_b_acc + BLOCK_SIZE_N // 2], acc1)


def triton_tma_persistent_matmul(a, b):
    M, K = a.shape
    N, K = b.shape
    dtype = torch.float32

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    SM = torch.cuda.get_device_properties("cuda").multi_processor_count

    BLOCK_M = 128
    BLOCK_K = 64
    BLOCK_N = 64
    GROUP_SIZE_M = 8

    a_desc = triton.tools.tensor_descriptor.TensorDescriptor(a, a.shape,
                                                             a.stride(),
                                                             [BLOCK_M, BLOCK_K])
    b_desc = triton.tools.tensor_descriptor.TensorDescriptor(b, b.shape,
                                                             b.stride(),
                                                             [BLOCK_N, BLOCK_K])
    c_desc = triton.tools.tensor_descriptor.TensorDescriptor(c, c.shape,
                                                             c.stride(),
                                                             [BLOCK_M,
                                                              BLOCK_N // 2])

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        return (min(SM,
                    triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), ),)

    tma_persistent_matmul_kernel[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_K,
        BLOCK_N,
        GROUP_SIZE_M,
        SM=SM)
    return c
