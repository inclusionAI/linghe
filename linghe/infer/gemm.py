"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def split_fp32_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    pid_k = tl.program_id(axis=0)

    k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_COUNT)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + pid_k * K // SPLIT_COUNT + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + pid_k * K // SPLIT_COUNT + offs_n[None, :] * K + offs_k[:, None]

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M)
        b = tl.load(b_ptrs)
        c = tl.dot(a, b, c)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    if SPLIT_COUNT == 1:
        tl.store(c_ptrs, c, mask=offs_m[:, None] < M)
    else:
        tl.atomic_add(c_ptrs, c, sem="relaxed", mask=offs_m[:, None] < M)


def triton_split_fp32_gemm(x: torch.Tensor, w: torch.Tensor):
    """
    return fp32 gemm result with fp16/bf16 inputs,
        it's mainly used for MoE router GEMM
    Args:
        a: left matrix with fp16/bf16 precision
        b: right matrix with fp16/bf16 precision

    Returns:
        c: output with fp32 precision
    """
    assert x.is_contiguous() and w.is_contiguous()
    M, K = x.size()
    N, K = w.size()

    MP = triton.next_power_of_2(M)
    BLOCK_SIZE_M = max(min(MP, 64), 16)
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 128
    SPLIT_COUNT = 1
    num_warps = 4
    num_stages = 3

    if MP <= 64:
        if N <= 16 * 128:
            SPLIT_COUNT = 4
            BLOCK_SIZE_N = 16
        elif N <= 64 * 128:
            BLOCK_SIZE_N = 16

    assert N % BLOCK_SIZE_N == 0
    assert K % (BLOCK_SIZE_K * SPLIT_COUNT) == 0

    if SPLIT_COUNT == 1:
        c = torch.empty(M, N, dtype=torch.float32, device=x.device)
    else:
        c = torch.zeros(M, N, dtype=torch.float32, device=x.device)

    grid = (
        SPLIT_COUNT,
        triton.cdiv(M, BLOCK_SIZE_M),
        triton.cdiv(N, BLOCK_SIZE_N),
    )
    split_fp32_gemm_kernel[grid](
        x,
        w,
        c,
        M,
        N,
        K,
        SPLIT_COUNT,
        BLOCK_SIZE_K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


@triton.jit
def split_tile_block_fp8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    stride_a_scale,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    TRANSPOSE_A_SCALE: tl.constexpr,
):
    # a tilewise quantization, b blockwise quantization.
    pid_k = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    if TRANSPOSE_A_SCALE:
        a_s_ptrs = a_s_ptr + offs_m
    else:
        a_s_ptrs = a_s_ptr + offs_m * K // 128
    b_s_ptrs = b_s_ptr + pid_n * BLOCK_SIZE_N // BLOCK_SIZE_K * K // BLOCK_SIZE_K

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    k = K // BLOCK_SIZE_K // SPLIT_COUNT
    for i in range(pid_k * k, pid_k * k + k):
        a = tl.load(a_ptrs + i * BLOCK_SIZE_K, mask=offs_m[:, None] < M)
        b = tl.load(b_ptrs + i * BLOCK_SIZE_K)
        if TRANSPOSE_A_SCALE:
            a_s = tl.load(a_s_ptrs + i * stride_a_scale, mask=offs_m < M)
        else:
            a_s = tl.load(a_s_ptrs + i, mask=offs_m < M)
        b_s = tl.load(b_s_ptrs + i)
        scale = a_s[:, None] * b_s
        accumulator += tl.dot(a, b) * scale
        # accumulators = tl.dot(a, b, accumulator)
        # accumulator += (accumulators - accumulator) * a_s[:, None] * b_s
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    # tl.store(c_ptrs, c)
    mask = offs_m[:, None] < M
    if SPLIT_COUNT > 1:
        tl.atomic_add(c_ptrs, accumulator, mask=mask, sem="relaxed")
    else:
        tl.store(c_ptrs, accumulator, mask=mask)


def triton_split_tile_block_fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    out_dtype=torch.bfloat16,
):
    assert a.is_contiguous() and b.is_contiguous()
    assert b_s.is_contiguous()

    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    stride_a_scale = a_s.stride(1)
    # print(f'{a.shape=} {a.stride()=} {b.shape=} {b.stride()=} {a_s.shape=} {a_s.stride()=} {b_s.shape=} {b_s.stride()=}')
    TRANSPOSE_A_SCALE = not a_s.is_contiguous()

    MP = triton.next_power_of_2(M)
    BLOCK_SIZE_M = max(min(MP, 64), 16)
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 128
    SPLIT_COUNT = 1
    num_warps = 4
    num_stages = 3

    if MP <= 64:
        if N <= 16 * 256:
            SPLIT_COUNT = min(K // BLOCK_SIZE_K, 4)
            BLOCK_SIZE_N = 16
            num_warps = 2 if MP <= 16 else 4
        else:
            BLOCK_SIZE_N = 16
            num_warps = 2 if MP <= 16 else 4
    elif M >= 1024 and N >= 4096:
        BLOCK_SIZE_N = 128

    grid = (SPLIT_COUNT, triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    if SPLIT_COUNT > 1:
        c = torch.zeros(M, N, dtype=out_dtype, device=a.device)
    else:
        c = torch.empty(M, N, dtype=out_dtype, device=a.device)
    split_tile_block_fp8_gemm_kernel[grid](
        a,
        b,
        c,
        a_s,
        b_s,
        stride_a_scale,
        M,
        N,
        K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        SPLIT_COUNT,
        TRANSPOSE_A_SCALE,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return c
