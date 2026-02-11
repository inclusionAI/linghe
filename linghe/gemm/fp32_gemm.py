# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl

# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


# fp32_gemm_configs = [
#     Config({"BLOCK_SIZE_K": block_k, "BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n}, num_stages=num_stages, num_warps=num_warps)
#     for block_k in [64, 128, 256]
#     for block_m in [32, 64, 128]
#     for block_n in [32, 64, 128]
#     for num_stages in [2, 3, 4, 5]
#     for num_warps in [4, 8]
# ]


# @triton.autotune(configs=fp32_gemm_configs, key=["M", "N", "K"])
@triton.jit
def fp32_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs)  # .to(tl.float32)
        b = tl.load(b_ptrs)  # .to(tl.float32)
        c = tl.dot(a, b, c)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


def triton_fp32_gemm(x: torch.Tensor, w: torch.Tensor):
    """
    return fp32 gemm result with fp16/bf16 inputs,
        it's mainly used for MoE router GEMM
        and DO NOT suitable for large size GEMM
    Args:
        a: left matrix with fp16/bf16 precision
        b: right matrix with fp16/bf16 precision

    Returns:
        c: output with fp32 precision
    """
    assert x.is_contiguous() and w.is_contiguous()
    M, K = x.size()
    N, K = w.size()
    assert M % 32 == 0 and K % 128 == 0 and N % 16 == 0
    c = torch.empty(M, N, dtype=torch.float32, device=x.device)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )  # noqa
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = max([x for x in [16, 32, 64, 128] if N % x == 0])
    num_warps = 4
    num_stages = 3
    fp32_gemm_kernel[grid](
        x,
        w,
        c,
        M,
        N,
        K,
        BLOCK_SIZE_K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


# @triton.autotune(configs=fp32_gemm_configs, key=["M", "N", "K"])
@triton.jit
def fp32_gemm_for_backward_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] + offs_k[:, None] * N

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(k):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs).to(tl.float32)
        c = tl.dot(a, b, c)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


def triton_fp32_gemm_for_backward(y: torch.Tensor, w: torch.Tensor):
    """
    mix precision gemm for backward, a@b.float()
    Args:
        a: input gradient, fp32
        b: gemm weight, bf16/fp16
    Returns:
        c: gradient of activation
    """
    assert y.is_contiguous() and w.is_contiguous()
    M, K = y.size()
    K, N = w.size()
    c = torch.empty((M, N), dtype=w.dtype, device=w.device)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )  # noqa
    BLOCK_SIZE_K = max([x for x in [16, 32, 64, 128] if K % x == 0])
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128
    num_warps = 4
    num_stages = 2
    fp32_gemm_for_backward_kernel[grid](
        y,
        w,
        c,
        M,
        N,
        K,
        BLOCK_SIZE_K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


# @triton.autotune(configs=fp32_gemm_configs, key=["M", "N", "K"])
@triton.jit
def fp32_gemm_for_update_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[None, :] + offs_k[:, None] * M
    b_ptrs = b_ptr + offs_n[None, :] + offs_k[:, None] * N

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.trans(tl.load(a_ptrs)).to(tl.float32)
        b = tl.load(b_ptrs).to(tl.float32)
        c = tl.dot(a, b, c)
        a_ptrs += BLOCK_SIZE_K * M
        b_ptrs += BLOCK_SIZE_K * N

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


def triton_fp32_gemm_for_update(y: torch.Tensor, x: torch.Tensor):
    """
    mix precision gemm for updaing weight
    Args:
        y: gradient of output, fp32
        x: input activation, bf16/fp16
    Returns:
        c: gradient of weight
    """
    assert y.is_contiguous() and x.is_contiguous()
    K, M = y.size()
    K, N = x.size()
    c = torch.empty((M, N), dtype=torch.float32, device=x.device)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )  # noqa
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_M = max([x for x in [16, 32] if M % x == 0])
    BLOCK_SIZE_N = 128
    num_warps = 4
    num_stages = 3
    fp32_gemm_for_update_kernel[grid](
        y,
        x,
        c,
        M,
        N,
        K,
        BLOCK_SIZE_K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


@triton.jit
def split_fp32_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_COUNT)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + pid_k * K // SPLIT_COUNT + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + pid_k * K // SPLIT_COUNT + offs_n[None, :] * K + offs_k[:, None]

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs)  # .to(tl.float32)
        b = tl.load(b_ptrs)  # .to(tl.float32)
        c = tl.dot(a, b, c)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    if SPLIT_COUNT == 1:
        tl.store(c_ptrs, c)
    else:
        tl.atomic_add(c_ptrs, c, sem="relaxed")


def triton_split_fp32_gemm(x: torch.Tensor, w: torch.Tensor):
    """
    return fp32 gemm result with fp16/bf16 inputs,
        it's mainly used for MoE router GEMM
        and DO NOT suitable for large size GEMM
    Args:
        a: left matrix with fp16/bf16 precision
        b: right matrix with fp16/bf16 precision

    Returns:
        c: output with fp32 precision
    """
    assert x.is_contiguous() and w.is_contiguous()
    M, K = x.size()
    N, K = w.size()
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = max([x for x in [16, 32, 64, 128] if N % x == 0])
    SPLIT_COUNT = min(triton.cdiv(K, 2048), 4)
    assert M % BLOCK_SIZE_M == 0 and K % BLOCK_SIZE_K == 0
    assert K % (BLOCK_SIZE_K * SPLIT_COUNT) == 0

    if SPLIT_COUNT == 1:
        c = torch.empty(M, N, dtype=torch.float32, device=x.device)
    else:
        c = torch.zeros(M, N, dtype=torch.float32, device=x.device)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
        SPLIT_COUNT,
    )  # noqa

    num_warps = 4
    num_stages = 3
    split_fp32_gemm_kernel[grid](
        x,
        w,
        c,
        M,
        N,
        K,
        BLOCK_SIZE_K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        SPLIT_COUNT,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c


# @triton.autotune(configs=fp32_gemm_configs, key=["M", "N", "K"])
@triton.jit
def split_fp32_gemm_for_backward_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_COUNT)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + pid_k * K // SPLIT_COUNT + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = (
        b_ptr + pid_k * K // SPLIT_COUNT * N + offs_n[None, :] + offs_k[:, None] * N
    )

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(k):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs).to(tl.float32)
        c = tl.dot(a, b, c)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    if SPLIT_COUNT == 1:
        tl.store(c_ptrs, c)
    else:
        tl.atomic_add(c_ptrs, c, sem="relaxed")


def triton_split_fp32_gemm_for_backward(y: torch.Tensor, w: torch.Tensor):
    """
    mix precision gemm for backward, a@b.float()
    Args:
        a: input gradient, fp32
        b: gemm weight, bf16/fp16
    Returns:
        c: gradient of activation
    """
    assert y.is_contiguous() and w.is_contiguous()
    M, K = y.size()
    K, N = w.size()
    BLOCK_SIZE_K = max([x for x in [16, 32, 64, 128] if K % x == 0])
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128
    assert M % BLOCK_SIZE_M == 0 and N % BLOCK_SIZE_N == 0
    SPLIT_COUNT = min(triton.cdiv(K, 2048), 8)
    if SPLIT_COUNT == 1:
        c = torch.empty((M, N), dtype=w.dtype, device=w.device)
    else:
        c = torch.zeros((M, N), dtype=torch.float32, device=w.device)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
        SPLIT_COUNT,
    )  # noqa

    num_warps = 4
    num_stages = 2
    split_fp32_gemm_for_backward_kernel[grid](
        y,
        w,
        c,
        M,
        N,
        K,
        BLOCK_SIZE_K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        SPLIT_COUNT,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if SPLIT_COUNT > 1:
        c = c.to(w.dtype)
    return c


# @triton.autotune(configs=fp32_gemm_configs, key=["M", "N", "K"])
@triton.jit
def split_fp32_gemm_for_update_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_COUNT)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (
        a_ptr + pid_k * K // SPLIT_COUNT * M + offs_m[None, :] + offs_k[:, None] * M
    )
    b_ptrs = (
        b_ptr + pid_k * K // SPLIT_COUNT * N + offs_n[None, :] + offs_k[:, None] * N
    )

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.trans(tl.load(a_ptrs)).to(tl.float32)
        b = tl.load(b_ptrs).to(tl.float32)
        c = tl.dot(a, b, c)
        a_ptrs += BLOCK_SIZE_K * M
        b_ptrs += BLOCK_SIZE_K * N

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    if SPLIT_COUNT == 1:
        tl.store(c_ptrs, c)
    else:
        tl.atomic_add(c_ptrs, c, sem="relaxed")


def triton_split_fp32_gemm_for_update(y: torch.Tensor, x: torch.Tensor):
    """
    mix precision gemm for updaing weight
    Args:
        y: gradient of output, fp32
        x: input activation, bf16/fp16
    Returns:
        c: gradient of weight
    """
    assert y.is_contiguous() and x.is_contiguous()
    K, M = y.size()
    K, N = x.size()
    BLOCK_SIZE_K = 64
    BLOCK_SIZE_M = max([x for x in [16, 32, 64] if M % x == 0])
    BLOCK_SIZE_N = 128
    SPLIT_COUNT = min(triton.cdiv(K, 2048), 8)
    if SPLIT_COUNT == 1:
        c = torch.empty((M, N), dtype=torch.float32, device=x.device)
    else:
        c = torch.zeros((M, N), dtype=torch.float32, device=x.device)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
        SPLIT_COUNT,
    )  # noqa

    num_warps = 2
    num_stages = 3
    split_fp32_gemm_for_update_kernel[grid](
        y,
        x,
        c,
        M,
        N,
        K,
        BLOCK_SIZE_K,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        SPLIT_COUNT,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c
