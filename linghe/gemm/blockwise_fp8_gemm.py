# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl

# adapt from deepseek
# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


@triton.jit
def fp8_gemm_bb_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
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
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    # b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    b_ptrs = b_ptr + offs_n[:, None] * K + offs_k[None, :]
    nb = K // BLOCK_SIZE_K

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, k):
        a_s = tl.load(a_s_ptr + pid_m * nb + i)
        b_s = tl.load(b_s_ptr + pid_n * nb + i)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, tl.trans(b)) * (a_s * b_s)
        # accumulator = tl.dot(a, tl.trans(b), accumulator)
        # accumulator += (accumulators-accumulator) * scale
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


# use for hadamard quantization, too slow on H800
def triton_bb_fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    out_dtype=torch.bfloat16,
    block_size=128,
):
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = torch.empty(M, N, dtype=out_dtype, device=a.device)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )  # noqa

    fp8_gemm_bb_kernel[grid](
        a,
        b,
        c,
        a_s,
        b_s,
        M,
        N,
        K,
        BLOCK_SIZE_K=block_size,
        BLOCK_SIZE_M=block_size,
        BLOCK_SIZE_N=block_size,
        num_warps=8,
        num_stages=4,
    )
    return c


# fp8_gemm_configs = [
#     Config({"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n},
#            num_stages=num_stages, num_warps=8)
#     for block_m in [32, 64, 128]
#     for block_n in [32, 64, 128]
#     for num_stages in [3, 4, 5, 6]
# ]


# @triton.autotune(configs=fp8_gemm_configs, key=["N", "K"])
@triton.jit
def fp8_gemm_tt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # a and b all tilewise quantization.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + offs_n * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1

    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


def triton_tt_fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    out_dtype=torch.bfloat16,
    block_size=128,
):
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = torch.empty(*a.size()[:-1], N, dtype=out_dtype, device=a.device)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )  # noqa
    fp8_gemm_tt_kernel[grid](
        a,
        b,
        c,
        a_s,
        b_s,
        M,
        N,
        K,
        BLOCK_SIZE_K=block_size,
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
    )
    return c
