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
        BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        SPLIT_COUNT: tl.constexpr
        ):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)

    k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_COUNT)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + pid_k * K // SPLIT_COUNT + offs_m[:, None] * K + offs_k[
                                                                      None, :]
    b_ptrs = b_ptr + pid_k * K // SPLIT_COUNT + offs_n[None, :] * K + offs_k[:,
                                                                      None]

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M)  # .to(tl.float32)
        b = tl.load(b_ptrs)  # .to(tl.float32)
        c = tl.dot(a, b, c)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    if SPLIT_COUNT == 1:
        tl.store(c_ptrs, c, mask=offs_m[:, None] < M)
    else:
        tl.atomic_add(c_ptrs, c, sem='relaxed', mask=offs_m[:, None] < M)


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

    if M >= 2048:
        BLOCK_SIZE_M = 128
        SPLIT_COUNT = 1
    else:
        BLOCK_SIZE_M = 32
        SPLIT_COUNT = min(triton.cdiv(K, 2048), 4)

    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 128
    assert N % BLOCK_SIZE_N == 0
    assert K % (BLOCK_SIZE_K * SPLIT_COUNT) == 0

    if SPLIT_COUNT == 1:
        c = torch.empty(M, N, dtype=torch.float32, device=x.device)
    else:
        c = torch.zeros(M, N, dtype=torch.float32, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),
                         triton.cdiv(N, META["BLOCK_SIZE_N"]),
                         SPLIT_COUNT)  # noqa
    num_warps = 4
    num_stages = 3
    split_fp32_gemm_kernel[grid](x, w, c,
                                 M, N, K,
                                 BLOCK_SIZE_K,
                                 BLOCK_SIZE_M,
                                 BLOCK_SIZE_N,
                                 SPLIT_COUNT,
                                 num_warps=num_warps,
                                 num_stages=num_stages
                                 )
    return c
