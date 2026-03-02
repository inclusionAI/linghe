"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl



# fp32_gemm_configs = [
#     triton.Config({"BLOCK_SIZE_M": block_m,
#                    "BLOCK_SIZE_N": block_n}, 
#                    num_stages=3, 
#                    num_warps=num_warps)
#     for block_m in [16, 32, 64, 128]
#     for block_n in [32, 64, 128]
#     for num_warps in [2, 4]
# ]
# @triton.autotune(configs=fp32_gemm_configs, key=["M", "N", "K", "SPLIT_COUNT"])
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
        BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    pid_k = tl.program_id(axis=0)

    k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_COUNT)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = (a_ptr
              + pid_k * K // SPLIT_COUNT
              + offs_m[:, None] * K
              + offs_k[None, :])
    b_ptrs = (b_ptr
              + pid_k * K // SPLIT_COUNT
              + offs_n[None, :] * K
              + offs_k[:, None])

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
        tl.atomic_add(c_ptrs, c, sem='relaxed', mask=offs_m[:, None] < M)


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

    if M * N >= 128 * 128 * 128:
        SPLIT_COUNT = 1
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 128
    elif M * N >= 32 * 128 * 128:
        SPLIT_COUNT = 1
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 128
    elif M * N >= 16 * 64 * 128:
        SPLIT_COUNT = 1
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 64
    else:
        SPLIT_COUNT = 4
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 32

    BLOCK_SIZE_K = 128
    
    if SPLIT_COUNT == 1:
        c = torch.empty(M, N, dtype=torch.float32, device=x.device)
    else:
        c = torch.zeros(M, N, dtype=torch.float32, device=x.device)

    grid = lambda META: (META["SPLIT_COUNT"],
                         triton.cdiv(M, META["BLOCK_SIZE_M"]),
                         triton.cdiv(N, META["BLOCK_SIZE_N"]),
                         )  # noqa
    num_warps = 4
    num_stages = 2
    split_fp32_gemm_kernel[grid](x, w, c,
                                 M, N, K,
                                 SPLIT_COUNT,
                                 BLOCK_SIZE_K,
                                 BLOCK_SIZE_M,
                                 BLOCK_SIZE_N,
                                 num_warps=num_warps,
                                 num_stages=num_stages
                                 )
    return c
