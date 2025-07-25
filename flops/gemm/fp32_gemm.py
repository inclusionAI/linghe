from typing import Optional

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
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs).to(tl.float32)
        b = tl.load(b_ptrs).to(tl.float32)
        # c += tl.dot(a, b)
        c = tl.dot(a, b, c)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


# a, bf16
# b, bf16
# c, fp32
def triton_fp32_gemm(a: torch.Tensor, b: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.size()
    N, K = b.size()
    assert N >= 128
    c = torch.empty(M, N, dtype=torch.float32, device=a.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),
                         triton.cdiv(N, META["BLOCK_SIZE_N"]))  # noqa
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128
    num_warps = 4
    num_stages = 3
    fp32_gemm_kernel[grid](a, b, c,
                           M, N, K,
                           BLOCK_SIZE_K,
                           BLOCK_SIZE_M,
                           BLOCK_SIZE_N,
                           num_warps=num_warps,
                           num_stages=num_stages
                           )
    return c


@triton.jit
def scaled_fp32_gemm_kernel(
        a_ptr,
        b_ptr,
        scale_ptr,
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
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs).to(tl.float32)
        b = tl.load(b_ptrs).to(tl.float32)
        # c += tl.dot(a, b)
        c = tl.dot(a, b, c)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K

    scale = tl.load(
        scale_ptr + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    c *= scale[:, None]

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


def triton_scaled_fp32_gemm(a: torch.Tensor, b: torch.Tensor,
                            scale: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.size()
    N, K = b.size()
    c = torch.empty(M, N, dtype=torch.float32, device=a.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),
                         triton.cdiv(N, META["BLOCK_SIZE_N"]))  # noqa
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128
    num_warps = 4
    num_stages = 3
    scaled_fp32_gemm_kernel[grid](a, b, scale, c,
                                  M, N, K,
                                  BLOCK_SIZE_K,
                                  BLOCK_SIZE_M,
                                  BLOCK_SIZE_N,
                                  num_warps=num_warps,
                                  num_stages=num_stages
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
        ACCUM: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] + offs_k[:, None] * N

    if ACCUM:
        c = tl.load(c_ptr + offs_m[:, None] * N + offs_n[None, :]).to(
            tl.float32)
    else:
        c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(k):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs).to(tl.float32)
        # c += tl.dot(a, b)
        c = tl.dot(a, b, c)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K * N
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


# a: router output, fp32
# b: router weight, bf16, should be transposed before calculation
# c: dy of rms, bf16, shoule be accumlated
def triton_fp32_gemm_for_backward(a: torch.Tensor, b: torch.Tensor,
                                  c: Optional[torch.Tensor] = None,
                                  accum=False):
    assert a.is_contiguous() and b.is_contiguous()
    M, K = a.size()
    K, N = b.size()
    if c is None:
        c = torch.empty((M, N), dtype=b.dtype, device=b.device)
        accum = False
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),
                         triton.cdiv(N, META["BLOCK_SIZE_N"]))  # noqa
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128
    num_warps = 4
    num_stages = 2
    fp32_gemm_for_backward_kernel[grid](a, b, c,
                                        M, N, K, accum,
                                        BLOCK_SIZE_K,
                                        BLOCK_SIZE_M,
                                        BLOCK_SIZE_N,
                                        num_warps=num_warps,
                                        num_stages=num_stages
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
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[None, :] + offs_k[:, None] * M
    b_ptrs = b_ptr + offs_n[None, :] + offs_k[:, None] * N

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # c = tl.load(c_ptr + offs_m[:, None] * N + offs_n[None, :]).to(tl.float32)
    for i in range(k):
        a = tl.trans(tl.load(a_ptrs)).to(tl.float32)
        b = tl.load(b_ptrs).to(tl.float32)
        # c += tl.dot(a, b)
        c = tl.dot(a, b, c)
        a_ptrs += BLOCK_SIZE_K * M
        b_ptrs += BLOCK_SIZE_K * N

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


# a: router output, fp32, should be transposed before calculation
# b: input of rms, bf16, should be transposed before calculation
def triton_fp32_gemm_for_update(a: torch.Tensor, b: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous()
    K, M = a.size()
    K, N = b.size()
    c = torch.empty((M, N), dtype=b.dtype, device=b.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),
                         triton.cdiv(N, META["BLOCK_SIZE_N"]))  # noqa
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128
    num_warps = 4
    num_stages = 3
    fp32_gemm_for_update_kernel[grid](a, b, c,
                                      M, N, K,
                                      BLOCK_SIZE_K,
                                      BLOCK_SIZE_M,
                                      BLOCK_SIZE_N,
                                      num_warps=num_warps,
                                      num_stages=num_stages
                                      )
    return c


@triton.jit
def scaled_fp32_gemm_for_update_kernel(
        a_ptr,
        b_ptr,
        scale_ptr,
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
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[None, :] + offs_k[:, None] * M
    b_ptrs = b_ptr + offs_n[None, :] + offs_k[:, None] * N

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # c = tl.load(c_ptr + offs_m[:, None] * N + offs_n[None, :]).to(tl.float32)
    for i in range(k):
        scale = tl.load(
            scale_ptr + i * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
        a = tl.trans(tl.load(a_ptrs)).to(tl.float32) * scale[None, :]
        b = tl.load(b_ptrs).to(tl.float32)
        # c += tl.dot(a, b)
        c = tl.dot(a, b, c)
        a_ptrs += BLOCK_SIZE_K * M
        b_ptrs += BLOCK_SIZE_K * N

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


# a: router output, fp32, should be transposed before calculation
# b: input of rms, bf16, should be transposed before calculation
# scale: 1/rms
def triton_scaled_fp32_gemm_for_update(a: torch.Tensor, b: torch.Tensor,
                                       scale: torch.Tensor):
    assert a.is_contiguous() and b.is_contiguous()
    K, M = a.size()
    K, N = b.size()
    c = torch.empty((M, N), dtype=b.dtype, device=b.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),
                         triton.cdiv(N, META["BLOCK_SIZE_N"]))  # noqa
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128
    num_warps = 4
    num_stages = 3
    scaled_fp32_gemm_for_update_kernel[grid](a, b, scale, c,
                                             M, N, K,
                                             BLOCK_SIZE_K,
                                             BLOCK_SIZE_M,
                                             BLOCK_SIZE_N,
                                             num_warps=num_warps,
                                             num_stages=num_stages
                                             )
    return c
