import torch
import triton
import triton.language as tl
from triton import Config

fp8_gemm_configs = [
    Config({"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n},
           num_stages=num_stages, num_warps=8)
    for block_m in [32, 64, 128]
    for block_n in [32, 64, 128]
    for num_stages in [3, 4, 5, 6]
]


# @triton.autotune(configs=fp8_gemm_configs, key=["N", "K"])
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
    # a blockwise quantization, b blockwise quantization.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    # b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    b_ptrs = b_ptr + offs_n[:, None] * K + offs_k[None, :]

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, k):
        a_s = tl.load(a_s_ptr + pid_m * (K // BLOCK_SIZE_K) + i)
        b_s = tl.load(b_s_ptr + pid_n * (K // BLOCK_SIZE_K) + i)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K,
                    other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K,
                    other=0.0)
        accumulator += tl.dot(a, tl.trans(b)) * a_s * b_s
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


@triton.autotune(configs=fp8_gemm_configs, key=["N", "K"])
@triton.jit
def fp8_gemm_tb_kernel(
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
    # a tilewise quantization, b blockwise quantization.
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        # a = tl.load(a_ptrs)
        # b = tl.load(b_ptrs)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K,
                    other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K,
                    other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        # accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        accumulators = tl.dot(a, b, accumulator)
        accumulator += (accumulators - accumulator) * a_s[:, None] * b_s[None,
                                                                     :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    # tl.store(c_ptrs, c)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)


@triton.autotune(configs=fp8_gemm_configs, key=["N", "K"])
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
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K,
                    other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K,
                    other=0.0)
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


def fp8_gemm(a: torch.Tensor, b: torch.Tensor, a_s: torch.Tensor,
             b_s: torch.Tensor, out_dtype=torch.bfloat16):
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=out_dtype)
    a_tile_scale = a_s.size(0) == a.size(0)
    b_tile_scale = b_s.size(0) == b.size(0)
    block_size = K // a_s.size(-1)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),  # noqa: E731
                         triton.cdiv(N, META["BLOCK_SIZE_N"]))  # noqa: E731
    if a_tile_scale and b_tile_scale:
        fp8_gemm_tt_kernel[grid](a, b, c, a_s, b_s, M, N, K, block_size)
    elif a_tile_scale and not b_tile_scale:
        fp8_gemm_tb_kernel[grid](a, b, c, a_s, b_s, M, N, K, block_size)
    elif not a_tile_scale and not b_tile_scale:
        fp8_gemm_bb_kernel[grid](a, b, c, a_s, b_s, M, N, K,
                                 BLOCK_SIZE_K=block_size,
                                 BLOCK_SIZE_M=block_size,
                                 BLOCK_SIZE_N=block_size, num_warps=8,
                                 num_stages=4)
    return c
