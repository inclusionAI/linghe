"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import Optional
import torch
import triton
import triton.language as tl


@triton.jit
def fp8_grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    as_ptr,
    bs_ptr,
    c_ptr,
    token_ids_ptr,
    expert_ids_ptr,
    token_count_ptr,
    padding_value,
    topk,
    as_stride,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    TRANSPOSE_A_SCALE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    token_count = tl.load(token_count_ptr)
    eid = tl.load(expert_ids_ptr + pid_m)
    if pid_m * BLOCK_SIZE_M >= token_count:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    token_ids = tl.load(token_ids_ptr + offs_m)
    tids = token_ids // topk
    mask = token_ids < padding_value

    k = K // BLOCK_SIZE_K
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + tids[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + eid * N * K + offs_n[None, :] * K + offs_k[:, None]

    sk = K // 128
    if TRANSPOSE_A_SCALE:
        as_ptrs = as_ptr + tids
    else:
        as_ptrs = as_ptr + tids * sk
    bs_ptrs = bs_ptr + eid * N // 128 * sk + pid_n * BLOCK_SIZE_N // 128 * sk

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs + i * BLOCK_SIZE_K, mask=mask[:, None])
        b = tl.load(b_ptrs + i * BLOCK_SIZE_K)
        if TRANSPOSE_A_SCALE:
            a_s = tl.load(as_ptrs + i * BLOCK_SIZE_K // 128 * as_stride, mask=mask)
        else:
            a_s = tl.load(as_ptrs + i * BLOCK_SIZE_K // 128, mask=mask)
        b_s = tl.load(bs_ptrs + i * BLOCK_SIZE_K // 128)
        scale = a_s[:, None] * b_s
        accumulator += tl.dot(a, b) * scale
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + token_ids[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, accumulator, mask=mask[:, None])


def triton_fp8_grouped_gemm(
    xq: torch.Tensor,
    wq: torch.Tensor,
    xs: torch.Tensor,
    ws: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    token_count: torch.Tensor,
    c: Optional[torch.Tensor] = None,
    block_size_m: int = 16,
    block_size_n: int = 128,
    padding_value: int = 9,
    topk: int = 9,
):
    """
    grouped gemm with fp8
    Returns:
        c: output with fp32 precision
    """
    assert xq.is_contiguous()
    assert wq.is_contiguous() and ws.is_contiguous()

    M = expert_ids.numel()
    N, K = wq.shape[1:]
    device = xq.device
    if c is None:
        c = torch.empty(M, N, dtype=torch.bfloat16, device=device)
    else:
        assert c.is_contiguous()

    TRANSPOSE_A_SCALE = not xs.is_contiguous()
    xs_stride = xs.stride(1)
    BLOCK_SIZE_K = 128  # only support BLOCK_SIZE_K <= 128
    num_warps = 4
    num_stages = 5

    if block_size_m == 16:
        if M <= 16 * 64 and N <= 512:
            block_size_n = 16
            num_warps = 2
            num_stages = 3
        else:
            block_size_n = 64
            num_warps = 4
            num_stages = 3
    elif block_size_m == 32:
        block_size_n = 64

    grid = (triton.cdiv(M, block_size_m), triton.cdiv(N, block_size_n))
    fp8_grouped_gemm_kernel[grid](
        xq,
        wq,
        xs,
        ws,
        c,
        token_ids,
        expert_ids,
        token_count,
        padding_value,
        topk,
        xs_stride,
        M,
        N,
        K,
        BLOCK_SIZE_K,
        block_size_m,
        block_size_n,
        TRANSPOSE_A_SCALE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c
