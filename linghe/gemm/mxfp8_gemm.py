# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import Optional, List

import torch
import triton
import triton.language as tl

from linghe.utils.transpose import triton_transpose


# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


# fp8_gemm_configs = [
#     Config({"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n},
#            num_stages=num_stages, num_warps=8)
#     for block_m in [32, 64, 128]
#     for block_n in [32, 64, 128]
#     for num_stages in [3, 4, 5, 6]
# ]

# @triton.autotune(configs=fp8_gemm_configs, key=["N", "K"])


@triton.jit
def mxfp8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACCUM: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = K // 32
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, 32)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

    a_s_ptrs = a_s_ptr + offs_m
    b_s_ptrs = b_s_ptr + offs_n

    if ACCUM:
        accumulator = tl.load(c_ptr + offs_m[:, None] * N + offs_n[None, :])
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_s = tl.exp2(tl.load(a_s_ptrs).to(tl.float32) - 127.0)
        b_s = tl.exp2(tl.load(b_s_ptrs).to(tl.float32) - 127.0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += 32
        b_ptrs += 32

        a_s_ptrs += M
        b_s_ptrs += N

    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


def triton_mxfp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    out: Optional[torch.tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    layout: str = "TN",
    accumulate: bool = False,
):
    """
    triton implementation to simulate mxfp8 grouped gemm
    layout is defined as the same in TE:
        TN: forward
        NN: bakcward
        NT: update(wgrad)
    layout is used to optimize BLOCK SIZE
    """
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()

    if layout == "TN":
        assert not accumulate
        a_s = a_s.t().contiguous()
        b_s = b_s.t().contiguous()
    elif layout == "NN":
        assert not accumulate
        b = triton_transpose(b)
        a_s = a_s.t().contiguous()
    else:
        a = triton_transpose(a)
        b = triton_transpose(b)

    M, K = a.shape
    N = b.size(0)

    if out is not None:
        assert out.is_contiguous()
    else:
        out = torch.empty(M, N, dtype=out_dtype, device=a.device)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    grid = (M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)
    mxfp8_gemm_kernel[grid](
        a,
        b,
        out,
        a_s,
        b_s,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        ACCUM=accumulate,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.jit
def mxfp8_gemm_forward_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = K // 32
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, 32)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

    a_s_ptrs = a_s_ptr + offs_m
    b_s_ptrs = b_s_ptr + offs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_s = tl.exp2(tl.load(a_s_ptrs).to(tl.float32) - 127.0)
        b_s = tl.exp2(tl.load(b_s_ptrs).to(tl.float32) - 127.0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += 32
        b_ptrs += 32

        a_s_ptrs += M
        b_s_ptrs += N

    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


def triton_mxfp8_gemm_forward(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    out: Optional[torch.tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
):
    """
    triton implementation to simulate mxfp8 gemm
    """
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()

    M, K = a.shape
    N = b.size(0)

    a_s = a_s.t().contiguous()
    b_s = b_s.t().contiguous()

    if out is not None:
        assert out.is_contiguous()
    else:
        out = torch.empty(M, N, dtype=out_dtype, device=a.device)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    grid = (M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)
    mxfp8_gemm_forward_kernel[grid](
        a,
        b,
        out,
        a_s,
        b_s,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.jit
def mxfp8_gemm_backward_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = K // 32
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, 32)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] + offs_k[:, None] * N
    a_s_ptrs = a_s_ptr + offs_m
    b_s_ptrs = b_s_ptr + offs_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_s = tl.exp2(tl.load(a_s_ptrs).to(tl.float32) - 127.0)
        b_s = tl.exp2(tl.load(b_s_ptrs).to(tl.float32) - 127.0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += 32
        b_ptrs += 32 * N
        a_s_ptrs += M
        b_s_ptrs += N

    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


def triton_mxfp8_gemm_backward(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    out: Optional[torch.tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
):
    """
    triton implementation to simulate mxfp8 gemm
    """
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()

    M, K = a.shape
    N = b.size(1)

    a_s = a_s.t().contiguous()

    if out is not None:
        assert out.is_contiguous()
    else:
        out = torch.empty(M, N, dtype=out_dtype, device=a.device)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    grid = (M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)
    mxfp8_gemm_backward_kernel[grid](
        a,
        b,
        out,
        a_s,
        b_s,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.jit
def mxfp8_gemm_update_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACCUM: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = K // 32
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, 32)
    a_ptrs = a_ptr + offs_m[None, :] + offs_k[:, None] * M
    # a_ptrs = a_ptr + offs_m[:, None] + offs_k[None, :] * M
    b_ptrs = b_ptr + offs_n[None, :] + offs_k[:, None] * N
    a_s_ptrs = a_s_ptr + offs_m
    b_s_ptrs = b_s_ptr + offs_n

    if ACCUM:
        c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        accumulator = tl.load(c_ptrs)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_s = tl.exp2(tl.load(a_s_ptrs).to(tl.float32) - 127.0)
        b_s = tl.exp2(tl.load(b_s_ptrs).to(tl.float32) - 127.0)
        accumulator += tl.dot(tl.trans(a), b) * a_s[:, None] * b_s[None, :]
        # accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += 32 * M
        b_ptrs += 32 * N
        a_s_ptrs += M
        b_s_ptrs += N

    c = accumulator.to(c_ptr.dtype.element_ty)
    # offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


def triton_mxfp8_gemm_update(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    out: Optional[torch.tensor] = None,
    out_dtype: torch.dtype = torch.float32,
    accumulate: bool = False,
):
    """
    triton implementation to simulate mxfp8 gemm
    """
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()

    K, M = a.shape
    N = b.size(1)

    if out is not None:
        assert out.is_contiguous()
    else:
        assert not accumulate
        out = torch.empty(M, N, dtype=out_dtype, device=a.device)
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    grid = (M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)  # noqa
    mxfp8_gemm_update_kernel[grid](
        a,
        b,
        out,
        a_s,
        b_s,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        ACCUM=accumulate,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.jit
def mxfp8_grouped_gemm_kernel(
    a_ptrs,
    b_ptrs,
    c_ptr,
    a_s_ptrs,
    b_s_ptrs,
    size_ptr,
    accum_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACCUM: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    pid_e = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    size = tl.load(size_ptr + pid_e)

    a_ptr = tl.load(a_ptrs + pid_e).to(tl.pointer_type(tl.float8e4nv))
    b_ptr = tl.load(b_ptrs + pid_e).to(tl.pointer_type(tl.float8e4nv))
    a_s_ptr = tl.load(a_s_ptrs + pid_e).to(tl.pointer_type(tl.uint8))
    b_s_ptr = tl.load(b_s_ptrs + pid_e).to(tl.pointer_type(tl.uint8))

    if LAYOUT != "NT":
        if pid_m * BLOCK_SIZE_M >= size:
            return

    if LAYOUT == "NT":
        K = size
        k = size // 32
        c_ptr = tl.load(c_ptr + pid_e).to(tl.pointer_type(tl.float32))
        MS = 0
    else:
        k = K // 32
        MS = tl.load(accum_ptr + pid_e) - size
        M = tl.cdiv(size, 128) * 128

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, 32)
    a_ps = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ps = b_ptr + offs_n[None, :] * K + offs_k[:, None]

    a_s_ps = a_s_ptr + offs_m
    b_s_ps = b_s_ptr + offs_n

    if ACCUM:
        accumulator = tl.load(c_ptr + offs_m[:, None] * N + offs_n[None, :])
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(k):
        a = tl.load(a_ps)
        b = tl.load(b_ps)
        a_s = tl.exp2(tl.load(a_s_ps).to(tl.float32) - 127.0)
        b_s = tl.exp2(tl.load(b_s_ps).to(tl.float32) - 127.0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ps += 32
        b_ps += 32
        a_s_ps += M
        b_s_ps += N

    c_ps = c_ptr + MS * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ps, accumulator, cache_modifier=".cs")


def triton_mxfp8_grouped_gemm(
    a: List[torch.Tensor],
    b: List[torch.Tensor],
    a_s: List[torch.Tensor],
    b_s: List[torch.Tensor],
    m_splits: List[int],
    out: Optional[torch.tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    layout: str = "TN",
    accumulate: bool = False,
):
    """
    triton implementation to simulate mxfp8 grouped gemm
    """
    device = a[0].device
    sizes = torch.tensor(m_splits, dtype=torch.int64).cuda(device, non_blocking=True)
    accums = torch.cumsum(sizes, 0)
    ms = sum(m_splits)

    if layout == "TN":
        assert not accumulate
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 128
        M = max(m_splits)
        K = a[0].size(1)
        N = b[0].size(0)
        # not work if in one line
        a_s = [x.t().contiguous() for x in a_s]
        b_s = [x.t().contiguous() for x in b_s]
        if out is None:
            out = torch.empty(ms, N, dtype=out_dtype, device=device)

    elif layout == "NN":
        assert not accumulate
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 128
        M = max(m_splits)
        K = a[0].size(1)
        N = b[0].size(1)
        b = [triton_transpose(x) for x in b]
        a_s = [x.t().contiguous() for x in a_s]
        if out is None:
            out = torch.empty(ms, N, dtype=out_dtype, device=device)
    else:
        assert out is not None
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        M = a[0].size(1)
        N = b[0].size(1)
        K = 0
        a = [triton_transpose(x) for x in a]
        b = [triton_transpose(x) for x in b]
        out_ptrs = torch.tensor([x.data_ptr() for x in out], dtype=torch.int64).cuda(
            device, non_blocking=True
        )

    as_ptrs = torch.tensor([x.data_ptr() for x in a_s], dtype=torch.int64).cuda(
        device, non_blocking=True
    )
    bs_ptrs = torch.tensor([x.data_ptr() for x in b_s], dtype=torch.int64).cuda(
        device, non_blocking=True
    )

    a_ptrs = torch.tensor([x.data_ptr() for x in a], dtype=torch.int64).cuda(
        device, non_blocking=True
    )
    b_ptrs = torch.tensor([x.data_ptr() for x in b], dtype=torch.int64).cuda(
        device, non_blocking=True
    )

    grid = (len(m_splits), M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)  # noqa
    mxfp8_grouped_gemm_kernel[grid](
        a_ptrs,
        b_ptrs,
        out_ptrs if layout == "NT" else out,
        as_ptrs,
        bs_ptrs,
        sizes,
        accums,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        ACCUM=accumulate,
        LAYOUT=layout,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.jit
def mxfp8_grouped_gemm_forward_kernel(
    a_ptrs,
    b_ptrs,
    c_ptr,
    a_s_ptrs,
    b_s_ptrs,
    size_ptr,
    accum_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_e = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    size = tl.load(size_ptr + pid_e)

    a_ptr = tl.load(a_ptrs + pid_e).to(tl.pointer_type(tl.float8e4nv))
    b_ptr = tl.load(b_ptrs + pid_e).to(tl.pointer_type(tl.float8e4nv))
    a_s_ptr = tl.load(a_s_ptrs + pid_e).to(tl.pointer_type(tl.uint8))
    b_s_ptr = tl.load(b_s_ptrs + pid_e).to(tl.pointer_type(tl.uint8))

    if pid_m * BLOCK_SIZE_M >= size:
        return

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, 32)
    a_ps = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ps = b_ptr + offs_n[None, :] * K + offs_k[:, None]

    a_s_ps = a_s_ptr + offs_m
    b_s_ps = b_s_ptr + offs_n
    M = tl.cdiv(size, 128) * 128

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    k = K // 32
    for i in range(k):
        a = tl.load(a_ps)
        b = tl.load(b_ps)
        a_s = tl.exp2(tl.load(a_s_ps).to(tl.float32) - 127.0)
        b_s = tl.exp2(tl.load(b_s_ps).to(tl.float32) - 127.0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ps += 32
        b_ps += 32
        a_s_ps += M
        b_s_ps += N

    MS = tl.load(accum_ptr + pid_e) - size
    c_ps = c_ptr + MS * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ps, accumulator, cache_modifier=".cs")


def triton_mxfp8_grouped_gemm_forward(
    a: List[torch.Tensor],
    b: List[torch.Tensor],
    a_s: List[torch.Tensor],
    b_s: List[torch.Tensor],
    m_splits: List[int],
    out: Optional[torch.tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
):
    """
    triton implementation to simulate mxfp8 grouped gemm
    """
    device = a[0].device
    sizes = torch.tensor(m_splits, dtype=torch.int64).cuda(device, non_blocking=True)
    accums = torch.cumsum(sizes, 0)
    a_ptrs = torch.tensor([x.data_ptr() for x in a], dtype=torch.int64).cuda(
        device, non_blocking=True
    )
    b_ptrs = torch.tensor([x.data_ptr() for x in b], dtype=torch.int64).cuda(
        device, non_blocking=True
    )

    # not work if in one line
    at_s = [x.t().contiguous() for x in a_s]
    as_ptrs = torch.tensor([x.data_ptr() for x in at_s], dtype=torch.int64).cuda(
        device, non_blocking=True
    )
    bt_s = [x.t().contiguous() for x in b_s]
    bs_ptrs = torch.tensor([x.data_ptr() for x in bt_s], dtype=torch.int64).cuda(
        device, non_blocking=True
    )

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128
    N = b[0].size(0)
    K = a[0].size(1)
    if out is None:
        out = torch.empty(sum(m_splits), N, dtype=out_dtype, device=device)

    grid = (len(m_splits), max(m_splits) // BLOCK_SIZE_M, N // BLOCK_SIZE_N)  # noqa
    mxfp8_grouped_gemm_forward_kernel[grid](
        a_ptrs,
        b_ptrs,
        out,
        as_ptrs,
        bs_ptrs,
        sizes,
        accums,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.jit
def mxfp8_grouped_gemm_backward_kernel(
    a_ptrs,
    b_ptrs,
    c_ptr,
    a_s_ptrs,
    b_s_ptrs,
    size_ptr,
    accum_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_e = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    size = tl.load(size_ptr + pid_e)

    a_ptr = tl.load(a_ptrs + pid_e).to(tl.pointer_type(tl.float8e4nv))
    b_ptr = tl.load(b_ptrs + pid_e).to(tl.pointer_type(tl.float8e4nv))
    a_s_ptr = tl.load(a_s_ptrs + pid_e).to(tl.pointer_type(tl.uint8))
    b_s_ptr = tl.load(b_s_ptrs + pid_e).to(tl.pointer_type(tl.uint8))

    if pid_m * BLOCK_SIZE_M >= size:
        return

    MS = tl.load(accum_ptr + pid_e) - size
    k = K // 32

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, 32)
    a_ps = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ps = b_ptr + offs_n[None, :] + offs_k[:, None] * N

    a_s_ps = a_s_ptr + offs_m
    b_s_ps = b_s_ptr + offs_n
    M = tl.cdiv(size, 128) * 128

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(k):
        a = tl.load(a_ps)
        b = tl.load(b_ps)
        a_s = tl.exp2(tl.load(a_s_ps).to(tl.float32) - 127.0)
        b_s = tl.exp2(tl.load(b_s_ps).to(tl.float32) - 127.0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ps += 32
        b_ps += 32 * N
        a_s_ps += M
        b_s_ps += N

    # offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ps = c_ptr + MS * N + offs_m[:, None] * N + offs_n[None, :]
    # tl.store(c_ps, c)
    tl.store(c_ps, accumulator, cache_modifier=".cs")


def triton_mxfp8_grouped_gemm_backward(
    a: List[torch.Tensor],
    b: List[torch.Tensor],
    a_s: List[torch.Tensor],
    b_s: List[torch.Tensor],
    m_splits: List[int],
    out: Optional[torch.tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
):
    """
    triton implementation to simulate mxfp8 grouped gemm
    layout is defined as the same in TE:
        TN: forward
        NN: bakcward
        NT: update(wgrad)
    """
    device = a[0].device
    sizes = torch.tensor(m_splits, dtype=torch.int64).cuda(device, non_blocking=True)
    accums = torch.cumsum(sizes, 0)
    a_ptrs = torch.tensor([x.data_ptr() for x in a], dtype=torch.int64).cuda(
        device, non_blocking=True
    )
    b_ptrs = torch.tensor([x.data_ptr() for x in b], dtype=torch.int64).cuda(
        device, non_blocking=True
    )

    # not work if in one line
    at_s = [x.t().contiguous() for x in a_s]
    as_ptrs = torch.tensor([x.data_ptr() for x in at_s], dtype=torch.int64).cuda(
        device, non_blocking=True
    )
    bs_ptrs = torch.tensor([x.data_ptr() for x in b_s], dtype=torch.int64).cuda(
        device, non_blocking=True
    )

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 128
    K = a[0].size(1)
    N = b[0].size(1)
    if out is None:
        out = torch.empty(sum(m_splits), N, dtype=out_dtype, device=device)

    grid = (len(m_splits), max(m_splits) // BLOCK_SIZE_M, N // BLOCK_SIZE_N)  # noqa
    mxfp8_grouped_gemm_backward_kernel[grid](
        a_ptrs,
        b_ptrs,
        out,
        as_ptrs,
        bs_ptrs,
        sizes,
        accums,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.jit
def mxfp8_grouped_gemm_update_kernel(
    a_ptrs,
    b_ptrs,
    c_ptr,
    a_s_ptrs,
    b_s_ptrs,
    size_ptr,
    accum_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACCUM: tl.constexpr,
):
    pid_e = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    size = tl.load(size_ptr + pid_e)

    a_ptr = tl.load(a_ptrs + pid_e).to(tl.pointer_type(tl.float8e4nv))
    b_ptr = tl.load(b_ptrs + pid_e).to(tl.pointer_type(tl.float8e4nv))
    a_s_ptr = tl.load(a_s_ptrs + pid_e).to(tl.pointer_type(tl.uint8))
    b_s_ptr = tl.load(b_s_ptrs + pid_e).to(tl.pointer_type(tl.uint8))

    c_ptr = tl.load(c_ptr + pid_e).to(tl.pointer_type(tl.float32))
    K = size
    k = tl.cdiv(K, 128) * 4

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, 32)
    a_ps = a_ptr + offs_m[None, :] + offs_k[:, None] * M
    b_ps = b_ptr + offs_n[None, :] + offs_k[:, None] * N
    a_s_ps = a_s_ptr + offs_m
    b_s_ps = b_s_ptr + offs_n

    if ACCUM:
        c_ps = c_ptr + offs_m[:, None] * N + offs_n[None, :]
        accumulator = tl.load(c_ps)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(k):
        a = tl.load(a_ps)
        b = tl.load(b_ps)
        a_s = tl.exp2(tl.load(a_s_ps).to(tl.float32) - 127.0)
        b_s = tl.exp2(tl.load(b_s_ps).to(tl.float32) - 127.0)
        accumulator += tl.dot(tl.trans(a), b) * a_s[:, None] * b_s[None, :]
        a_ps += 32 * M
        b_ps += 32 * N
        a_s_ps += M
        b_s_ps += N

    # offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ps = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    # tl.store(c_ps, c)
    tl.store(c_ps, accumulator, cache_modifier=".cs")


def triton_mxfp8_grouped_gemm_update(
    a: List[torch.Tensor],
    b: List[torch.Tensor],
    a_s: List[torch.Tensor],
    b_s: List[torch.Tensor],
    m_splits: List[int],
    out: Optional[List[torch.tensor]],
    out_dtype: torch.dtype = torch.float32,
    accumulate: bool = False,
):
    """
    triton implementation to simulate mxfp8 grouped gemm
    layout is defined as the same in TE:
        TN: forward
        NN: bakcward
        NT: update(wgrad)
    layout is used to optimize BLOCK SIZE
    """
    device = a[0].device
    sizes = torch.tensor(m_splits, dtype=torch.int64).cuda(device, non_blocking=True)
    accums = torch.cumsum(sizes, 0)
    a_ptrs = torch.tensor([x.data_ptr() for x in a], dtype=torch.int64).cuda(
        device, non_blocking=True
    )
    b_ptrs = torch.tensor([x.data_ptr() for x in b], dtype=torch.int64).cuda(
        device, non_blocking=True
    )

    as_ptrs = torch.tensor([x.data_ptr() for x in a_s], dtype=torch.int64).cuda(
        device, non_blocking=True
    )
    bs_ptrs = torch.tensor([x.data_ptr() for x in b_s], dtype=torch.int64).cuda(
        device, non_blocking=True
    )

    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    M = a[0].size(1)
    N = b[0].size(1)
    out_ptrs = torch.tensor([x.data_ptr() for x in out], dtype=torch.int64).cuda(
        device, non_blocking=True
    )

    grid = (len(m_splits), M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)  # noqa
    mxfp8_grouped_gemm_update_kernel[grid](
        a_ptrs,
        b_ptrs,
        out_ptrs,
        as_ptrs,
        bs_ptrs,
        sizes,
        accums,
        M,
        N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        ACCUM=accumulate,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.jit
def native_mxfp8_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_s_ptr,
    b_s_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ACCUM: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = K // 32
    PM = tl.cdiv(M, 128) * 128
    PN = tl.cdiv(N, 128) * 128

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, 32)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]

    a_s_ptrs = a_s_ptr + offs_m
    b_s_ptrs = b_s_ptr + offs_n

    if ACCUM:
        accumulator = tl.load(c_ptr + offs_m[:, None] * N + offs_n[None, :])
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_s = tl.exp2(tl.load(a_s_ptrs).to(tl.float32) - 127.0)
        b_s = tl.exp2(tl.load(b_s_ptrs).to(tl.float32) - 127.0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += 32
        b_ptrs += 32

        a_s_ptrs += PM
        b_s_ptrs += PN

    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


def triton_native_mxfp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_s: torch.Tensor,
    b_s: torch.Tensor,
    out: Optional[torch.tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    layout: str = "TN",
    accumulate: bool = False,
):
    """
    triton implementation to simulate mxfp8 grouped gemm
    layout is defined as the same in TE:
        TN: forward
        NN: bakcward
        NT: update(wgrad)
    layout is used to optimize BLOCK SIZE
    """
    assert a.is_contiguous() and b.is_contiguous()
    assert a_s.is_contiguous() and b_s.is_contiguous()

    if layout == "TN":
        assert not accumulate
        M, K = a.shape
        N = b.size(0)
        BLOCK_SIZE_M = max([x for x in [32, 64] if M % x == 0])
        BLOCK_SIZE_N = 128
        a_s = a_s.t().contiguous()
        b_s = b_s.t().contiguous()
    elif layout == "NN":
        assert not accumulate
        M, K = a.shape
        N = b.size(1)
        BLOCK_SIZE_M = max([x for x in [32, 64] if M % x == 0])
        BLOCK_SIZE_N = 128
        b = triton_transpose(b)
        a_s = a_s.t().contiguous()
    else:
        K, M = a.shape
        N = b.size(1)
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        a = triton_transpose(a)
        b = triton_transpose(b)

    if out is not None:
        assert out.is_contiguous()
    else:
        out = torch.empty(M, N, dtype=out_dtype, device=a.device)
    grid = (M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)
    native_mxfp8_gemm_kernel[grid](
        a,
        b,
        out,
        a_s,
        b_s,
        M,
        N,
        K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        ACCUM=accumulate,
        num_warps=4,
        num_stages=3,
    )
    return out


def triton_native_mxfp8_grouped_gemm(
    a: List[torch.Tensor],
    b: List[torch.Tensor],
    a_s: List[torch.Tensor],
    b_s: List[torch.Tensor],
    out: List[torch.Tensor],
    m_splits: List[int],
    layout: str = "TN",
):
    device = a[0].device
    dtype = a[0].dtype
    if layout != "NT":
        outs = torch.split(out, m_splits)
    else:
        outs = out
    for i, m in enumerate(m_splits):
        triton_native_mxfp8_gemm(a[i], b[i], a_s[i], b_s[i], out=outs[i], layout=layout)
    return out
