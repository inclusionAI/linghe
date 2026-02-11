# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import itertools

import torch
import triton
import triton.language as tl
from triton import Config

from linghe.tools.util import round_up

# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


@triton.jit
def transpose_kernel(
    x_ptr, t_ptr, M, N, H: tl.constexpr, W: tl.constexpr, EVEN: tl.constexpr
):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    offs = (
        rid * H * N + cid * W + tl.arange(0, H)[:, None] * N + tl.arange(0, W)[None, :]
    )
    toffs = (
        rid * H + cid * M * W + tl.arange(0, W)[:, None] * M + tl.arange(0, H)[None, :]
    )
    if EVEN:
        y = tl.trans(tl.load(x_ptr + offs))
        tl.store(t_ptr + toffs, y)
    else:
        y = tl.trans(
            tl.load(
                x_ptr + offs,
                mask=(cid * W + tl.arange(0, W)[None, :] < N)
                & (rid * H + tl.arange(0, H)[:, None] < M),
            )
        )
        tl.store(
            t_ptr + toffs,
            y,
            mask=(cid * W + tl.arange(0, W)[:, None] < N)
            & (rid * H + tl.arange(0, H)[None, :] < M),
        )


@triton.jit
def transpose_inner_dims_kernel(
    x_ptr, t_ptr, B, M, b_stride, m_stride, N: tl.constexpr
):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    offs = rid * b_stride + cid * m_stride + tl.arange(0, N)
    toffs = cid * B * N + rid * N + tl.arange(0, N)
    y = tl.load(x_ptr + offs)
    tl.store(t_ptr + toffs, y)


@triton.jit
def transpose_outer_dims_kernel(
    x_ptr, t_ptr, M, N, H: tl.constexpr, W: tl.constexpr, EVEN: tl.constexpr
):
    bid = tl.program_id(axis=0)
    rid = tl.program_id(axis=1)
    cid = tl.program_id(axis=2)
    offs = (
        bid * M * N
        + rid * H * N
        + cid * W
        + tl.arange(0, H)[:, None] * N
        + tl.arange(0, W)[None, :]
    )
    toffs = (
        bid * M * N
        + rid * H
        + cid * M * W
        + tl.arange(0, W)[:, None] * M
        + tl.arange(0, H)[None, :]
    )
    if EVEN:
        y = tl.trans(tl.load(x_ptr + offs))
        tl.store(t_ptr + toffs, y)
    else:
        y = tl.trans(
            tl.load(
                x_ptr + offs,
                mask=(cid * W + tl.arange(0, W)[None, :] < N)
                & (rid * H + tl.arange(0, H)[:, None] < M),
            )
        )
        tl.store(
            t_ptr + toffs,
            y,
            mask=(cid * W + tl.arange(0, W)[:, None] < N)
            & (rid * H + tl.arange(0, H)[None, :] < M),
        )


def triton_transpose(x: torch.Tensor, inner=True):
    """
    transpose x with dim0 and dim1
    Args:
        x: input tensor
        inner: inner dim if True, outer dim if False

    Returns:
        transposed tensor
    """
    shape = x.shape
    rank = len(shape)
    assert rank <= 4
    if rank == 2:
        M, N = shape
        device = x.device
        t = torch.empty((N, M), device=device, dtype=x.dtype)
        H = 64
        W = 32 if x.dtype.itemsize == 1 else 16
        EVEN = M % H == 0 and N % W == 0
        num_stages = 5
        num_warps = 2

        grid = (triton.cdiv(M, H), triton.cdiv(N, W))
        transpose_kernel[grid](
            x, t, M, N, H, W, EVEN, num_stages=num_stages, num_warps=num_warps
        )
    elif inner:
        stride = x.stride()
        if rank == 4:
            B, M, N = shape[0], shape[1], shape[2] * shape[3]
            assert stride[2] == shape[3], "must be contiguous in last two dims"
            t = torch.empty((M, B, shape[2], shape[3]), device=x.device, dtype=x.dtype)
        else:
            B, M, N = shape
            t = torch.empty((M, B, N), device=x.device, dtype=x.dtype)
        b_stride = stride[0]
        m_stride = stride[1]
        num_stages = 5
        num_warps = 2
        grid = (B, M)
        transpose_inner_dims_kernel[grid](
            x,
            t,
            B,
            M,
            b_stride,
            m_stride,
            N,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    else:

        if rank == 4:
            B, M, N = shape[0] * shape[1], shape[2], shape[3]
            t = torch.empty((shape[0], shape[1], N, M), device=x.device, dtype=x.dtype)
        else:
            B, M, N = shape
            t = torch.empty((B, N, M), device=x.device, dtype=x.dtype)

        H = 64
        W = 32 if x.dtype.itemsize == 1 else 16
        EVEN = M % H == 0 and N % W == 0
        num_stages = 5
        num_warps = 2

        grid = (B, triton.cdiv(M, H), triton.cdiv(N, W))
        transpose_outer_dims_kernel[grid](
            x, t, M, N, H, W, EVEN, num_stages=num_stages, num_warps=num_warps
        )
    return t


@triton.jit
def transpose_and_pad_kernel(
    x_ptr, t_ptr, M, N, P, H: tl.constexpr, W: tl.constexpr, EVEN: tl.constexpr
):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    offs = (
        rid * H * N + cid * W + tl.arange(0, H)[:, None] * N + tl.arange(0, W)[None, :]
    )
    toffs = (
        rid * H + cid * P * W + tl.arange(0, W)[:, None] * P + tl.arange(0, H)[None, :]
    )
    if EVEN:
        y = tl.load(x_ptr + offs)
    else:
        y = tl.load(x_ptr + offs, mask=(rid * H + tl.arange(0, H)[:, None] < M))
    y = tl.trans(y)
    if EVEN:
        tl.store(t_ptr + toffs, y)
    else:
        # paddings are filled with 0
        tl.store(t_ptr + toffs, y, mask=(rid * H + tl.arange(0, H)[None, :] < P))


def triton_transpose_and_pad(x, out=None, pad=True):
    """
    transpose x and padding the column size to be mutiplier of 32,
    it is used for calculated gradient of weight with torch._scaled__mm
    Args:
        x: input tensor
        out:
        pad: whether need padding

    Returns:
        out: output tensor
    """
    # fat block, shape:[H,W]
    assert x.is_contiguous()
    M, N = x.shape
    P = round_up(M, b=32) if pad else M
    device = x.device
    if out is None:
        out = torch.empty((N, P), device=device, dtype=x.dtype)

    H = 32
    W = 64
    num_stages = 5
    num_warps = 2
    assert N % W == 0
    EVEN = M % H == 0 and M == P
    grid = (triton.cdiv(P, H), triton.cdiv(N, W))
    transpose_and_pad_kernel[grid](
        x, out, M, N, P, H, W, EVEN, num_stages=num_stages, num_warps=num_warps
    )
    return out


@triton.jit
def batch_transpose_kernel(xs_ptr, xts_ptr, M, N, H: tl.constexpr, W: tl.constexpr):
    eid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    x_ptr = tl.load(xs_ptr + eid).to(tl.pointer_type(xts_ptr.dtype.element_ty))
    offs = cid * W + tl.arange(0, H)[:, None] * N + tl.arange(0, W)[None, :]
    toffs = (
        eid * M * N
        + cid * W * M
        + tl.arange(0, W)[:, None] * M
        + tl.arange(0, H)[None, :]
    )
    for i in range(0, M, H):
        y = tl.trans(tl.load(x_ptr + offs))
        tl.store(xts_ptr + toffs, y)
        offs += N * H
        toffs += H


def triton_batch_transpose(xs, xts=None):
    """
    batch transpose x
    Args:
        xs: input tensor list, [M, N]*expert
    Returns:
        xts: output tensor list, [N,M]*expert
    """
    assert all([x.is_contiguous() for x in xs])
    M, N = xs[0].shape
    n_experts = len(xs)
    device = xs[0].device
    if xts is None:
        xts = torch.empty((M * n_experts, N), device=device, dtype=xs[0].dtype)
    pointers = torch.tensor([x.data_ptr() for x in xs], dtype=torch.int64).cuda(
        device, non_blocking=True
    )

    H = 32
    W = 64
    num_stages = 2
    num_warps = 8
    grid = (n_experts, N // W)
    batch_transpose_kernel[grid](
        pointers, xts, M, N, H, W, num_stages=num_stages, num_warps=num_warps
    )
    outputs = torch.split(xts, [M] * n_experts)
    return outputs


@triton.jit
def batch_transpose_and_pad_kernel(
    x_ptr,
    t_ptr,
    count_ptr,
    accum_ptr,
    pad_accum_ptr,
    N,
    H: tl.constexpr,
    W: tl.constexpr,
):
    eid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    count = tl.load(count_ptr + eid)
    ei = tl.load(accum_ptr + eid)
    si = ei - count
    pad_si = tl.load(pad_accum_ptr + eid)
    P = tl.cdiv(count, 32) * 32
    offs = si * N + cid * W + tl.arange(0, H)[:, None] * N + tl.arange(0, W)[None, :]
    toffs = (
        pad_si * N
        + cid * W * P
        + tl.arange(0, W)[:, None] * P
        + tl.arange(0, H)[None, :]
    )
    for i in range(0, P, H):
        y = tl.trans(tl.load(x_ptr + offs, mask=i + tl.arange(0, H)[:, None] < count))
        # paddings are filled with 0
        tl.store(t_ptr + toffs, y, mask=i + tl.arange(0, H)[None, :] < P)
        offs += N * H
        toffs += H


def triton_batch_transpose_and_pad(x, count_list, x_t=None, pad=True):
    """
    transpose and pad each tensor stored in x
    Args:
        x: [sum(bs), N]
        count_list: a python list of token count
        pad: whether pad to mutiplier of 32,
            padding value should be filled with 0 if padded

    Returns:
        x_t: output tensor
    """
    assert x.is_contiguous()
    assert pad
    # block shape:[H,W]
    M, N = x.shape
    n_experts = len(count_list)
    # NOTE: b must be 32, or kernel will miscalculated padding size with original size
    pad_sizes = [round_up(x, b=32) for x in count_list]
    counts = torch.tensor(count_list, dtype=torch.int32, device=x.device)
    pad_accum_sizes = torch.tensor(
        list(itertools.accumulate(pad_sizes, initial=0)),
        dtype=torch.int32,
        device=x.device,
    )
    accums = torch.cumsum(counts, 0)
    device = x.device
    if x_t is None:
        x_t = torch.empty((sum(pad_sizes) * N), device=device, dtype=x.dtype)
    H = 256
    W = 64
    num_stages = 2
    num_warps = 8
    grid = (n_experts, N // W)
    batch_transpose_and_pad_kernel[grid](
        x,
        x_t,
        counts,
        accums,
        pad_accum_sizes,
        N,
        H,
        W,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    split_size = [x * N for x in pad_sizes]
    chunks = torch.split(x_t.view(torch.uint8), split_size)
    outputs = []
    for i, p in enumerate(pad_sizes):
        outputs.append(chunks[i].view(torch.float8_e4m3fn).view(N, p))
    return outputs


configs = [
    Config({"H": H, "W": W}, num_stages=num_stages, num_warps=num_warps)
    for H in [32, 64, 128, 256, 512]
    for W in [16, 32, 64]
    for num_stages in [3, 4, 5, 6]
    for num_warps in [2, 4, 8]
]


@triton.autotune(configs=configs, key=["M", "N", "D"])
@triton.jit
def opt_transpose_kernel(x_ptr, t_ptr, M, N, D, H: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, col-wise write
    offs = pid * W + tl.arange(0, H)[:, None] * N + tl.arange(0, W)[None, :]
    toffs = pid * W * M + tl.arange(0, W)[:, None] * M + tl.arange(0, H)[None, :]
    m = tl.cdiv(M, H)
    for i in range(m):
        y = tl.trans(tl.load(x_ptr + offs))
        tl.store(t_ptr + toffs, y)
        offs += H * N
        toffs += H


def triton_opt_transpose(x):
    assert x.is_contiguous()
    M, N = x.shape
    device = x.device
    D = 0 if x.dtype.itemsize == 1 else 1
    t = torch.empty((N, M), device=device, dtype=x.dtype)
    grid = lambda META: (N // META["W"],)  # noqa
    opt_transpose_kernel[grid](x, t, M, N, D)
    return t
