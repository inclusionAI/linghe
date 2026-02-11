# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl

from linghe.tools.util import round_up
from linghe.utils.transpose import triton_transpose_and_pad


@triton.jit
def tokenwise_smooth_quant_kernel(
    x_ptr,
    q_ptr,
    ss_ptr,
    qs_ptr,
    max_ptr,
    M,
    T,
    N: tl.constexpr,
    W: tl.constexpr,
    EVEN: tl.constexpr,
    REVERSE: tl.constexpr,
    ROUND: tl.constexpr,
    CALIBRATE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    smooth_scale = tl.load(ss_ptr + tl.arange(0, N))[None, :]
    if not REVERSE:
        smooth_scale = 1.0 / smooth_scale

    if CALIBRATE:
        output_maxs = tl.zeros((W, N), dtype=tl.float32)
    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        if EVEN:
            x = tl.load(
                x_ptr
                + pid * W * T * N
                + i * N * W
                + tl.arange(0, W)[:, None] * N
                + tl.arange(0, N)[None, :]
            ).to(tl.float32)
        else:
            x = tl.load(
                x_ptr
                + pid * W * T * N
                + i * N * W
                + tl.arange(0, W)[:, None] * N
                + tl.arange(0, N)[None, :],
                mask=indices[:, None] < M,
            ).to(tl.float32)
        if CALIBRATE:
            output_maxs = tl.maximum(tl.abs(x), output_maxs)
        x *= smooth_scale
        x_max = tl.max(tl.abs(x), axis=1)
        scale = tl.maximum(x_max / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        if EVEN:
            tl.store(
                qs_ptr + pid * W * T + i * W + tl.arange(0, W),
                scale,
            )
        else:
            tl.store(
                qs_ptr + pid * W * T + i * W + tl.arange(0, W), scale, mask=indices < M
            )

        x /= scale[:, None]
        xq = x.to(q_ptr.dtype.element_ty)
        if EVEN:
            tl.store(
                q_ptr
                + pid * W * T * N
                + i * N * W
                + tl.arange(0, W)[:, None] * N
                + tl.arange(0, N)[None, :],
                xq,
            )
        else:
            tl.store(
                q_ptr
                + pid * W * T * N
                + i * N * W
                + tl.arange(0, W)[:, None] * N
                + tl.arange(0, N)[None, :],
                xq,
                mask=indices[:, None] < M,
            )
    if CALIBRATE:
        output_maxs = tl.max(output_maxs, 0)
        tl.store(max_ptr + pid * N + tl.arange(0, N), output_maxs)


@triton.jit
def blockwise_smooth_quant_kernel(
    x_ptr,
    q_ptr,
    ss_ptr,
    qs_ptr,
    max_ptr,
    M,
    N,
    H: tl.constexpr,
    W: tl.constexpr,
    EVEN: tl.constexpr,
    REVERSE: tl.constexpr,
    ROUND: tl.constexpr,
    CALIBRATE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    offs = pid * W * N + tl.arange(0, W)[:, None] * N + tl.arange(0, H)[None, :]
    soffs = tl.arange(0, H)
    x_max = tl.zeros((W,), dtype=tl.float32)
    n = tl.cdiv(N, H)
    for i in range(n):
        smooth_scale = tl.load(ss_ptr + soffs)
        if EVEN:
            x = tl.load(x_ptr + offs).to(tl.float32)
        else:
            x = tl.load(x_ptr + offs, mask=pid * W + tl.arange(0, W)[:, None] < M).to(
                tl.float32
            )
        if CALIBRATE:
            output_maxs = tl.max(x.abs(), 0)
            tl.store(max_ptr + pid * N + i * H + tl.arange(0, H), output_maxs)
        if REVERSE:
            x = x * smooth_scale
        else:
            x = x / smooth_scale
        x_max = tl.maximum(tl.max(tl.abs(x), axis=1), x_max)
        offs += H
        soffs += H

    scale = tl.maximum(x_max / 448, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    tl.store(
        qs_ptr + pid * W + tl.arange(0, W), scale, mask=pid * W + tl.arange(0, W) < M
    )

    s = (1.0 / scale)[:, None]

    offs = pid * W * N + tl.arange(0, W)[:, None] * N + tl.arange(0, H)[None, :]
    soffs = tl.arange(0, H)
    for i in range(n):
        smooth_scale = tl.load(ss_ptr + soffs)
        if EVEN:
            x = tl.load(x_ptr + offs)
        else:
            x = tl.load(x_ptr + offs, mask=pid * W + tl.arange(0, W)[:, None] < M)

        if REVERSE:
            xq = (x.to(tl.float32) * smooth_scale * s).to(q_ptr.dtype.element_ty)
        else:
            xq = (x.to(tl.float32) / smooth_scale * s).to(q_ptr.dtype.element_ty)

        if EVEN:
            tl.store(q_ptr + offs, xq)
        else:
            # tl.store(q_ptr+offs, xq, mask=(i*H+tl.arange(0, H)[None,:]<N)&(pid*W+tl.arange(0, W)[:,None]<M))
            tl.store(q_ptr + offs, xq, mask=pid * W + tl.arange(0, W)[:, None] < M)
        offs += H
        soffs += H


def triton_smooth_quant(
    x,
    smooth_scale,
    x_q=None,
    x_scale=None,
    reverse=False,
    round_scale=False,
    calibrate=False,
):
    """"""
    M, N = x.shape
    device = x.device
    if x_q is None:
        x_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    if x_scale is None:
        x_scale = torch.empty((M,), device=device, dtype=torch.float32)
    if triton.next_power_of_2(N) == N and N <= 8192:
        W = 8192 // N
        T = 8
        # it may used in shard weight quantization, therefore M is not batch size
        # assert M % (W * T) == 0
        EVEN = M % (W * T) == 0
        g = triton.cdiv(M, W * T)
        if calibrate:
            x_maxs = torch.empty((g, N), device=device, dtype=torch.bfloat16)
        else:
            x_maxs = None
        tokenwise_smooth_quant_kernel[(g,)](
            x,
            x_q,
            smooth_scale,
            x_scale,
            x_maxs,
            M,
            T,
            N,
            W,
            EVEN,
            reverse,
            round_scale,
            calibrate,
            num_stages=3,
            num_warps=4,
        )
        if calibrate:
            x_maxs = x_maxs.amax(0).float()
    else:
        H = max([x for x in [256, 512, 1024, 2048] if N % x == 0])
        W = 8 if M > 8192 else 4
        EVEN = M % W == 0
        T = triton.cdiv(M, W)
        if calibrate:
            x_maxs = torch.empty((T, N), device=device, dtype=torch.bfloat16)
        else:
            x_maxs = None
        grid = (T,)
        blockwise_smooth_quant_kernel[grid](
            x,
            x_q,
            smooth_scale,
            x_scale,
            x_maxs,
            M,
            N,
            H,
            W,
            EVEN,
            reverse,
            round_scale,
            calibrate,
            num_stages=3,
            num_warps=4,
        )
        if calibrate:
            x_maxs = x_maxs.amax(0).float()

    return x_q, x_scale, x_maxs


@triton.jit
def subrow_smooth_quant_kernel(
    x_ptr,
    q_ptr,
    ss_ptr,
    qs_ptr,
    subrow_scales_ptr,
    tail_ri,
    tail_si,
    head_ri,
    head_ei,
    size,
    N,
    W: tl.constexpr,
    TAIL: tl.constexpr,
    HEAD: tl.constexpr,
    REVERSE: tl.constexpr,
    ROUND: tl.constexpr,
):
    if TAIL:
        # scale is saved as max/448
        scale = tl.maximum(tl.load(subrow_scales_ptr), 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        # scale only stores in subrow with leading values

        T = tl.cdiv(N - tail_si, W)
        for i in range(T):
            mask = tail_si + i * W + tl.arange(0, W) < N
            if REVERSE:
                smooth_scale = tl.load(
                    ss_ptr + tail_si + i * W + tl.arange(0, W), mask=mask
                )
            else:
                smooth_scale = tl.load(
                    ss_ptr + tail_si + i * W + tl.arange(0, W), other=1e30, mask=mask
                )
                smooth_scale = 1.0 / smooth_scale
            x = tl.load(x_ptr + i * W + tl.arange(0, W), mask=mask).to(tl.float32)
            x *= smooth_scale
            x /= scale
            xq = tl.minimum(tl.maximum(x, -448), 448)
            tl.store(
                q_ptr + tail_ri * N + tail_si + i * W + tl.arange(0, W),
                xq.to(q_ptr.dtype.element_ty),
                mask=mask,
            )

    if HEAD:
        # scale is saved as max/448
        scale = tl.maximum(tl.load(subrow_scales_ptr + 1), 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(qs_ptr + head_ri, scale)

        T = tl.cdiv(head_ei, W)
        for i in range(T):
            mask = i * W + tl.arange(0, W) < head_ei
            if REVERSE:
                smooth_scale = tl.load(ss_ptr + i * W + tl.arange(0, W), mask=mask)
            else:
                smooth_scale = tl.load(
                    ss_ptr + i * W + tl.arange(0, W), other=1e30, mask=mask
                )
                smooth_scale = 1.0 / smooth_scale
            x = tl.load(x_ptr + size - head_ei + i * W + tl.arange(0, W), mask=mask).to(
                tl.float32
            )
            x *= smooth_scale
            x /= scale
            xq = tl.minimum(tl.maximum(x, -448), 448)
            tl.store(
                q_ptr + head_ri * N + i * W + tl.arange(0, W),
                xq.to(q_ptr.dtype.element_ty),
                mask=mask,
            )


def triton_subrow_smooth_quant(
    x,
    smooth_scale,
    x_q,
    x_scale,
    subrow_scales,
    offset,
    size,
    reverse=False,
    round_scale=False,
):
    """"""
    M, N = x_q.shape
    W = 128
    if offset % N == 0:
        tail_ri = 0
        tail_si = 0
        TAIL = False
    else:
        tail_ri = offset // N
        tail_si = offset % N
        TAIL = True

    if (offset + size) % N == 0:
        head_ri = 0
        head_ei = 0  # head_size = head_ei
        HEAD = False
    else:
        head_ri = (offset + size) // N
        head_ei = (offset + size) % N
        HEAD = True

    grid = (1,)
    subrow_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scale,
        x_scale,
        subrow_scales,
        tail_ri,
        tail_si,
        head_ri,
        head_ei,
        size,
        N,
        W,
        TAIL,
        HEAD,
        reverse,
        round_scale,
        num_stages=3,
        num_warps=1,
    )


@triton.jit
def depracated_tokenwise_smooth_quant_kernel(
    x_ptr,
    q_ptr,
    ss_ptr,
    qs_ptr,
    M,
    W,
    N: tl.constexpr,
    REVERSE: tl.constexpr,
    ROUND: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    smooth_scale = tl.load(ss_ptr + tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0 / smooth_scale

    for i in range(W):
        x = tl.load(
            x_ptr + pid * W * N + i * N + tl.arange(0, N), mask=pid * W + i < M
        ).to(tl.float32)
        x *= smooth_scale
        x_max = tl.maximum(tl.max(tl.abs(x)), 1e-30)

        scale = x_max / 448.0
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(qs_ptr + pid * W + i, scale, mask=pid * W + i < M)

        x /= scale
        xq = x.to(q_ptr.dtype.element_ty)
        tl.store(
            q_ptr + pid * W * N + i * N + tl.arange(0, N), xq, mask=pid * W + i < M
        )


def triton_depracated_tokenwise_smooth_quant(
    x, smooth_scale, x_q=None, x_scale=None, reverse=False, round_scale=False
):
    """"""
    # row-wise read, row-wise write
    M, N = x.shape
    device = x.device
    if x_q is None:
        x_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    if x_scale is None:
        x_scale = torch.empty((M,), device=device, dtype=torch.float32)
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    W = triton.cdiv(M, sm)
    grid = (sm,)
    depracated_tokenwise_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scale,
        x_scale,
        M,
        W,
        N,
        reverse,
        round_scale,
        num_stages=3,
        num_warps=8,
    )
    return x_q, x_scale


@triton.jit
def batch_smooth_quant_kernel(
    x_ptr,
    q_ptr,
    ss_ptr,
    qs_ptr,
    xm_ptr,
    count_ptr,
    accum_ptr,
    T,
    N: tl.constexpr,
    REVERSE: tl.constexpr,
    ROUND: tl.constexpr,
    CALIBRATE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    i_expert = pid // T
    i_batch = pid % T

    # row-wise read, row-wise write
    smooth_scale = tl.load(ss_ptr + i_expert * N + tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0 / smooth_scale

    if CALIBRATE:
        x_maxs = tl.zeros((N,), dtype=tl.float32)

    count = tl.load(count_ptr + i_expert)
    ei = tl.load(accum_ptr + i_expert)
    si = ei - count

    n = tl.cdiv(count, T)  # samples for each task
    for i in range(i_batch * n, min((i_batch + 1) * n, count)):
        x = tl.load(x_ptr + si * N + i * N + tl.arange(0, N)).to(tl.float32)
        if CALIBRATE:
            x_maxs = tl.maximum(x_maxs, x.abs())
        x *= smooth_scale
        scale = tl.maximum(tl.max(tl.abs(x)) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))

        tl.store(qs_ptr + si + i, scale)

        s = 1.0 / scale
        x *= s
        xq = x.to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + si * N + i * N + tl.arange(0, N), xq)

    if CALIBRATE:
        tl.store(xm_ptr + pid * N + tl.arange(0, N), x_maxs)


"""
select and smooth and quant
x: [bs, dim]
smooth_scales: [n_experts, dim]
token_count_per_expert: [n_experts]
x_q: [bs, dim]
x_scale: [bs]
"""


def triton_batch_smooth_quant(
    x,
    smooth_scales,
    token_count_per_expert,
    x_q=None,
    x_scale=None,
    x_maxs=None,
    reverse=False,
    round_scale=False,
    calibrate=False,
):
    """"""
    M, N = x.shape
    device = x.device
    n_expert = token_count_per_expert.shape[0]
    assert 128 % n_expert == 0
    if x_q is None:
        x_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    if x_scale is None:
        x_scale = torch.empty((M,), device=device, dtype=torch.float32)
    accum_token_count = torch.cumsum(token_count_per_expert, 0)
    T = 128 // n_expert
    if calibrate and x_maxs is None:
        x_maxs = torch.empty((128, N), device=device, dtype=torch.float32)

    grid = (128,)
    batch_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scales,
        x_scale,
        x_maxs,
        token_count_per_expert,
        accum_token_count,
        T,
        N,
        reverse,
        round_scale,
        calibrate,
        num_stages=3,
        num_warps=8,
    )
    if calibrate:
        x_maxs = x_maxs.view(n_expert, T, N).amax(1)
    return x_q, x_scale, x_maxs


@triton.jit
def batch_pad_transpose_smooth_quant_kernel(
    x_ptr,
    q_ptr,
    ss_ptr,
    qs_ptr,
    count_ptr,
    accum_ptr,
    N,
    H: tl.constexpr,
    W: tl.constexpr,
    E: tl.constexpr,
    REVERSE: tl.constexpr,
    ROUND: tl.constexpr,
):
    eid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)

    count = tl.load(count_ptr + eid)
    ei = tl.load(accum_ptr + eid)
    si = ei - count
    round_count = tl.cdiv(count, 32) * 32

    counts = tl.load(count_ptr + tl.arange(0, E))
    n_blocks = tl.cdiv(counts, 128)
    bias = tl.sum(tl.where(tl.arange(0, E) < eid, n_blocks, 0))

    n = tl.cdiv(count, H)
    maxs = tl.zeros((H, W), dtype=tl.float32)
    for i in range(n):
        # col-wise read, row-wise write
        indices = i * H + tl.arange(0, H)
        smooth_scale = tl.load(ss_ptr + indices, mask=indices < count)
        if not REVERSE:
            smooth_scale = 1.0 / smooth_scale

        x = tl.load(
            x_ptr
            + si * N
            + i * H * N
            + bid * W
            + tl.arange(0, H)[:, None]
            + tl.arange(0, W)[None, :],
            mask=indices[:, None] < count,
        ).to(tl.float32)
        x *= smooth_scale[:, None]
        maxs = tl.maximum(maxs, tl.abs(x))

    maxs = tl.max(maxs, 0)
    scale = tl.maximum(tl.max(maxs, 0) / 448.0, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(qs_ptr + eid * N + bid * W + tl.arange(0, W), scale)
    s = 1.0 / scale

    for i in range(n):
        # col-wise read, row-wise write
        indices = i * H + tl.arange(0, H)
        smooth_scale = tl.load(ss_ptr + indices, mask=indices < count)
        if not REVERSE:
            smooth_scale = 1.0 / smooth_scale

        x = tl.load(
            x_ptr
            + si * N
            + i * H * N
            + bid * W
            + tl.arange(0, H)[:, None]
            + tl.arange(0, W)[None, :],
            mask=indices[:, None] < count,
        ).to(tl.float32)
        x *= smooth_scale[:, None]
        x *= s
        xq = tl.trans(x.to(q_ptr.dtype.element_ty))
        tl.store(
            q_ptr
            + bias * N
            + bid * W * round_count
            + i * H
            + tl.arange(0, W)[:, None]
            + tl.arange(0, H)[None, :],
            xq,
            mask=indices[None, :] < round_count,
        )


"""
used in silu backward
pad to multiple of 32 and transpose and smooth quant
x: [sum(token_per_expert), dim]
smooth_scales: [sum(token_per_expert)]
token_count_per_expert: [n_experts]
splits: list of token_count_per_expert
x_q: [sum(roundup(token_per_expert)) * dim]
x_scale: [n_experts, dim]
"""


def triton_batch_pad_transpose_smooth_quant(
    x,
    smooth_scales,
    token_count_per_expert,
    splits,
    x_q=None,
    x_scale=None,
    x_maxs=None,
    reverse=False,
    round_scale=False,
):
    """"""
    M, N = x.shape
    device = x.device
    n_expert = token_count_per_expert.shape[0]
    round_splits = [(x + 31) // 32 * 32 for x in splits]
    round_size = sum(round_splits)
    if x_q is None:
        x_q = torch.empty((round_size, N), device=device, dtype=torch.float8_e4m3fn)
    if x_scale is None:
        x_scale = torch.empty((n_expert, N), device=device, dtype=torch.float32)
    accum_token_count = torch.cumsum(token_count_per_expert, 0)
    H = 128
    W = 32
    grid = (n_expert, N // W)
    batch_pad_transpose_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scales,
        x_scale,
        token_count_per_expert,
        accum_token_count,
        N,
        H,
        W,
        n_expert,
        reverse,
        round_scale,
        num_stages=3,
        num_warps=8,
    )
    return x_q, x_scale


@triton.jit
def transpose_smooth_quant_kernel(
    x_ptr,
    q_ptr,
    ss_ptr,
    qs_ptr,
    M,
    N,
    P,
    H: tl.constexpr,
    W: tl.constexpr,
    EVEN: tl.constexpr,
    REVERSE: tl.constexpr,
    ROUND: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # col-wise read, row-wise write
    offs = pid * W + tl.arange(0, H)[:, None] * N + tl.arange(0, W)[None, :]
    soffs = tl.arange(0, H)
    x_max = tl.zeros((W,), dtype=tl.float32)
    m = tl.cdiv(P, H)
    for i in range(m):
        if EVEN:
            x = tl.load(x_ptr + offs)
            smooth_scale = tl.load(ss_ptr + soffs)[:, None]
        else:
            x = tl.load(
                x_ptr + offs,
                mask=(i * H + tl.arange(0, H)[:, None] < M)
                & (pid * W + tl.arange(0, W)[None, :] < N),
            )
            other = 0.0 if REVERSE else 1e30
            smooth_scale = tl.load(ss_ptr + soffs, mask=soffs < M, other=other)[:, None]
        if REVERSE:
            x = x * smooth_scale
        else:
            x = x / smooth_scale
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0), x_max)
        offs += H * N
        soffs += H

    scale = tl.maximum(x_max / 448.0, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    if EVEN:
        tl.store(qs_ptr + pid * W + tl.arange(0, W), scale)
    else:
        tl.store(
            qs_ptr + pid * W + tl.arange(0, W),
            scale,
            mask=pid * W + tl.arange(0, W) < N,
        )

    s = (1.0 / scale)[None, :]
    offs = pid * W + tl.arange(0, H)[:, None] * N + tl.arange(0, W)[None, :]
    soffs = tl.arange(0, H)
    toffs = pid * W * P + tl.arange(0, W)[:, None] * P + tl.arange(0, H)[None, :]
    for i in range(m):
        if EVEN:
            x = tl.load(x_ptr + offs).to(tl.float32)
            smooth_scale = tl.load(ss_ptr + soffs)[:, None]
        else:
            x = tl.load(x_ptr + offs, mask=(i * H + tl.arange(0, H)[:, None] < M)).to(
                tl.float32
            )
            other = 0.0 if REVERSE else 1e30
            smooth_scale = tl.load(ss_ptr + soffs, mask=soffs < M, other=other)[:, None]

        if REVERSE:
            x = (x * smooth_scale * s).to(q_ptr.dtype.element_ty)
        else:
            x = (x / smooth_scale * s).to(q_ptr.dtype.element_ty)
        if EVEN:
            tl.store(q_ptr + toffs, tl.trans(x))
        else:
            # mask with P instead of M
            tl.store(
                q_ptr + toffs, tl.trans(x), mask=(i * H + tl.arange(0, H)[None, :] < P)
            )
        offs += H * N
        toffs += H
        soffs += H


def triton_transpose_smooth_quant(
    x, smooth_scale, reverse=False, pad=False, round_scale=False
):
    # col-wise read, row-wise write
    # M should be padded if M % 32 != 0
    """"""
    M, N = x.shape
    device = x.device
    P = (M + 31) // 32 * 32 if pad else M
    x_q = torch.empty((N, P), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((N,), device=device, dtype=torch.float32)
    H = 1024
    W = 16  # if N >= 4096 else 16
    assert N % W == 0
    EVEN = P % H == 0 and M == P

    grid = (triton.cdiv(N, W),)
    transpose_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scale,
        x_scale,
        M,
        N,
        P,
        H,
        W,
        EVEN,
        reverse,
        round_scale,
        num_stages=3,
        num_warps=4 if N >= 8192 else 4,
    )
    return x_q, x_scale


@triton.jit
def transpose_rescale_smooth_quant_kernel(
    x_ptr,
    q_ptr,
    org_smooth_scale_ptr,
    org_quant_scale_ptr,
    transpose_smooth_scale_ptr,
    transpose_quant_scale_ptr,
    M,
    N,
    P,
    H: tl.constexpr,
    W: tl.constexpr,
    EVEN: tl.constexpr,
    ROUND: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # col-wise read, row-wise write
    offs = pid * W + tl.arange(0, H)[:, None] * N + tl.arange(0, W)[None, :]
    soffs = tl.arange(0, H)
    x_max = tl.zeros((W,), dtype=tl.float32)
    org_smooth_scale = tl.load(org_smooth_scale_ptr + pid * W + tl.arange(0, W))[
        None, :
    ]

    m = tl.cdiv(P, H)
    for i in range(m):
        if EVEN:
            x = tl.load(x_ptr + offs).to(tl.float32)
            org_quant_scale = tl.load(org_quant_scale_ptr + soffs)[:, None]
            transpose_smooth_scale = tl.load(transpose_smooth_scale_ptr + soffs)[
                :, None
            ]
        else:
            x = tl.load(x_ptr + offs, mask=(i * H + tl.arange(0, H)[:, None] < M)).to(
                tl.float32
            )
            org_quant_scale = tl.load(
                org_quant_scale_ptr + soffs, mask=soffs < M, other=0.0
            )[:, None]
            transpose_smooth_scale = tl.load(
                transpose_smooth_scale_ptr + soffs, mask=soffs < M, other=0.0
            )[:, None]

        x = x / org_smooth_scale * (org_quant_scale * transpose_smooth_scale)
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0), x_max)
        offs += H * N
        soffs += H

    scale = tl.maximum(x_max / 448.0, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))

    tl.store(transpose_quant_scale_ptr + pid * W + tl.arange(0, W), scale)

    s = (1.0 / scale)[None, :]

    offs = pid * W + tl.arange(0, H)[:, None] * N + tl.arange(0, W)[None, :]
    soffs = tl.arange(0, H)
    toffs = pid * W * P + tl.arange(0, W)[:, None] * P + tl.arange(0, H)[None, :]
    for i in range(m):

        if EVEN:
            x = tl.load(x_ptr + offs).to(tl.float32)
            org_quant_scale = tl.load(org_quant_scale_ptr + soffs)[:, None]
            transpose_smooth_scale = tl.load(transpose_smooth_scale_ptr + soffs)[
                :, None
            ]
        else:
            x = tl.load(
                x_ptr + offs,
                mask=(i * H + tl.arange(0, H)[:, None] < M)
                & (pid * W + tl.arange(0, W)[None, :] < N),
            ).to(tl.float32)
            org_quant_scale = tl.load(
                org_quant_scale_ptr + soffs, mask=soffs < M, other=0.0
            )[:, None]
            transpose_smooth_scale = tl.load(
                transpose_smooth_scale_ptr + soffs, mask=soffs < M, other=0.0
            )[:, None]

        x = x * s / org_smooth_scale * (org_quant_scale * transpose_smooth_scale)
        x = tl.trans(x.to(q_ptr.dtype.element_ty))
        if EVEN:
            tl.store(q_ptr + toffs, x)
        else:
            tl.store(q_ptr + toffs, x, mask=(i * H + tl.arange(0, H)[None, :] < P))
        offs += H * N
        toffs += H
        soffs += H


"""
x_q is colwise smooth and rowwise quant
org_smooth_scale and transpose_smooth_scale is reversed
smooth scale and quant scale should be power of 2
step: dequant x_q -> apply smooth scale -> quant -> transpose -> pad
implement: x_q/org_smooth_scale*(org_quant_scale*smooth_scale) -> colwise quant and transpose
"""


def triton_transpose_rescale_smooth_quant(
    x_q,
    org_smooth_scale,
    org_quant_scale,
    transpose_smooth_scale,
    reverse=True,
    pad=False,
    round_scale=False,
):
    """"""
    assert reverse
    M, N = x_q.shape
    device = x_q.device
    P = round_up(M, b=32) if pad else M
    xt_q = torch.empty((N, P), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((N,), device=device, dtype=torch.float32)
    H = 256
    W = 16
    assert N % W == 0
    EVEN = P == M and M % H == 0

    grid = (triton.cdiv(N, W),)
    transpose_rescale_smooth_quant_kernel[grid](
        x_q,
        xt_q,
        org_smooth_scale,
        org_quant_scale,
        transpose_smooth_scale,
        x_scale,
        M,
        N,
        P,
        H,
        W,
        EVEN,
        round_scale,
        num_stages=4,
        num_warps=8,
    )

    return xt_q, x_scale


"""
megatron fp8 training steps:
step 0: init w smooth scale w_smooth
step 1: smooth and quant w after w is updated by optimizer
step 2: in forward step, columnwise smooth x and rowwise quant x, calc y=x@w; 
            meanwhile, record the columnwise max of x, it is used to update w_smooth
step 3: in dgrad step, columnwise smooth y and rowwise quant y, transpose x, calc dx=y@wT 
step 4: in wgrad step, dequant then smooth an then quant y_q to get yt_q, calc dw=yT@x

alternative (it's not suitable for fp8 combine):
step 4: in wgrad step, rowwise smooth y and columnwise quant y and transpose to get yt_q, calc dw=yT@x

"""

"""
divide x by smooth_scale and row-wise quantization
smooth scale is updated by square root of x's column-wise maxs, and set in weight's x_maxs attr

transpose: transpose quantized x for wgrad
pad: # pad M to be multiplier of 32, including quant scales and transposed x

"""


# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_smooth_quant_input(
    x,
    smooth_scale,
    x_q=None,
    x_scale=None,
    xt_q=None,
    transpose=True,
    pad=True,
    round_scale=False,
):
    """"""
    x_q, x_scale, x_maxs = triton_smooth_quant(
        x,
        smooth_scale,
        x_q=x_q,
        x_scale=x_scale,
        reverse=False,
        round_scale=round_scale,
    )

    if transpose:
        xt_q = triton_transpose_and_pad(x_q, out=xt_q, pad=pad)
    else:
        xt_q = None
    xt_scale = smooth_scale

    return x_q, xt_q, x_scale, xt_scale


# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_smooth_quant_gradient(
    y,
    smooth_scale,
    transpose_smooth_scale,
    reverse=True,
    transpose=True,
    pad=True,
    round_scale=False,
):
    """"""
    assert reverse, (
        "args `smooth_scale` and/or `transpose_smooth_scale` "
        "must be in reciprocal format in triton_smooth_quant_grad"
    )
    y_q, y_scale, _ = triton_smooth_quant(
        y, smooth_scale, reverse=True, round_scale=round_scale
    )
    if transpose:
        yt_q, yt_scale = triton_transpose_smooth_quant(
            y, transpose_smooth_scale, reverse=True, pad=pad, round_scale=round_scale
        )
    else:
        yt_q, yt_scale = None, None

    return y_q, yt_q, y_scale, yt_scale


def triton_smooth_quant_weight(
    w, smooth_scale, w_q, quant_scale, subrow_scales, offset=0, round_scale=False
):
    """"""
    assert w.ndim == 1
    assert w_q.size(1) == smooth_scale.size(0)

    size = w.numel()
    M, N = w_q.shape

    if size == M * N:
        triton_smooth_quant(
            w.view(M, N),
            smooth_scale,
            x_q=w_q,
            x_scale=quant_scale,
            round_scale=round_scale,
        )
    elif offset % N == 0 and size % N == 0:
        n_row = size // N
        row_id = offset // N
        w_q_slice = w_q[row_id : row_id + n_row]
        quant_scale_slice = quant_scale[row_id : row_id + n_row]
        triton_smooth_quant(
            w.view(n_row, N),
            smooth_scale,
            x_q=w_q_slice,
            x_scale=quant_scale_slice,
            round_scale=round_scale,
        )
    else:
        row_si = (offset - 1) // N + 1
        row_ei = (offset + size) // N
        col_si = offset % N
        col_ei = (offset + size) % N
        n_row = row_ei - row_si
        mw_offset = 0 if col_si == 0 else N - col_si
        w_q_slice = w_q[row_si:row_ei]
        quant_scale_slice = quant_scale[row_si:row_ei]
        w_slice = w[mw_offset : mw_offset + n_row * N].view(n_row, N)
        triton_smooth_quant(
            w_slice,
            smooth_scale,
            x_q=w_q_slice,
            x_scale=quant_scale_slice,
            round_scale=round_scale,
        )

        # subrow scale is writed by the row with leading master weights
        if col_si > 0 or col_ei > 0:
            triton_subrow_smooth_quant(
                w,
                smooth_scale,
                w_q,
                quant_scale,
                subrow_scales,
                offset,
                size,
                reverse=False,
                round_scale=round_scale,
            )
