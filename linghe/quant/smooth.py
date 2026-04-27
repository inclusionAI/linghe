# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def tokenwise_smooth_quant_kernel(
    x_ptr,
    q_ptr,
    ss_ptr,
    qs_ptr,
    M,
    T,
    N: tl.constexpr,
    W: tl.constexpr,
    EVEN: tl.constexpr,
    REVERSE: tl.constexpr,
    ROUND: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    smooth_scale = tl.load(ss_ptr + tl.arange(0, N))[None, :]
    if not REVERSE:
        smooth_scale = 1.0 / smooth_scale

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
        x *= smooth_scale
        x_max = tl.max(tl.abs(x), axis=1)
        scale = tl.maximum(x_max / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        if EVEN:
            tl.store(qs_ptr + pid * W * T + i * W + tl.arange(0, W), scale)
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


@triton.jit
def blockwise_smooth_quant_kernel(
    x_ptr,
    q_ptr,
    ss_ptr,
    qs_ptr,
    M,
    N,
    H: tl.constexpr,
    W: tl.constexpr,
    EVEN: tl.constexpr,
    REVERSE: tl.constexpr,
    ROUND: tl.constexpr,
):
    pid = tl.program_id(axis=0)
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
    x, smooth_scale, x_q=None, x_scale=None, reverse=False, round_scale=False
):
    # it may be used for sharded weight quantization, therefore M is not exact batch size
    M, N = x.shape
    device = x.device
    if x_q is None:
        x_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    if x_scale is None:
        x_scale = torch.empty((M,), device=device, dtype=torch.float32)
    if triton.next_power_of_2(N) == N and N <= 8192:
        W = 8192 // N
        T = 8
        EVEN = M % (W * T) == 0
        g = triton.cdiv(M, W * T)
        tokenwise_smooth_quant_kernel[(g,)](
            x,
            x_q,
            smooth_scale,
            x_scale,
            M,
            T,
            N,
            W,
            EVEN,
            reverse,
            round_scale,
            num_stages=3,
            num_warps=4,
        )
    else:
        # N may be 576 in MLA
        H = max([x for x in [64, 128, 256, 512, 1024, 2048] if N % x == 0])
        W = 8 if M > 8192 else 4
        EVEN = M % W == 0
        T = triton.cdiv(M, W)
        grid = (T,)
        blockwise_smooth_quant_kernel[grid](
            x,
            x_q,
            smooth_scale,
            x_scale,
            M,
            N,
            H,
            W,
            EVEN,
            reverse,
            round_scale,
            num_stages=3,
            num_warps=4,
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
    offs = pid * W + tl.arange(0, H)[:, None] * N + tl.arange(0, W)[None, :]
    soffs = tl.arange(0, H)
    x_max = tl.zeros((W,), dtype=tl.float32)
    m = tl.cdiv(P, H)
    for i in range(m):
        if EVEN:
            x = tl.load(x_ptr + offs)
            smooth_scale = tl.load(ss_ptr + soffs)[:, None]
        else:
            mask = (i * H + tl.arange(0, H)[:, None] < M) & (
                pid * W + tl.arange(0, W)[None, :] < N
            )
            x = tl.load(x_ptr + offs, mask=mask)
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
    x, smooth_scale, reverse=False, pad=True, round_scale=False
):
    # M should be padded to mutiple of 32 if pad is True
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
def batch_smooth_quant_kernel(
    x_ptr,
    q_ptr,
    ss_ptr,
    qs_ptr,
    count_ptr,
    accum_ptr,
    T,
    N: tl.constexpr,
    REVERSE: tl.constexpr,
    ROUND: tl.constexpr,
):
    eid = tl.program_id(axis=0)
    tid = tl.program_id(axis=1)

    smooth_scale = tl.load(ss_ptr + eid * N + tl.arange(0, N))
    if not REVERSE:
        smooth_scale = 1.0 / smooth_scale

    count = tl.load(count_ptr + eid)
    ei = tl.load(accum_ptr + eid)
    si = ei - count

    n = tl.cdiv(count, T)  # tokens per block
    for i in range(tid * n, min((tid + 1) * n, count)):
        x = tl.load(x_ptr + si * N + i * N + tl.arange(0, N)).to(tl.float32)
        x *= smooth_scale
        scale = tl.maximum(tl.max(tl.abs(x)) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))

        tl.store(qs_ptr + si + i, scale)

        s = 1.0 / scale
        x *= s
        xq = x.to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + si * N + i * N + tl.arange(0, N), xq)


def triton_batch_smooth_quant(
    x, smooth_scales, token_count_per_expert, reverse=False, round_scale=False
):
    """
    smooth quant
    x: [sum(tokens), dim]
    smooth_scales: [n_experts, dim]
    token_count_per_expert: [n_experts]
    reverse: x * smooth_scale if reverse else x / smooth_scale
    x_scale: [bs]
    """
    M, N = x.shape
    device = x.device
    n_expert = token_count_per_expert.shape[0]
    x_q = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((M,), device=device, dtype=torch.float32)
    accum_token_count = torch.cumsum(token_count_per_expert, 0)
    T = 128

    grid = (n_expert, T)
    batch_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scales,
        x_scale,
        token_count_per_expert,
        accum_token_count,
        T,
        N,
        reverse,
        round_scale,
        num_stages=3,
        num_warps=4,
    )
    return x_q, x_scale


@triton.jit
def batch_transpose_smooth_quant_kernel(
    x_ptr,
    q_ptr,
    ss_ptr,
    qs_ptr,
    count_ptr,
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
    round_count = tl.cdiv(count, 32) * 32

    counts = tl.load(count_ptr + tl.arange(0, E))
    si = tl.sum(tl.where(tl.arange(0, E) < eid, counts, 0))

    round_si = tl.sum(tl.where(tl.arange(0, E) < eid, tl.cdiv(counts, 32), 0)) * 32

    n = tl.cdiv(count, H)
    maxs = tl.zeros((H, W), dtype=tl.float32)
    for i in range(n):
        indices = i * H + tl.arange(0, H)
        smooth_scale = tl.load(ss_ptr + si + indices, mask=indices < count)
        if not REVERSE:
            smooth_scale = 1.0 / smooth_scale

        x = tl.load(
            x_ptr
            + si * N
            + i * H * N
            + bid * W
            + tl.arange(0, H)[:, None] * N
            + tl.arange(0, W)[None, :],
            mask=indices[:, None] < count,
        ).to(tl.float32)
        x *= smooth_scale[:, None]
        maxs = tl.maximum(maxs, tl.abs(x))

    scale = tl.maximum(tl.max(maxs, 0) / 448.0, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(qs_ptr + eid * N + bid * W + tl.arange(0, W), scale)
    s = 1.0 / scale

    for i in range(n):
        indices = i * H + tl.arange(0, H)
        smooth_scale = tl.load(ss_ptr + si + indices, mask=indices < count)
        if not REVERSE:
            smooth_scale = 1.0 / smooth_scale

        x = tl.load(
            x_ptr
            + si * N
            + i * H * N
            + bid * W
            + tl.arange(0, H)[:, None] * N
            + tl.arange(0, W)[None, :],
            mask=indices[:, None] < count,
        ).to(tl.float32)
        x *= smooth_scale[:, None]
        x *= s
        xq = tl.trans(x.to(q_ptr.dtype.element_ty))
        tl.store(
            q_ptr
            + round_si * N
            + bid * W * round_count
            + i * H
            + tl.arange(0, W)[:, None] * round_count
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


def triton_batch_transpose_smooth_quant(
    x,
    smooth_scales,
    token_count_per_expert,
    splits,
    pad=True,
    reverse=False,
    round_scale=False,
):
    """"""
    assert pad and reverse
    M, N = x.shape
    device = x.device
    n_expert = token_count_per_expert.shape[0]
    round_splits = [(x + 31) // 32 * 32 for x in splits]
    x_q = torch.empty((sum(round_splits), N), device=device, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty((n_expert, N), device=device, dtype=torch.float32)
    H = 128
    W = 32
    grid = (n_expert, N // W)
    batch_transpose_smooth_quant_kernel[grid](
        x,
        x_q,
        smooth_scales,
        x_scale,
        token_count_per_expert,
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
    P = (M + 31) // 32 * 32 if pad else M
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
