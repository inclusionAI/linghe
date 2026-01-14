# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def deprecated_mla_forward_kernel(
        Q,
        K,
        V,
        Out,
        LSE,
        ML,
        softmax_scale,
        stride_q,
        stride_k,
        stride_v,
        L,
        M: tl.constexpr,
        N: tl.constexpr,
        CAUSAL: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.num_programs(2) - tl.program_id(2) - 1
    H = tl.num_programs(1)

    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)

    offs_0 = tl.arange(0, 128)  # nope
    offs_1 = tl.arange(0, 64)  # pe

    # [B, L, H, 192】
    q0_ptrs = (
            Q
            + (bid * L + mid * M) * stride_q
            + hid * 192
            + (offs_m[:, None] * stride_q + offs_0[None, :])
    )
    q1_ptrs = (
            Q
            + (bid * L + mid * M) * stride_q
            + hid * 192
            + 128
            + (offs_m[:, None] * stride_q + offs_1[None, :])
    )

    k0_ptrs = (
            K
            + bid * L * stride_k
            + hid * 192
            + (offs_n[:, None] * stride_k + offs_0[None, :])
    )

    k1_ptrs = (
            K
            + bid * L * stride_k
            + hid * 192
            + 128
            + (offs_n[:, None] * stride_k + offs_1[None, :])
    )

    v_ptrs = (
            V
            + bid * L * stride_v
            + hid * 128
            + (offs_n[:, None] * stride_v + offs_0[None, :])
    )

    q0 = tl.load(q0_ptrs)
    q1 = tl.load(q1_ptrs)

    lse = tl.zeros((M,), dtype=tl.float32)
    acc_o = tl.zeros((M, 128), dtype=tl.float32)
    if CAUSAL:
        steps = tl.cdiv(mid * M + M, N)
    else:
        steps = L // N
    for i in range(0, steps):
        n = i * N
        n = tl.multiple_of(n, N)

        k1 = tl.load(k1_ptrs + n * stride_k)

        qk = tl.dot(q1, tl.trans(k1))

        k0 = tl.load(k0_ptrs + n * stride_k)

        qk = tl.dot(q0, tl.trans(k0), qk)

        qk += tl.where((mid * M + offs_m)[:, None] >= (n + offs_n)[None, :],
                       0.0, -1e9)

        p = tl.exp(qk * softmax_scale)
        lse += tl.sum(p, 1)

        v = tl.load(v_ptrs + n * stride_v)
        p = p.to(V.dtype.element_ty)
        acc_o = tl.dot(p, v, acc_o)

    acc_o = acc_o / lse[:, None]

    # [B, L, H, 128]
    out_ptrs = (
            Out
            + (bid * L + mid * M) * H * 128
            + hid * 128
            + (offs_m[:, None] * 128 * H + offs_0[None, :])
    )

    tl.store(out_ptrs, acc_o)
    tl.store(LSE + bid * H * L + hid * L + mid * M + tl.arange(0, M), lse)


@triton.jit
def mla_forward_kernel(
        Q,
        K,
        V,
        Out,
        LSE,
        ML,
        softmax_scale,
        clip_value,
        stride_q,
        stride_k,
        stride_v,
        L,
        M: tl.constexpr,
        N: tl.constexpr,
        CAUSAL: tl.constexpr,
        SAFE: tl.constexpr,
        CLIP: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.num_programs(2) - tl.program_id(2) - 1
    H = tl.num_programs(1)

    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)

    offs_0 = tl.arange(0, 128)
    offs_1 = tl.arange(0, 64)

    # [B, L, H, 192】
    q0_ptrs = (
            Q
            + (bid * L + mid * M) * stride_q
            + hid * 192
            + (offs_m[:, None] * stride_q + offs_1[None, :])
    )

    k0_ptrs = (
            K
            + bid * L * stride_k
            + hid * 192
            + (offs_n[:, None] * stride_k + offs_1[None, :])
    )

    v_ptrs = (
            V
            + bid * L * stride_v
            + hid * 128
            + (offs_n[:, None] * stride_v + offs_0[None, :])
    )

    q0 = tl.load(q0_ptrs)
    q1 = tl.load(q0_ptrs + 64)
    q2 = tl.load(q0_ptrs + 128)

    acc_o = tl.zeros((M, 128), dtype=tl.float32)
    if SAFE:
        max_logits = tl.zeros((M,), dtype=tl.float32) - 10000.0
        lse = tl.zeros((M,), dtype=tl.float32)
    else:
        lse = tl.zeros((M,), dtype=tl.float32) + 1e-30

    if CAUSAL:
        steps = tl.cdiv(mid * M + M, N)
    else:
        steps = L // N

    for i in range(0, steps):
        n = i * N
        n = tl.multiple_of(n, N)

        k0 = tl.load(k0_ptrs + n * stride_k)
        k1 = tl.load(k0_ptrs + n * stride_k + 64)
        k2 = tl.load(k0_ptrs + n * stride_k + 128)

        qk = tl.dot(q0, tl.trans(k0))
        qk = tl.dot(q1, tl.trans(k1), qk)
        qk = tl.dot(q2, tl.trans(k2), qk)

        if CAUSAL:
            qk += tl.where((mid * M + offs_m)[:, None] >= (n + offs_n)[None, :],
                           0.0, -1e9)

        qk *= softmax_scale

        if SAFE:
            latest_max_logits = tl.maximum(max_logits, tl.max(qk, 1))
            p = tl.exp(qk - latest_max_logits[:, None])
            rescale = tl.exp(max_logits - latest_max_logits)
            lse = lse * rescale + tl.sum(p, 1)
            v = tl.load(v_ptrs + n * stride_v)
            p = p.to(V.dtype.element_ty)
            acc_o = acc_o * rescale[:, None]
            acc_o = tl.dot(p, v, acc_o)
            max_logits = latest_max_logits
        else:
            if CLIP:
                p = tl.exp(tl.minimum(qk, clip_value))
            else:
                p = tl.exp(qk)
            lse += tl.sum(p, 1)
            v = tl.load(v_ptrs + n * stride_v)
            p = p.to(V.dtype.element_ty)
            acc_o = tl.dot(p, v, acc_o)

    acc_o = acc_o / lse[:, None]

    # [B, L, H, 128]
    out_ptrs = (
            Out
            + (bid * L + mid * M) * H * 128
            + hid * 128
            + (offs_m[:, None] * 128 * H + offs_0[None, :])
    )

    tl.store(out_ptrs, acc_o)
    tl.store(LSE + bid * H * L + hid * L + mid * M + tl.arange(0, M), lse)
    if SAFE:
        tl.store(ML + bid * H * L + hid * L + mid * M + tl.arange(0, M),
                 max_logits)


def triton_mla_forward(q, k, v, causal=True, safe=True, clip_value=None):
    # q: [B, L, H, 192]
    # k: [B, L, H, 192]
    # v: [B, L, H, 128]
    B, L, H, _ = q.shape
    assert k.size(1) == L
    M = 256
    N = 64
    assert L % M == 0
    assert L % N == 0
    assert M >= N

    o = torch.empty((B, L, H, 128), dtype=q.dtype, device=q.device)
    lse = torch.empty((B, H, L), dtype=torch.float32, device=q.device)
    max_logits = torch.empty((B, H, L), dtype=torch.float32, device=q.device)
    softmax_scale = 128 ** (-0.5)

    clip = clip_value is not None
    clip_value = clip_value * softmax_scale if clip else 0.0

    if clip and clip_value + math.log(L) < 88.7:
        safe = False

    num_m_block = L // M
    num_stages = 2
    num_warps = 8

    grid = (B, H, num_m_block)
    mla_forward_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        max_logits,
        softmax_scale,
        clip_value,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        L,
        M,
        N,
        causal,
        safe,
        clip,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o, lse, max_logits


# dp and p dot sum
@triton.jit
def naive_mla_ds_kernel(
        GO,
        Q,
        K,
        V,
        LSE,
        ML,
        DS,
        softmax_scale,
        stride_q,
        stride_k,
        stride_v,
        L,
        M: tl.constexpr,
        N: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.num_programs(2) - tl.program_id(2) - 1
    H = tl.num_programs(1)

    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)

    offs_0 = tl.arange(0, 128)  # nope
    offs_1 = tl.arange(0, 64)  # pe

    # [B, L, H, 192】
    q0_ptrs = (
            Q
            + (bid * L + mid * M) * stride_q
            + hid * 192
            + (offs_m[:, None] * stride_q + offs_1[None, :])
    )

    k0_ptrs = (
            K
            + bid * L * stride_k
            + hid * 192
            + (offs_n[:, None] * stride_k + offs_1[None, :])
    )

    v_ptrs = (
            V
            + bid * L * stride_v
            + hid * 128
            + (offs_n[:, None] * stride_v + offs_0[None, :])
    )

    go_ptrs = (
            GO
            + (bid * L + mid * M) * H * 128
            + hid * 128
            + (offs_m[:, None] * 128 * H + offs_0[None, :])
    )

    ds = tl.zeros((M,), dtype=tl.float32)

    q0 = tl.load(q0_ptrs)
    q1 = tl.load(q0_ptrs + 64)
    q2 = tl.load(q0_ptrs + 128)

    go = tl.load(go_ptrs)
    steps = tl.cdiv(mid * M + M, N)

    ds = tl.zeros((M, N), dtype=tl.float32)

    for i in range(0, steps):
        n = i * N
        n = tl.multiple_of(n, N)

        k0 = tl.load(k0_ptrs + n * stride_k)

        qk = tl.dot(q0, tl.trans(k0))

        k1 = tl.load(k0_ptrs + n * stride_k + 64)

        qk = tl.dot(q1, tl.trans(k1), qk)

        k2 = tl.load(k0_ptrs + n * stride_k + 128)

        qk = tl.dot(q2, tl.trans(k2), qk)

        qk += tl.where((mid * M + offs_m)[:, None] >= (n + offs_n)[None, :],
                       0.0, -1e9)

        p = tl.exp(qk * softmax_scale)  # [M, N]
        v = tl.load(v_ptrs + n * stride_v)
        dp = tl.dot(go, tl.trans(v))  # [M, 128]@[128, N]=[M,N]
        # ds += tl.sum(p * dp, 1)  # score
        ds += p * dp  # score

    lse = tl.load(LSE + bid * H * L + hid * L + mid * M + tl.arange(0, M))
    ds = ds.sum(1) / lse
    tl.store(DS + bid * H * L + hid * L + mid * M + tl.arange(0, M), ds)


# dp and p dot sum
@triton.jit
def mla_ds_kernel(
        G,
        O,
        DS,
        L,
        M: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.program_id(2)
    H = tl.num_programs(1)

    offs_m = tl.arange(0, M)
    offs_0 = tl.arange(0, 128)  # nope

    # [B, L, H, 128】
    offs = ((bid * L + mid * M) * H * 128
            + hid * 128
            + (offs_m[:, None] * H * 128 + offs_0[None, :])
            )

    mask = mid * M + offs_m < L
    g = tl.load(G + offs, mask=mask[:, None]).to(tl.float32)
    o = tl.load(O + offs, mask=mask[:, None]).to(tl.float32)
    ds = tl.sum(g * o, 1)
    tl.store(DS + bid * H * L + hid * L + mid * M + tl.arange(0, M), ds,
             mask=mask)


@triton.jit
def deprecated_mla_backward_kernel(
        GO,
        Q,
        K,
        V,
        GQ,
        GK,
        GV,
        LSE,
        ML,
        DS,
        softmax_scale,
        stride_q,
        stride_k,
        stride_v,
        L,
        M: tl.constexpr,
        N: tl.constexpr,
        ATOMIC: tl.constexpr,  # not used
        CAUSAL: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    nid = tl.program_id(2)
    H = tl.num_programs(1).to(tl.int64)
    B = tl.num_programs(0)

    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)

    offs_1 = tl.arange(0, 64)  # pe

    # [B, L, H, 192】
    q0_ptrs = (
            Q
            + bid * L * stride_q
            + hid * 192
            + (offs_m[:, None] * stride_q + offs_1[None, :])
    )

    k0_ptrs = (
            K
            + (bid * L + nid * N) * stride_k
            + hid * 192
            + (offs_n[:, None] * stride_k + offs_1[None, :])
    )

    k0 = tl.load(k0_ptrs)
    k1 = tl.load(k0_ptrs + 64)
    k2 = tl.load(k0_ptrs + 128)

    v0_ptrs = (
            V
            + (bid * L + nid * N) * stride_v
            + hid * 128
            + (offs_n[:, None] * stride_v + offs_1[None, :])
    )
    v0 = tl.load(v0_ptrs)
    v1 = tl.load(v0_ptrs + 64)

    go_ptrs = (
            GO
            + bid * L * H * 128
            + hid * 128
            + (offs_m[:, None] * 128 * H + offs_1[None, :])
    )

    dq0_ptrs = (
            GQ
            + nid * B * L * H * 192
            + bid * L * H * 192
            + hid * 192
            + (offs_m[:, None] * H * 192 + offs_1[None, :])
    )

    dv0 = tl.zeros((N, 64), dtype=tl.float32)
    dv1 = tl.zeros((N, 64), dtype=tl.float32)
    dk0 = tl.zeros((N, 64), dtype=tl.float32)
    dk1 = tl.zeros((N, 64), dtype=tl.float32)
    dk2 = tl.zeros((N, 64), dtype=tl.float32)
    if CAUSAL:
        step = nid * N
    else:
        step = 0
    for m in range(step, L, M):
        lse = tl.load(LSE + bid * H * L + hid * L + m + tl.arange(0, M))
        ds = tl.load(DS + bid * H * L + hid * L + m + tl.arange(0, M))

        q0 = tl.load(q0_ptrs + m * stride_q)
        q1 = tl.load(q0_ptrs + m * stride_q + 64)
        q2 = tl.load(q0_ptrs + m * stride_q + 128)

        qk = tl.dot(q0, tl.trans(k0))

        qk = tl.dot(q1, tl.trans(k1), qk)

        qk = tl.dot(q2, tl.trans(k2), qk)

        go0 = tl.load(go_ptrs + m * H * 128)
        go1 = tl.load(go_ptrs + m * H * 128 + 64)

        if CAUSAL:
            qk += tl.where((m + offs_m)[:, None] >= (nid * N + offs_n)[None, :],
                           0.0, -1e9)
        p = tl.exp(qk * softmax_scale) / lse[:, None]

        dp = tl.dot(go0, tl.trans(v0))  # [M, 128]@[128, N]=[M,N]
        dp = tl.dot(go1, tl.trans(v1), dp)
        dp = p * (dp - ds[:, None]) * softmax_scale  # score

        p = p.to(V.dtype.element_ty)
        dv0 = tl.dot(tl.trans(p), go0, dv0)  # [N, M]@[M, 128]=[N, 128]
        dv1 = tl.dot(tl.trans(p), go1, dv1)  # [N, M]@[M, 128]=[N, 128]

        dp = dp.to(V.dtype.element_ty)
        dk0 = tl.dot(tl.trans(dp), q0, dk0)  # [N, M]@[M, 128]=[N, 128]
        dk1 = tl.dot(tl.trans(dp), q1, dk1)  # [N, M]@[M, 64]=[N, 64]
        dk2 = tl.dot(tl.trans(dp), q2, dk2)  # [N, M]@[M, 64]=[N, 64]
        dq0 = tl.dot(dp, k0)  # [M, N]@[N, 128]=[M, 128]
        tl.store(dq0_ptrs + m * H * 192, dq0)

        dq1 = tl.dot(dp, k1)  # [M, N]@[N, 64]=[M, 64]
        tl.store(dq0_ptrs + m * H * 192 + 64, dq1)

        dq2 = tl.dot(dp, k2)  # [M, N]@[N, 64]=[M, 64]
        tl.store(dq0_ptrs + m * H * 192 + 128, dq2)

    gv_ptrs = (
            GV
            + (bid * L + nid * N) * H * 128
            + hid * 128
            + (offs_n[:, None] * 128 * H + offs_1[None, :])
    )

    tl.store(gv_ptrs, dv0)
    tl.store(gv_ptrs + 64, dv1)

    gk0_ptrs = (
            GK
            + (bid * L + nid * N) * H * 192
            + hid * 192
            + (offs_n[:, None] * 192 * H + offs_1[None, :])
    )
    tl.store(gk0_ptrs, dk0)
    tl.store(gk0_ptrs + 64, dk1)
    tl.store(gk0_ptrs + 128, dk2)


@triton.jit
def mla_backward_kernel(
        GO,
        Q,
        K,
        V,
        GQ,
        GK,
        GV,
        LSE,
        ML,
        DS,
        softmax_scale,
        clip_value,
        stride_q,
        stride_k,
        stride_v,
        L,
        M: tl.constexpr,
        N: tl.constexpr,
        ATOMIC: tl.constexpr,
        CAUSAL: tl.constexpr,
        SAFE: tl.constexpr,
        CLIP: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    nid = tl.program_id(2)
    H = tl.num_programs(1).to(tl.int64)
    B = tl.num_programs(0)

    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)

    offs_0 = tl.arange(0, 128)  # nope
    offs_1 = tl.arange(0, 64)  # pe

    # [B, L, H, 192】
    q0_ptrs = (
            Q
            + bid * L * stride_q
            + hid * 192
            + (offs_m[:, None] * stride_q + offs_0[None, :])
    )
    q1_ptrs = (
            Q
            + bid * L * stride_q
            + hid * 192
            + 128
            + (offs_m[:, None] * stride_q + offs_1[None, :])
    )

    k0_ptrs = (
            K
            + (bid * L + nid * N) * stride_k
            + hid * 192
            + (offs_n[:, None] * stride_k + offs_0[None, :])
    )
    k1_ptrs = (
            K
            + (bid * L + nid * N) * stride_k
            + hid * 192
            + 128
            + (offs_n[:, None] * stride_k + offs_1[None, :])
    )
    k0 = tl.load(k0_ptrs)
    k1 = tl.load(k1_ptrs)

    v_ptrs = (
            V
            + (bid * L + nid * N) * stride_v
            + hid * 128
            + (offs_n[:, None] * stride_v + offs_0[None, :])
    )
    v = tl.load(v_ptrs)

    go_ptrs = (
            GO
            + bid * L * H * 128
            + hid * 128
            + (offs_m[:, None] * 128 * H + offs_0[None, :])
    )

    if ATOMIC:
        # [B, L, H, 192]
        dq0_ptrs = (
                GQ
                + bid * L * H * 192
                + hid * 192
                + (offs_m[:, None] * H * 192 + offs_0[None, :])
        )
        dq1_ptrs = (
                GQ
                + bid * L * H * 192
                + hid * 192
                + 128
                + (offs_m[:, None] * H * 192 + offs_1[None, :])
        )
    else:
        dq0_ptrs = (
                GQ
                + nid * B * L * H * 192
                + bid * L * H * 192
                + hid * 192
                + (offs_m[:, None] * H * 192 + offs_0[None, :])
        )
        dq1_ptrs = (
                GQ
                + nid * B * L * H * 192
                + bid * L * H * 192
                + hid * 192
                + 128
                + (offs_m[:, None] * H * 192 + offs_1[None, :])
        )

    dv = tl.zeros((N, 128), dtype=tl.float32)
    dk0 = tl.zeros((N, 128), dtype=tl.float32)
    dk1 = tl.zeros((N, 64), dtype=tl.float32)
    if CAUSAL:
        step = nid * N
        n_steps = tl.cdiv(L - step, M)
    else:
        step = 0
        n_steps = tl.cdiv(L, M)

    for i in range(n_steps):
        m = step + (n_steps - 1 - i) * M
        lse = 1 / tl.load(LSE + bid * H * L + hid * L + m + tl.arange(0, M))
        if SAFE:
            max_logits = tl.load(
                ML + bid * H * L + hid * L + m + tl.arange(0, M))
        ds = tl.load(DS + bid * H * L + hid * L + m + tl.arange(0, M))

        q0 = tl.load(q0_ptrs + m * stride_q)
        q1 = tl.load(q1_ptrs + m * stride_q)
        go = tl.load(go_ptrs + m * H * 128)

        if CAUSAL:
            qk = tl.where((m + offs_m)[:, None] >= (nid * N + offs_n)[None, :],
                          0.0, -10000.0)
            qk = tl.dot(q1, tl.trans(k1), qk)
            qk = tl.dot(q0, tl.trans(k0), qk)
        else:
            qk = tl.dot(q1, tl.trans(k1))
            qk = tl.dot(q0, tl.trans(k0), qk)

        qk *= softmax_scale
        if CLIP:
            qk = tl.minimum(qk, clip_value)

        if SAFE:
            p = tl.exp(qk - max_logits[:, None]) * lse[:, None]
        else:
            p = tl.exp(qk) * lse[:, None]

        # impl 0
        dp = tl.dot(go, tl.trans(v))  # [M, 128]@[128, N]=[M,N]
        dp = p * (dp - ds[:, None]) * softmax_scale  # score
        # impl 1
        # dp = tl.zeros((1, N), dtype=tl.float32) - ds[:,None]
        # dp = tl.dot(go, tl.trans(v), dp)  # [M, 128]@[128, N]=[M,N]
        # dp = softmax_scale * dp * p  # score

        p = p.to(V.dtype.element_ty)
        dv = tl.dot(tl.trans(p), go, dv)  # [N, M]@[M, 128]=[N, 128]

        dp = dp.to(V.dtype.element_ty)
        dq0 = tl.dot(dp, k0)  # [M, N]@[N, 128]=[M, 128]
        dq1 = tl.dot(dp, k1)  # [M, N]@[N, 64]=[M, 64]
        if ATOMIC:
            tl.atomic_add(dq0_ptrs + m * H * 192, dq0, sem='relaxed')
            tl.atomic_add(dq1_ptrs + m * H * 192, dq1, sem='relaxed')
        else:
            tl.store(dq0_ptrs + m * H * 192, dq0)
            tl.store(dq1_ptrs + m * H * 192, dq1)

        dp = tl.trans(dp)
        dk0 = tl.dot(dp, q0, dk0)  # [N, M]@[M, 128]=[N, 128]
        dk1 = tl.dot(dp, q1, dk1)  # [N, M]@[M, 64]=[N, 64]

    gv_ptrs = (
            GV
            + (bid * L + nid * N) * H * 128
            + hid * 128
            + (offs_n[:, None] * 128 * H + offs_0[None, :])
    )

    tl.store(gv_ptrs, dv)

    gk0_ptrs = (
            GK
            + (bid * L + nid * N) * H * 192
            + hid * 192
            + (offs_n[:, None] * 192 * H + offs_0[None, :])
    )
    tl.store(gk0_ptrs, dk0)

    gk1_ptrs = (
            GK
            + (bid * L + nid * N) * H * 192
            + hid * 192
            + 128
            + (offs_n[:, None] * 192 * H + offs_1[None, :])
    )
    tl.store(gk1_ptrs, dk1)


# ragged sum
@triton.jit
def mla_rs_kernel(
        Q,
        O,
        H: tl.constexpr,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
        CAUSAL: tl.constexpr
):
    bid = tl.program_id(0)
    L = tl.num_programs(1).to(tl.int64)
    lid = L - tl.program_id(1) - 1
    kid = tl.program_id(2)
    B = tl.num_programs(0)

    offs_n = tl.arange(0, BLOCK)

    # [L//N, B, L, H, 192】
    q_ptrs = (
            Q
            + bid * L * H * 192
            + lid * H * 192
            + kid * BLOCK
            + offs_n
    )
    o = tl.zeros((BLOCK,), dtype=tl.float32)
    if CAUSAL:
        steps = tl.cdiv(lid + 1, N)
    else:
        steps = L // N

    for i in range(steps):
        o += tl.load(q_ptrs + i * B * L * H * 192).to(tl.float32)

    o_ptrs = (
            O
            + bid * L * H * 192
            + lid * H * 192
            + kid * BLOCK
            + offs_n
    )

    tl.store(o_ptrs, o)


# should use triton>=3.5.1 for better performance
# hpc: high precision cache
def triton_mla_backward(go, o, q, k, v, lse, max_logits, causal=True, safe=True,
                        atomic=True, hpc=False, clip_value=None):
    # q: [B, L, H, 192]
    # k: [B, L, H, 192]
    # v: [B, L, H, 128]
    B, L, H, _ = q.shape
    assert k.size(1) == L

    device = q.device
    dtype = q.dtype

    ds = torch.empty((B, H, L), dtype=torch.float32, device=device)

    softmax_scale = 128 ** (-0.5)
    clip = clip_value is not None
    clip_value = clip_value * softmax_scale if clip else 0.0

    M = 64
    num_n_block = L // M
    num_warps = 4
    num_stages = 2
    grid = (B, H, num_n_block)
    mla_ds_kernel[grid](
        go,
        o,
        ds,
        L,
        M,
        num_warps=num_warps,
        num_stages=num_stages
    )

    M = 32
    N = 128
    if atomic:
        gq = torch.zeros((B, L, H, 192), dtype=torch.float32 if hpc else dtype,
                         device=device)
    else:
        gq = torch.empty((L // N, B, L, H, 192),
                         dtype=torch.float32 if hpc else dtype, device=device)

    gk = torch.empty((B, L, H, 192), dtype=dtype, device=device)
    gv = torch.empty((B, L, H, 128), dtype=dtype, device=device)
    assert L % M == 0
    assert L % N == 0
    assert N >= M
    num_n_block = L // N
    num_warps = 8
    num_stages = 5
    grid = (B, H, num_n_block)
    mla_backward_kernel[grid](
        go,
        q,
        k,
        v,
        gq,
        gk,
        gv,
        lse,
        max_logits,
        ds,
        softmax_scale,
        clip_value,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        L,
        M,
        N,
        atomic,
        causal,
        safe,
        clip,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if atomic:
        if hpc:
            gq = gq.to(q.dtype)
    else:
        qo = torch.empty((B, L, H, 192), dtype=dtype, device=device)
        BLOCK = max([x for x in [64, 1024, 2048, 4096] if H * 192 % x == 0])
        NB = H * 192 // BLOCK
        grid = (B, L, NB)
        num_warps = 2
        num_stages = 3
        mla_rs_kernel[grid](gq,
                            qo,
                            H,
                            N,
                            BLOCK,
                            causal,
                            num_warps=num_warps,
                            num_stages=num_stages,
                            )
        gq = qo
    return gq, gk, gv


@triton.jit
def varlen_mla_forward_kernel(
        Q,
        K,
        V,
        CU,
        PCU,
        Out,
        LSE,
        ML,
        softmax_scale,
        stride_q,
        stride_k,
        stride_v,
        T,
        clip_value,
        M: tl.constexpr,
        N: tl.constexpr,
        CAUSAL: tl.constexpr,
        PAD: tl.constexpr,
        SAFE: tl.constexpr,
        CLIP: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.num_programs(2) - tl.program_id(2) - 1
    H = tl.num_programs(1)

    if PAD:
        c01 = tl.load(CU + bid + tl.arange(0, 2))
        c0, c1 = tl.split(c01)
        length = c1 - c0
        pc01 = tl.load(PCU + bid + tl.arange(0, 2))
        pc0, pc1 = tl.split(pc01)
        padded_length = pc1 - pc0
        if mid + 1 > tl.cdiv(padded_length, M):
            return
    else:
        c01 = tl.load(CU + bid + tl.arange(0, 2))
        c0, c1 = tl.split(c01)
        length = c1 - c0
        if mid + 1 > tl.cdiv(length, M):
            return

    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)

    offs_0 = tl.arange(0, 128)
    offs_1 = tl.arange(0, 64)

    # [T, H, 192】
    q0_ptrs = (
            Q
            + (c0 + mid * M) * stride_q
            + hid * 192
            + (offs_m[:, None] * stride_q + offs_1[None, :])
    )

    k0_ptrs = (
            K
            + c0 * stride_k
            + hid * 192
            + (offs_n[:, None] * stride_k + offs_1[None, :])
    )

    v_ptrs = (
            V
            + c0 * stride_v
            + hid * 128
            + (offs_n[:, None] * stride_v + offs_0[None, :])
    )

    m_mask = (mid * M + offs_m) < length

    q0 = tl.load(q0_ptrs, mask=m_mask[:, None])
    q1 = tl.load(q0_ptrs + 64, mask=m_mask[:, None])
    q2 = tl.load(q0_ptrs + 128, mask=m_mask[:, None])

    acc_o = tl.zeros((M, 128), dtype=tl.float32)
    if SAFE:
        max_logits = tl.zeros((M,), dtype=tl.float32) - 10000.0
        lse = tl.zeros((M,), dtype=tl.float32)
    else:
        lse = tl.zeros((M,), dtype=tl.float32) + 1e-30

    if CAUSAL:
        steps = tl.cdiv(mid * M + M, N)
    else:
        steps = tl.cdiv(length, N)

    for i in range(0, steps):
        n = i * N
        n = tl.multiple_of(n, N)
        n_mask = (n + offs_n) < length

        k0 = tl.load(k0_ptrs + n * stride_k, mask=n_mask[:, None])
        k1 = tl.load(k0_ptrs + n * stride_k + 64, mask=n_mask[:, None])
        k2 = tl.load(k0_ptrs + n * stride_k + 128, mask=n_mask[:, None])

        qk = tl.dot(q0, tl.trans(k0))
        qk = tl.dot(q1, tl.trans(k1), qk)
        qk = tl.dot(q2, tl.trans(k2), qk)

        if CAUSAL:
            qk += tl.where(
                ((mid * M + offs_m)[:, None] >= (n + offs_n)[None, :]) & (
                n_mask[None, :]), 0.0, -1e9)
        else:
            qk += tl.where(n_mask[None, :], 0.0, -1e9)

        qk *= softmax_scale

        if SAFE:
            latest_max_logits = tl.maximum(max_logits, tl.max(qk, 1))
            p = tl.exp(qk - latest_max_logits[:, None])
            rescale = tl.exp(max_logits - latest_max_logits)
            lse = lse * rescale + tl.sum(p, 1)
            v = tl.load(v_ptrs + n * stride_v, mask=n_mask[:, None])
            p = p.to(V.dtype.element_ty)
            acc_o = acc_o * rescale[:, None]
            acc_o = tl.dot(p, v, acc_o)
            max_logits = latest_max_logits
        else:
            if CLIP:
                p = tl.exp(tl.minimum(qk, clip_value))
            else:
                p = tl.exp(qk)
            lse += tl.sum(p, 1)
            v = tl.load(v_ptrs + n * stride_v, mask=n_mask[:, None])
            p = p.to(V.dtype.element_ty)
            acc_o = tl.dot(p, v, acc_o)

    acc_o = acc_o / lse[:, None]

    # [T, H, 128]
    out_ptrs = (
            Out
            + (c0 + mid * M) * H * 128
            + hid * 128
            + (offs_m[:, None] * H * 128 + offs_0[None, :])
    )

    tl.store(out_ptrs, acc_o, mask=m_mask[:, None])
    # [H, T]
    tl.store(LSE + hid * T + c0 + mid * M + tl.arange(0, M), lse, mask=m_mask)
    if SAFE:
        tl.store(ML + hid * T + c0 + mid * M + tl.arange(0, M), max_logits,
                 mask=m_mask)


def triton_varlen_mla_forward(q, k, v, cu_seqlens, max_q_length,
                              padded_cu_seqlens=None, causal=True, safe=True,
                              clip_value=None):
    # q: [T, H, 192]
    # k: [T, H, 192]
    # v: [T, H, 128]
    T, H, _ = q.shape
    B = cu_seqlens.size(0) - 1
    assert k.size(0) == T
    M = 256
    N = 64
    assert M >= N

    o = torch.empty((T, H, 128), dtype=q.dtype, device=q.device)
    lse = torch.empty((H, T), dtype=torch.float32, device=q.device)
    max_logits = torch.empty((H, T), dtype=torch.float32, device=q.device)
    softmax_scale = 128 ** (-0.5)
    clip = clip_value is not None
    clip_value = clip_value * softmax_scale if clip else 0.0
    if clip and clip_value + math.log(max_q_length) < 88.7:
        safe = False
    PAD = padded_cu_seqlens is not None

    num_m_block = triton.cdiv(max_q_length, M)
    num_stages = 2
    num_warps = 8

    grid = (B, H, num_m_block)
    varlen_mla_forward_kernel[grid](
        q,
        k,
        v,
        cu_seqlens,
        padded_cu_seqlens,
        o,
        lse,
        max_logits,
        softmax_scale,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        T,
        clip_value,
        M,
        N,
        causal,
        PAD,
        safe,
        clip,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o, lse, max_logits


@triton.jit
def varlen_mla_backward_kernel(
        GO,
        Q,
        K,
        V,
        CU,
        PCU,
        GQ,
        GK,
        GV,
        LSE,
        ML,
        DS,
        softmax_scale,
        clip_value,
        stride_q,
        stride_k,
        stride_v,
        T,
        M: tl.constexpr,
        N: tl.constexpr,
        ATOMIC: tl.constexpr,
        CAUSAL: tl.constexpr,
        PAD: tl.constexpr,
        SAFE: tl.constexpr,
        CLIP: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    nid = tl.program_id(2)
    H = tl.num_programs(1).to(tl.int64)

    if PAD:
        c1 = tl.load(CU + bid + 1)
        c0 = tl.load(CU + bid)
        length = c1 - c0
        pc0 = tl.load(PCU + bid)
        pc1 = tl.load(PCU + bid + 1)
        padded_length = pc1 - pc0
        if nid + 1 > tl.cdiv(padded_length, N):
            return
    else:
        c1 = tl.load(CU + bid + 1)
        c0 = tl.load(CU + bid)
        length = c1 - c0
        if nid + 1 > tl.cdiv(length, N):
            return

    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)

    offs_0 = tl.arange(0, 128)  # nope
    offs_1 = tl.arange(0, 64)  # pe

    n_mask = (nid * N + offs_n) < length

    # [B, L, H, 192】
    q0_ptrs = (
            Q
            + c0 * stride_q
            + hid * 192
            + (offs_m[:, None] * stride_q + offs_0[None, :])
    )
    q1_ptrs = (
            Q
            + c0 * stride_q
            + hid * 192
            + 128
            + (offs_m[:, None] * stride_q + offs_1[None, :])
    )

    k0_ptrs = (
            K
            + (c0 + nid * N) * stride_k
            + hid * 192
            + (offs_n[:, None] * stride_k + offs_0[None, :])
    )
    k1_ptrs = (
            K
            + (c0 + nid * N) * stride_k
            + hid * 192
            + 128
            + (offs_n[:, None] * stride_k + offs_1[None, :])
    )
    k0 = tl.load(k0_ptrs, mask=n_mask[:, None])
    k1 = tl.load(k1_ptrs, mask=n_mask[:, None])

    v_ptrs = (
            V
            + (c0 + nid * N) * stride_v
            + hid * 128
            + (offs_n[:, None] * stride_v + offs_0[None, :])
    )
    v = tl.load(v_ptrs, mask=n_mask[:, None])

    go_ptrs = (
            GO
            + c0 * H * 128
            + hid * 128
            + (offs_m[:, None] * 128 * H + offs_0[None, :])
    )

    if ATOMIC:
        # [B, L, H, 192]
        dq0_ptrs = (
                GQ
                + c0 * H * 192
                + hid * 192
                + (offs_m[:, None] * H * 192 + offs_0[None, :])
        )
        dq1_ptrs = (
                GQ
                + c0 * H * 192
                + hid * 192
                + 128
                + (offs_m[:, None] * H * 192 + offs_1[None, :])
        )

    else:
        dq0_ptrs = (
                GQ
                + nid * T * H * 192
                + c0 * H * 192
                + hid * 192
                + (offs_m[:, None] * H * 192 + offs_0[None, :])
        )
        dq1_ptrs = (
                GQ
                + nid * T * H * 192
                + c0 * H * 192
                + hid * 192
                + 128
                + (offs_m[:, None] * H * 192 + offs_1[None, :])
        )

    dv = tl.zeros((N, 128), dtype=tl.float32)
    dk0 = tl.zeros((N, 128), dtype=tl.float32)
    dk1 = tl.zeros((N, 64), dtype=tl.float32)
    if CAUSAL:
        step = nid * N
        n_steps = tl.cdiv(length - step, M)
    else:
        step = 0
        n_steps = tl.cdiv(length, M)

    for i in range(n_steps):
        m = step + (n_steps - 1 - i) * M
        m_mask = (m + offs_m) < length

        # [H, T]
        lse = 1 / tl.load(LSE + hid * T + c0 + m + tl.arange(0, M), mask=m_mask,
                          other=1e30)
        if SAFE:
            max_logits = tl.load(ML + hid * T + c0 + m + tl.arange(0, M),
                                 mask=m_mask)
        ds = tl.load(DS + hid * T + c0 + m + tl.arange(0, M), mask=m_mask)

        q0 = tl.load(q0_ptrs + m * stride_q, mask=m_mask[:, None])
        q1 = tl.load(q1_ptrs + m * stride_q, mask=m_mask[:, None])
        go = tl.load(go_ptrs + m * H * 128, mask=m_mask[:, None])

        if CAUSAL:
            qk = tl.where(
                ((m + offs_m)[:, None] >= (nid * N + offs_n)[None, :]) & (
                n_mask[None, :]), 0.0, -1e9)
            qk = tl.dot(q1, tl.trans(k1), qk)
            qk = tl.dot(q0, tl.trans(k0), qk)
        else:
            qk = tl.dot(q1, tl.trans(k1))
            qk = tl.dot(q0, tl.trans(k0), qk)
            qk += tl.where((n_mask[None, :]), 0.0, -1e9)

        qk *= softmax_scale
        if CLIP:
            qk = tl.minimum(qk, clip_value)

        if SAFE:
            p = tl.exp(qk - max_logits[:, None]) * lse[:, None]
        else:
            p = tl.exp(qk) * lse[:, None]

        # impl 0
        dp = tl.dot(go, tl.trans(v))  # [M, 128]@[128, N]=[M,N]
        dp = p * (dp - ds[:, None]) * softmax_scale  # score
        # impl 1
        # dp = tl.zeros((1, N), dtype=tl.float32) - ds[:,None]
        # dp = tl.dot(go, tl.trans(v), dp)  # [M, 128]@[128, N]=[M,N]
        # dp = softmax_scale * dp * p  # score

        p = p.to(V.dtype.element_ty)
        dv = tl.dot(tl.trans(p), go, dv)  # [N, M]@[M, 128]=[N, 128]

        dp = dp.to(V.dtype.element_ty)
        dq0 = tl.dot(dp, k0)  # [M, N]@[N, 128]=[M, 128]
        dq1 = tl.dot(dp, k1)  # [M, N]@[N, 64]=[M, 64]
        if ATOMIC:
            tl.atomic_add(dq0_ptrs + m * H * 192, dq0, mask=m_mask[:, None],
                          sem='relaxed')
            tl.atomic_add(dq1_ptrs + m * H * 192, dq1, mask=m_mask[:, None],
                          sem='relaxed')
        else:
            tl.store(dq0_ptrs + m * H * 192, dq0, mask=m_mask[:, None])
            tl.store(dq1_ptrs + m * H * 192, dq1, mask=m_mask[:, None])

        dp = tl.trans(dp)
        dk0 = tl.dot(dp, q0, dk0)  # [N, M]@[M, 128]=[N, 128]
        dk1 = tl.dot(dp, q1, dk1)  # [N, M]@[M, 64]=[N, 64]

    gv_ptrs = (
            GV
            + (c0 + nid * N) * H * 128
            + hid * 128
            + (offs_n[:, None] * 128 * H + offs_0[None, :])
    )
    tl.store(gv_ptrs, dv, mask=n_mask[:, None])

    gk0_ptrs = (
            GK
            + (c0 + nid * N) * H * 192
            + hid * 192
            + (offs_n[:, None] * 192 * H + offs_0[None, :])
    )
    tl.store(gk0_ptrs, dk0, mask=n_mask[:, None])

    gk1_ptrs = (
            GK
            + (c0 + nid * N) * H * 192
            + hid * 192
            + 128
            + (offs_n[:, None] * 192 * H + offs_1[None, :])
    )
    tl.store(gk1_ptrs, dk1, mask=n_mask[:, None])


# ragged sum
@triton.jit
def varlen_mla_rs_kernel(
        Q,
        O,
        CU,
        B,
        PB: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        BLOCK: tl.constexpr,
        CAUSAL: tl.constexpr
):
    tid = tl.program_id(0)
    T = tl.num_programs(0).to(tl.int64)
    kid = tl.program_id(1)

    cu = tl.load(CU + tl.arange(0, PB), mask=tl.arange(0, PB) <= B)
    c0 = tl.max(tl.where(cu > tid, 0, cu), 0)
    c1 = tl.min(tl.where(cu <= c0, 2 ** 24, cu), 0)
    length = c1 - c0
    pid = tid - c0

    offs_n = tl.arange(0, BLOCK)

    # [max_q_length//N, T, H, 192]
    q_ptrs = (
            Q
            + tid * H * 192
            + kid * BLOCK
            + offs_n
    )
    o = tl.zeros((BLOCK,), dtype=tl.float32)
    if CAUSAL:
        steps = tl.cdiv(pid + 1, N)
    else:
        steps = tl.cdiv(length, N)

    for i in range(steps):
        o += tl.load(q_ptrs + i * T * H * 192).to(tl.float32)

    # [T, H, 192]
    o_ptrs = (
            O
            + tid * H * 192
            + kid * BLOCK
            + offs_n
    )

    tl.store(o_ptrs, o)


# should use triton>=3.5.1 for better performance
# hpc: high precision cache
def triton_varlen_mla_backward(go, o, q, k, v, lse,
                               max_logits, cu_seqlens, max_q_length,
                               padded_cu_seqlens=None, causal=True,
                               safe=True, hpc=False, atomic=True,
                               clip_value=None
                               ):
    # q: [T, H, 192]
    # k: [T, H, 192]
    # v: [T, H, 128]
    T, H, _ = q.shape
    assert k.size(0) == T
    B = cu_seqlens.size(0) - 1
    PADDED = padded_cu_seqlens is not None

    device = q.device
    dtype = q.dtype

    ds = torch.empty((H, T), dtype=torch.float32, device=device)

    softmax_scale = 128 ** (-0.5)
    clip = clip_value is not None
    clip_value = clip_value * softmax_scale if clip else 0.0

    M = 64
    num_warps = 4
    num_stages = 2
    grid = (1, H, triton.cdiv(T, M))
    mla_ds_kernel[grid](
        go,
        o,
        ds,
        T,
        M,
        num_warps=num_warps,
        num_stages=num_stages
    )

    M = 32
    N = 128
    num_n_block = triton.cdiv(T, N)
    if atomic:
        gq = torch.zeros((T, H, 192), dtype=torch.float32 if hpc else dtype,
                         device=device)
    else:
        gq = torch.empty((num_n_block, T, H, 192),
                         dtype=torch.float32 if hpc else dtype, device=q.device)

    gk = torch.empty((T, H, 192), dtype=dtype, device=device)
    gv = torch.empty((T, H, 128), dtype=dtype, device=device)
    assert N >= M
    num_warps = 8
    num_stages = 5
    grid = (B, H, num_n_block)
    varlen_mla_backward_kernel[grid](
        go,
        q,
        k,
        v,
        cu_seqlens,
        padded_cu_seqlens,
        gq,
        gk,
        gv,
        lse,
        max_logits,
        ds,
        softmax_scale,
        clip_value,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        T,
        M,
        N,
        atomic,
        causal,
        PADDED,
        safe,
        clip,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if atomic:
        if hpc:
            gq = gq.to(q.dtype)
    else:
        qo = torch.empty((T, H, 192), dtype=dtype, device=device)
        BLOCK = max([x for x in [64, 1024, 2048, 4096] if H * 192 % x == 0])
        NB = H * 192 // BLOCK
        grid = (T, NB)
        num_warps = 4
        num_stages = 3
        PB = max(triton.next_power_of_2(B), 128)
        varlen_mla_rs_kernel[grid](gq,
                                   qo,
                                   padded_cu_seqlens if PADDED else cu_seqlens,
                                   B,
                                   PB,
                                   H,
                                   N,
                                   BLOCK,
                                   causal,
                                   num_warps=num_warps,
                                   num_stages=num_stages,
                                   )
        gq = qo
    return gq, gk, gv


@triton.jit
def deprecated_fp8_mla_forward_kernel(
        Q,
        K,
        V,
        QS,
        KS,
        VS,
        Out,
        LSE,
        ML,
        softmax_scale,
        stride_q,
        stride_k,
        stride_v,
        L,
        M: tl.constexpr,
        N: tl.constexpr,
        CAUSAL: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.num_programs(2) - tl.program_id(2) - 1
    H = tl.num_programs(1)

    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)

    offs_0 = tl.arange(0, 128)  # nope
    offs_1 = tl.arange(0, 64)  # pe

    # [B, L, H, 192】
    q0_ptrs = (
            Q
            + (bid * L + mid * M) * stride_q
            + hid * 192
            + (offs_m[:, None] * stride_q + offs_0[None, :])
    )
    q1_ptrs = (
            Q
            + (bid * L + mid * M) * stride_q
            + hid * 192
            + 128
            + (offs_m[:, None] * stride_q + offs_1[None, :])
    )

    # [B, H, L]
    qs_ptrs = (
            QS
            + bid * H * L
            + hid * L
            + mid * M
            + offs_m
    )

    k0_ptrs = (
            K
            + bid * L * stride_k
            + hid * 192
            + (offs_n[:, None] * stride_k + offs_0[None, :])
    )

    k1_ptrs = (
            K
            + bid * L * stride_k
            + hid * 192
            + 128
            + (offs_n[:, None] * stride_k + offs_1[None, :])
    )

    # [B, H, L]
    ks_ptrs = (
            KS
            + bid * H * L
            + hid * L
            + offs_n
    )

    v_ptrs = (
            V
            + bid * L * stride_v
            + hid * 128
            + (offs_n[:, None] * stride_v + offs_0[None, :])
    )

    q0 = tl.load(q0_ptrs)
    q1 = tl.load(q1_ptrs)
    qs = tl.load(qs_ptrs)

    lse = tl.zeros((M,), dtype=tl.float32)
    acc_o = tl.zeros((M, 128), dtype=tl.float32)
    if CAUSAL:
        steps = tl.cdiv(mid * M + M, N)
    else:
        steps = L // N
    for i in range(0, steps):
        n = i * N
        n = tl.multiple_of(n, N)

        k1 = tl.load(k1_ptrs + n * stride_k)
        ks = tl.load(ks_ptrs + n)

        qk = tl.dot(q1, tl.trans(k1))

        k0 = tl.load(k0_ptrs + n * stride_k)

        qk = tl.dot(q0, tl.trans(k0), qk)

        qk += tl.where((mid * M + offs_m)[:, None] >= (n + offs_n)[None, :],
                       0.0, -1e9)
        qk = qk * qs[:, None] * ks[None, :]

        p = tl.exp(qk * softmax_scale)
        lse += tl.sum(p, 1)

        v = tl.load(v_ptrs + n * stride_v)
        p = p.to(V.dtype.element_ty)
        acc_o = tl.dot(p, v, acc_o)

    acc_o = acc_o / lse[:, None]

    # [B, L, H, 128]
    out_ptrs = (
            Out
            + (bid * L + mid * M) * H * 128
            + hid * 128
            + (offs_m[:, None] * 128 * H + offs_0[None, :])
    )

    tl.store(out_ptrs, acc_o)
    tl.store(LSE + bid * H * L + hid * L + mid * M + tl.arange(0, M), lse)


@triton.jit
def padding_fp8_mla_forward_kernel(
        Q,
        K,
        V,
        QS,
        KS,
        VS,
        Out,
        LSE,
        ML,
        softmax_scale,
        stride_q,
        stride_k,
        stride_v,
        L,
        M: tl.constexpr,
        N: tl.constexpr,
        CAUSAL: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.num_programs(2) - tl.program_id(2) - 1
    H = tl.num_programs(1)

    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)

    offs_0 = tl.arange(0, 256)
    offs_1 = tl.arange(0, 128)

    # [B, L, H, 192】
    q0_ptrs = (
            Q
            + (bid * L + mid * M) * stride_q
            + hid * 192
            + (offs_m[:, None] * stride_q + offs_0[None, :])
    )

    # [B, H, L]
    qs_ptrs = (
            QS
            + bid * H * L
            + hid * L
            + mid * M
            + offs_m
    )

    k0_ptrs = (
            K
            + bid * L * stride_k
            + hid * 192
            + (offs_n[:, None] * stride_k + offs_0[None, :])
    )

    # [B, H, L]
    ks_ptrs = (
            KS
            + bid * H * L
            + hid * L
            + offs_n
    )

    v_ptrs = (
            V
            + bid * L * stride_v
            + hid * 128
            + (offs_n[:, None] * stride_v + offs_1[None, :])
    )
    mask = offs_0 < 192
    q0 = tl.load(q0_ptrs, mask=mask[None, :])
    qs = tl.load(qs_ptrs)

    lse = tl.zeros((M,), dtype=tl.float32)
    acc_o = tl.zeros((M, 128), dtype=tl.float32)
    if CAUSAL:
        steps = tl.cdiv(mid * M + M, N)
    else:
        steps = L // N
    for i in range(0, steps):
        n = i * N
        n = tl.multiple_of(n, N)

        k0 = tl.load(k0_ptrs + n * stride_k, mask=mask[None, :])
        ks = tl.load(ks_ptrs + n)

        qk = tl.dot(q0, tl.trans(k0))

        qk += tl.where((mid * M + offs_m)[:, None] >= (n + offs_n)[None, :],
                       0.0, -1e9)
        qk = qk * qs[:, None] * ks[None, :]

        p = tl.exp(qk * softmax_scale)
        lse += tl.sum(p, 1)

        v = tl.load(v_ptrs + n * stride_v)
        p = p.to(V.dtype.element_ty)

        acc_o = tl.dot(p, v, acc_o)

    acc_o = acc_o / lse[:, None]

    # [B, L, H, 128]
    out_ptrs = (
            Out
            + (bid * L + mid * M) * H * 128
            + hid * 128
            + (offs_m[:, None] * 128 * H + offs_1[None, :])
    )

    tl.store(out_ptrs, acc_o)
    tl.store(LSE + bid * H * L + hid * L + mid * M + tl.arange(0, M), lse)


@triton.jit
def fp8_mla_forward_kernel(
        Q,
        K,
        V,
        QS,
        KS,
        VS,
        Out,
        LSE,
        ML,
        softmax_scale,
        stride_q,
        stride_k,
        stride_v,
        L,
        M: tl.constexpr,
        N: tl.constexpr,
        CAUSAL: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.num_programs(2) - tl.program_id(2) - 1
    H = tl.num_programs(1)

    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)

    offs_0 = tl.arange(0, 128)
    offs_1 = tl.arange(0, 64)

    # [B, L, H, 192]
    q0_ptrs = (
            Q
            + (bid * L + mid * M) * stride_q
            + hid * 192
            + (offs_m[:, None] * stride_q + offs_1[None, :])
    )

    # [B, H, L]
    qs_ptrs = (
            QS
            + bid * H * L
            + hid * L
            + mid * M
            + offs_m
    )

    k0_ptrs = (
            K
            + bid * L * stride_k
            + hid * 192
            + (offs_n[:, None] * stride_k + offs_1[None, :])
    )

    # [B, H, L]
    ks_ptrs = (
            KS
            + bid * H * L
            + hid * L
            + offs_n
    )

    v_ptrs = (
            V
            + bid * L * stride_v
            + hid * 128
            + (offs_n[:, None] * stride_v + offs_0[None, :])
    )

    if VS is not None:
        # [B, H, L]
        vs_ptrs = (
                VS
                + bid * H * L
                + hid * L
                + offs_n
        )

    q0 = tl.load(q0_ptrs)
    q1 = tl.load(q0_ptrs + 64)
    q2 = tl.load(q0_ptrs + 128)

    qs = tl.load(qs_ptrs) * softmax_scale

    lse = tl.zeros((M,), dtype=tl.float32)
    acc_o = tl.zeros((M, 128), dtype=tl.float32)
    if CAUSAL:
        steps = tl.cdiv(mid * M + M, N)
    else:
        steps = L // N

    for i in range(0, steps):
        n = i * N
        n = tl.multiple_of(n, N)

        k0 = tl.load(k0_ptrs + n * stride_k)
        k1 = tl.load(k0_ptrs + n * stride_k + 64)
        k2 = tl.load(k0_ptrs + n * stride_k + 128)
        ks = tl.load(ks_ptrs + n)

        if CAUSAL:
            qk = tl.where((mid * M + offs_m)[:, None] >= (n + offs_n)[None, :],
                          0.0, -1e9)
            qk = tl.dot(q0, tl.trans(k0), qk)
            qk = tl.dot(q1, tl.trans(k1), qk)
            qk = tl.dot(q2, tl.trans(k2), qk)
        else:
            qk = tl.dot(q0, tl.trans(k0))
            qk = tl.dot(q1, tl.trans(k1), qk)
            qk = tl.dot(q2, tl.trans(k2), qk)
        qk = qk * qs[:, None] * ks[None, :]

        p = tl.exp(qk)
        lse += tl.sum(p, 1)

        if VS is None:
            p = p.to(V.dtype.element_ty)
            v = tl.load(v_ptrs + n * stride_v)
            acc_o = tl.dot(p, v, acc_o)
        else:
            vs = tl.load(vs_ptrs)
            p = p * vs
            pm = tl.max(p, 1) / 448
            p = (p / pm[:, None]).to(V.dtype.element_ty)
            v = tl.load(v_ptrs + n * stride_v)
            acc_o = acc_o + tl.dot(p, v) * pm[:, None]

    tl.store(LSE + bid * H * L + hid * L + mid * M + tl.arange(0, M), lse)
    acc_o = acc_o / lse[:, None]
    # [B, L, H, 128]
    out_ptrs = (
            Out
            + (bid * L + mid * M) * H * 128
            + hid * 128
            + (offs_m[:, None] * 128 * H + offs_0[None, :])
    )
    tl.store(out_ptrs, acc_o)


def triton_fp8_mla_forward(q, k, v, qs, ks, vs=None, causal=True,
                           out_dtype=torch.bfloat16):
    # q: [B, L, H, 192]
    # k: [B, L, H, 192]
    # v: [B, L, H, 128]
    B, L, H, _ = q.shape
    assert k.size(1) == L
    M = 128
    N = 128
    assert L % M == 0
    assert L % N == 0
    assert M >= N

    o = torch.empty((B, L, H, 128), dtype=out_dtype, device=q.device)
    lse = torch.empty((B, H, L), dtype=torch.float32, device=q.device)
    max_logits = torch.empty((B, H, L), dtype=torch.float32, device=q.device)
    softmax_scale = 128 ** (-0.5)

    num_m_block = L // M
    num_warps = 8
    num_stages = 3

    grid = (B, H, num_m_block)
    fp8_mla_forward_kernel[grid](
        q,
        k,
        v,
        qs,
        ks,
        vs,
        o,
        lse,
        max_logits,
        softmax_scale,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        L,
        M,
        N,
        causal,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o, lse, max_logits
