# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math

import torch
import triton
import triton.language as tl



@triton.jit
def _mla_forward_kernel(
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
    steps = tl.cdiv(mid * M + M, N)

    for i in range(0, steps):
        n = i * N
        n = tl.multiple_of(n, N)

        k1 = tl.load(k1_ptrs + n * stride_k)

        qk = tl.dot(q1, tl.trans(k1))

        k0 = tl.load(k0_ptrs + n * stride_k)

        qk = tl.dot(q0, tl.trans(k0), qk)

        qk += tl.where( (mid * M + offs_m)[:, None] >= (n + offs_n)[None, :], 0.0, float("-inf"))

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
    # tl.store(ML + bid * H * L + hid * L + mid * M + tl.arange(0, M), tl.zeros((M,), dtype=tl.float32))



@triton.jit
def mla_forward_kernel(
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

    lse = tl.zeros((M,), dtype=tl.float32)
    acc_o = tl.zeros((M, 128), dtype=tl.float32)
    steps = tl.cdiv(mid * M + M, N)

    for i in range(0, steps):
        n = i * N
        n = tl.multiple_of(n, N)

        # qk = tl.where( (mid * M + offs_m)[:, None] >= (n + offs_n)[None, :], 0.0, -10000.0)
        k0 = tl.load(k0_ptrs + n * stride_k)

        qk = tl.dot(q0, tl.trans(k0))

        k1 = tl.load(k0_ptrs + n * stride_k + 64)

        qk = tl.dot(q1, tl.trans(k1), qk)

        k2 = tl.load(k0_ptrs + n * stride_k + 128)

        qk = tl.dot(q2, tl.trans(k2), qk)

        qk += tl.where( (mid * M + offs_m)[:, None] >= (n + offs_n)[None, :], 0.0, float("-inf"))

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
    # tl.store(ML + bid * H * L + hid * L + mid * M + tl.arange(0, M), tl.zeros((M,), dtype=tl.float32))


def triton_mla_forward(q, k, v):
    # q: [B, L, H, 192]
    # k: [B, L, H, 192]
    # v: [B, L, H, 128]
    B, L, H, _ = q.shape
    assert k.size(1) == L
    M = 256
    N = 64  # H20:64 H800:128
    assert L % M == 0
    assert L % N == 0
    assert M >= N

    o = torch.empty((B, L, H, 128), dtype=q.dtype, device=q.device)
    lse = torch.empty((B, H, L), dtype=torch.float32, device=q.device)
    max_logits = torch.empty((B, H, L), dtype=torch.float32, device=q.device)
    softmax_scale = 1.0 / math.sqrt(128)

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
        q.stride(1),
        k.stride(1),
        v.stride(1),
        L,
        M,
        N,
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

        qk += tl.where( (mid * M + offs_m)[:, None] >= (n + offs_n)[None, :], 0.0, float("-inf"))

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
    mid = tl.num_programs(2) - tl.program_id(2) - 1
    H = tl.num_programs(1)

    offs_m = tl.arange(0, M)  
    offs_0 = tl.arange(0, 128)  # nope

    # [B, L, H, 128】
    offs = ((bid * L + mid * M) * H * 128
            + hid * 128
            + (offs_m[:, None] * H * 128 + offs_0[None, :])
        )

    g = tl.load(G + offs).to(tl.float32)
    o = tl.load(O + offs).to(tl.float32)
    ds = tl.sum(g*o, 1)
    tl.store(DS + bid * H * L + hid * L + mid * M + tl.arange(0, M), ds)


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
    stride_q,
    stride_k,
    stride_v,
    L,
    M: tl.constexpr,
    N: tl.constexpr,
    ATOMIC: tl.constexpr,
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
    for m in range(nid * N, L, M):
        # max_logits = tl.load(ML + bid * H * L + hid * L + mid * M + tl.arange(0, M))
        lse = tl.load(LSE + bid * H * L + hid * L + m + tl.arange(0, M))
        ds = tl.load(DS + bid * H * L + hid * L + m + tl.arange(0, M))

        q0 = tl.load(q0_ptrs + m * stride_q)
        q1 = tl.load(q1_ptrs + m * stride_q)
        go = tl.load(go_ptrs + m * H * 128)

        qk = tl.where( (m + offs_m)[:, None] >= (nid * N + offs_n)[None, :], 0.0, -10000.0)
        qk = tl.dot(q1, tl.trans(k1), qk)
        qk = tl.dot(q0, tl.trans(k0), qk)

        # qk += tl.where( (m + offs_m)[:, None] >= (nid * N + offs_n)[None, :], 0.0, float("-inf"))
        p = tl.exp(qk * softmax_scale)/lse[:, None]

        dp = tl.dot(go, tl.trans(v))  # [M, 128]@[128, N]=[M,N]
        dp = p * (dp - ds[:,None]) * softmax_scale  # score
        # dp = -p * ds[:,None]
        # dp = tl.dot(go, tl.trans(v), dp)  # [M, 128]@[128, N]=[M,N]
        # dp *= softmax_scale  # score

        p = p.to(V.dtype.element_ty)
        dv = tl.dot(tl.trans(p), go, dv)  # [N, M]@[M, 128]=[N, 128]
        
        dp = dp.to(V.dtype.element_ty)
        dk0 = tl.dot(tl.trans(dp), q0, dk0)  # [N, M]@[M, 128]=[N, 128]
        dk1 = tl.dot(tl.trans(dp), q1, dk1)  # [N, M]@[M, 64]=[N, 64]
        dq0 = tl.dot(dp, k0)  # [M, N]@[N, 128]=[M, 128]
        dq1 = tl.dot(dp, k1)  # [M, N]@[N, 64]=[M, 64]
        if ATOMIC:
            tl.atomic_add(dq0_ptrs + m * H * 192, dq0)
            tl.atomic_add(dq1_ptrs + m * H * 192, dq1)
        else:
            tl.store(dq0_ptrs + m * H * 192, dq0)
            tl.store(dq1_ptrs + m * H * 192, dq1)

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



@triton.jit
def depracated_mla_backward_kernel(
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
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    nid = tl.num_programs(2) - tl.program_id(2) - 1
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

    # dq0_ptrs = (
    #     GQ
    #     + bid * L * H * 192
    #     + hid * 192
    #     + (offs_m[:, None] * H * 192 + offs_0[None, :])
    # )

    dq0_ptrs = (
        GQ
        + nid * B * L * H * 192
        + bid * L * H * 192
        + hid * 192
        + (offs_m[:, None] * H * 192 + offs_1[None, :])
    )

    dv = tl.zeros((N, 128), dtype=tl.float32)
    dk0 = tl.zeros((N, 64), dtype=tl.float32)
    dk1 = tl.zeros((N, 64), dtype=tl.float32)
    dk2 = tl.zeros((N, 64), dtype=tl.float32)
    for m in range(nid * N, L, M):
        lse = tl.load(LSE + bid * H * L + hid * L + m + tl.arange(0, M))
        # max_logits = tl.load(ML + bid * H * L + hid * L + mid * M + tl.arange(0, M))
        ds = tl.load(DS + bid * H * L + hid * L + m + tl.arange(0, M))

        q0 = tl.load(q0_ptrs + m * stride_q)
        qk = tl.dot(q0, tl.trans(k0))

        q1 = tl.load(q0_ptrs + m * stride_q + 64)
        qk = tl.dot(q1, tl.trans(k1), qk)

        q2 = tl.load(q0_ptrs + m * stride_q + 128)
        qk = tl.dot(q2, tl.trans(k2), qk)

        go = tl.load(go_ptrs + m * H * 128)

        qk += tl.where( (m + offs_m)[:, None] >= (nid * N + offs_n)[None, :], 0.0, float("-inf"))
        p = tl.exp(qk * softmax_scale)/lse[:, None]

        dp = tl.dot(go, tl.trans(v))  # [M, 128]@[128, N]=[M,N]
        dp = p * (dp - ds[:,None]) * softmax_scale  # score

        p = p.to(V.dtype.element_ty)
        dv = tl.dot(tl.trans(p), go, dv)  # [N, M]@[M, 128]=[N, 128]
        
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
        + (offs_n[:, None] * 128 * H + offs_0[None, :])
    )

    tl.store(gv_ptrs, dv)

    gk0_ptrs = (
        GK
        + (bid * L + nid * N) * H * 192
        + hid * 192
        + (offs_n[:, None] * 192 * H + offs_1[None, :])
    )
    tl.store(gk0_ptrs, dk0)
    tl.store(gk0_ptrs + 64, dk1)
    tl.store(gk0_ptrs + 128, dk2)


# ragged sum
@triton.jit
def mla_rs_kernel(
    Q,
    O,
    H: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr
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
    o = tl.zeros((BLOCK, ), dtype=tl.float32)
    for i in range(tl.cdiv(lid + 1, N)):
        o += tl.load(q_ptrs + i * B * L * H * 192)

    o_ptrs = (
        O
        + bid * L * H * 192
        + lid * H * 192
        + kid * BLOCK 
        + offs_n
    )

    tl.store(o_ptrs, o)

# should use triton>=3.5.1 for better performance
def triton_mla_backward(go, o, q, k, v, lse, max_logits):
    # q: [B, L, H, 192]
    # k: [B, L, H, 192]
    # v: [B, L, H, 128]
    B, L, H, _ = q.shape
    assert k.size(1) == L

    device = q.device
    dtype = q.dtype

    ds = torch.empty((B, H, L), dtype=torch.float32, device=device)

    softmax_scale = 1.0 / math.sqrt(128)

    native = False
    if native:
        M = 128
        N = 64
        assert L % M == 0
        assert L % N == 0
        assert M >= N
        num_m_block = L // M
        num_warps = 8
        num_stages = 2
        grid = (B, H, num_m_block)
        naive_mla_ds_kernel[grid](
            go,
            q,
            k,
            v,
            lse,
            max_logits,
            ds,
            softmax_scale,
            q.stride(1),
            k.stride(1),
            v.stride(1),
            L,
            M,
            N,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        M = 64
        assert L % M == 0
        num_m_block = L // M
        num_warps = 4
        num_stages = 2
        grid = (B, H, num_m_block)
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
    atomic = False
    if atomic:
        gq = torch.zeros((B, L, H, 192), dtype=torch.float32, device=device)
    else:
        gq = torch.empty((L//N, B, L, H, 192), dtype=torch.float32, device=q.device)
    
    gk = torch.empty((B, L, H, 192), dtype=dtype, device=device)
    gv = torch.empty((B, L, H, 128), dtype=dtype, device=device)
    assert L % M == 0
    assert L % N == 0
    assert N >= M
    num_n_block = L // N
    num_warps = 8
    num_stages = 3
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
        q.stride(1),
        k.stride(1),
        v.stride(1),
        L,
        M,
        N,
        atomic,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if atomic:
        gq = gq.to(q.dtype)
    else:
        qo = torch.empty((B, L, H, 192), dtype=dtype, device=device)
        BLOCK = max([x for x in [64,1024,2048,4096] if H * 192 % x == 0])
        NB = H * 192 // BLOCK
        grid = (B, L, NB)
        num_warps = 4
        num_stages = 3
        mla_rs_kernel[grid](gq,
                            qo,
                            H,
                            N,
                            BLOCK,
                            num_warps=num_warps,
                            num_stages=num_stages,
                            )
        gq = qo
    return gq, gk, gv

