# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fp32_lightning_attention_forward_kernel(
    Q,
    K,
    V,
    S,
    Out,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_s,
    decay_scales,
    L,
    D: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    N = D // VD
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)

    c0 = bid * L

    decay_scale = -tl.load(decay_scales + hid)

    offs_b = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, KD)
    offs_v = tl.arange(0, VD)

    q_ptrs = (
        Q
        + c0 * stride_q
        + hid * D
        + kid * KD
        + (offs_b[:, None] * stride_q + offs_k[None, :])
    )
    k_ptrs = (
        K
        + c0 * stride_k
        + hid * D
        + kid * KD
        + (offs_b[:, None] * stride_k + offs_k[None, :])
    )
    v_ptrs = (
        V
        + c0 * stride_v
        + hid * D
        + vid * VD
        + (offs_b[:, None] * stride_v + offs_v[None, :])
    )
    out_ptrs = (
        Out
        + c0 * D * H
        + hid * D
        + vid * VD
        + (offs_b[:, None] * H * D + offs_v[None, :])
    )
    s_ptrs = (
        S
        + bid * stride_s
        + hid * D * D
        + kid * D * KD
        + vid * VD
        + (offs_k[:, None] * D + offs_v[None, :])
    )
    state = tl.zeros((KD, VD), dtype=tl.float32)
    block_decay = tl.exp(decay_scale * BLOCK)

    for n in range(0, L, BLOCK):
        n = tl.multiple_of(n, BLOCK)

        q = tl.load(q_ptrs + n * stride_q)  # .to(tl.float32)
        k = tl.trans(tl.load(k_ptrs + n * stride_k))  # .to(tl.float32)
        v = tl.load(v_ptrs + n * stride_v).to(tl.float32)
        b_offs = BLOCK - 1 - offs_b
        decays = tl.exp(decay_scale * b_offs)
        inv_decays = 1 / decays

        q = q * inv_decays[:, None]
        k = k * decays[None, :]
        qk = tl.dot(q, k) * softmax_scale
        qk = tl.where(offs_b[None, :] <= offs_b[:, None], qk, 0.0)
        o = tl.dot(qk, v)

        o = tl.dot(q, state) * block_decay * softmax_scale + o

        state = state * block_decay + tl.dot(k, v)

        if KD == D:
            tl.store(out_ptrs + n * H * D, o.to(Out.dtype.element_ty))
        else:
            tl.atomic_add(
                out_ptrs + n * H * D, o.to(Out.dtype.element_ty), sem="relaxed"
            )

    tl.store(s_ptrs, state)


@triton.jit
def lightning_attention_forward_kernel(
    Q,
    K,
    V,
    S,
    Out,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_s,
    decay_scales,
    L,
    D: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    N = D // VD
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)

    c0 = bid * L

    decay_scale = -tl.load(decay_scales + hid)

    offs_b = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, KD)
    offs_v = tl.arange(0, VD)

    q_ptrs = (
        Q
        + c0 * stride_q
        + hid * D
        + kid * KD
        + (offs_b[:, None] * stride_q + offs_k[None, :])
    )
    k_ptrs = (
        K
        + c0 * stride_k
        + hid * D
        + kid * KD
        + (offs_b[:, None] * stride_k + offs_k[None, :])
    )
    v_ptrs = (
        V
        + c0 * stride_v
        + hid * D
        + vid * VD
        + (offs_b[:, None] * stride_v + offs_v[None, :])
    )
    out_ptrs = (
        Out
        + c0 * D * H
        + hid * D
        + vid * VD
        + (offs_b[:, None] * H * D + offs_v[None, :])
    )
    s_ptrs = (
        S
        + bid * stride_s
        + hid * D * D
        + kid * D * KD
        + vid * VD
        + (offs_k[:, None] * D + offs_v[None, :])
    )
    state = tl.zeros((KD, VD), dtype=tl.float32)
    block_decay = tl.exp(decay_scale * BLOCK)
    mask = tl.exp(decay_scale * (offs_b[:, None] - offs_b[None, :]))
    mask = tl.where(offs_b[None, :] <= offs_b[:, None], mask, 0.0) * softmax_scale
    b_offs = BLOCK - 1 - offs_b
    decays = tl.exp(decay_scale * b_offs)
    inv_decays = 1 / decays * block_decay * softmax_scale

    for n in range(0, L, BLOCK):
        n = tl.multiple_of(n, BLOCK)

        q = tl.load(q_ptrs + n * stride_q)
        k = tl.trans(tl.load(k_ptrs + n * stride_k))
        v = tl.load(v_ptrs + n * stride_v)

        qk = tl.dot(q, k) * mask
        o = tl.dot(qk.to(v.dtype), v)

        o = tl.dot((q * inv_decays[:, None]).to(q.dtype), state.to(q.dtype), o)

        state *= block_decay
        state = tl.dot((k * decays[None, :]).to(v.dtype), v, state)

        if KD == D:
            tl.store(out_ptrs + n * H * D, o.to(Out.dtype.element_ty))
        else:
            tl.atomic_add(
                out_ptrs + n * H * D, o.to(Out.dtype.element_ty), sem="relaxed"
            )

    tl.store(s_ptrs, state)


# (k_dim_block, length, qo_heads, d)
@triton.jit
def _output_sum_kernel(T, O, DIM: tl.constexpr, NUM_BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    length = tl.num_programs(0)
    x = tl.zeros((DIM,), dtype=tl.float32)
    for i in range(NUM_BLOCK):
        x += tl.load(T + i * length * DIM + pid * DIM + tl.arange(0, DIM)).to(
            tl.float32
        )
    tl.store(O + pid * DIM + tl.arange(0, DIM), x)


def triton_lightning_attention_forward(
    q, k, v, decay_scales, hpc=False, hp=False, softmax_scale=None
):
    B, L, H, D = q.shape
    h = k.shape[2]
    assert H == h, "triton_lightning_attention_forward does NOT support GQA currently"

    if softmax_scale is None:
        softmax_scale = D ** (-0.5)

    KD = 32
    VD = 128
    BLOCK = 32
    device = q.device
    dtype = q.dtype

    num_warps = 2  # 2
    num_stages = 5  # 3

    k_dim_block = D // KD
    v_dim_block = D // VD
    if k_dim_block == 1:
        outputs = torch.empty((B, L, H, D), device=device, dtype=dtype)
    else:
        outputs = torch.zeros(
            (B, L, H, D), device=device, dtype=torch.float32 if hpc else dtype
        )

    s = torch.empty((B, H, D, D), device=device, dtype=torch.float32)
    assert L % BLOCK == 0 and BLOCK <= 64

    kernel = (
        fp32_lightning_attention_forward_kernel
        if hp
        else lightning_attention_forward_kernel
    )
    grid = (B, H, k_dim_block * v_dim_block)
    kernel[grid](
        q,
        k,
        v,
        s,
        outputs,
        softmax_scale,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        s.stride(0),
        decay_scales,
        L,
        D=D,
        KD=KD,
        VD=VD,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if k_dim_block > 1 and hpc:
        o = outputs.to(dtype)
    else:
        o = outputs

    return o, s


@triton.jit
def fp32_lightning_attention_q_backward_kernel(
    Q,
    K,
    V,
    G,
    DQ,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_g,
    decay_scales,
    L,
    D: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    N = D // VD
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)

    c0 = bid * L

    decay_scale = -tl.load(decay_scales + hid)

    offs_b = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, KD)
    offs_v = tl.arange(0, VD)

    q_ptrs = (
        Q
        + c0 * stride_q
        + hid * D
        + kid * KD
        + (offs_b[:, None] * stride_q + offs_k[None, :])
    )
    k_ptrs = (
        K
        + c0 * stride_k
        + hid * D
        + kid * KD
        + (offs_b[:, None] * stride_k + offs_k[None, :])
    )
    v_ptrs = (
        V
        + c0 * stride_v
        + hid * D
        + vid * VD
        + (offs_b[:, None] * stride_v + offs_v[None, :])
    )
    g_ptrs = (
        G
        + c0 * D * H
        + hid * D
        + vid * VD
        + (offs_b[:, None] * stride_g + offs_v[None, :])
    )
    dq_ptrs = (
        DQ
        + c0 * D * H
        + hid * D
        + kid * KD
        + (offs_b[:, None] * H * D + offs_k[None, :])
    )

    state = tl.zeros((KD, VD), dtype=tl.float32)
    mask = tl.exp((offs_b[:, None] - offs_b[None, :]) * decay_scale)
    mask = tl.where(offs_b[None, :] <= offs_b[:, None], mask, 0.0)

    n_steps = tl.cdiv(L, BLOCK)
    for i in range(n_steps):
        n = i * BLOCK
        n = tl.multiple_of(n, BLOCK)

        q = tl.load(q_ptrs + n * stride_q).to(tl.float32)
        k = tl.load(k_ptrs + n * stride_k).to(tl.float32)
        v = tl.load(v_ptrs + n * stride_v).to(tl.float32)
        g = tl.load(g_ptrs + n * stride_g).to(tl.float32)

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        qk *= mask

        decay_offs = BLOCK - 1 - offs_b

        block_decay = tl.exp(decay_scale * BLOCK)
        decays = tl.exp(decay_scale * decay_offs)  # [0.01, 0.1, 1]

        state = state * block_decay

        dqk = tl.dot(g, tl.trans(v)) * mask * softmax_scale

        dq = (
            tl.dot(dqk, k)
            + tl.dot(g * decays[:, None], tl.trans(state)) * softmax_scale
        )

        if VD == D:
            tl.store(dq_ptrs + n * H * D, dq.to(DQ.dtype.element_ty))
        else:
            tl.atomic_add(
                dq_ptrs + n * H * D, dq.to(DQ.dtype.element_ty), sem="relaxed"
            )

        state = state + tl.dot(tl.trans(k * decays[:, None]), v)


@triton.jit
def lightning_attention_q_backward_kernel(
    Q,
    K,
    V,
    G,
    DQ,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_g,
    decay_scales,
    L,
    D: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    N = D // VD
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)

    c0 = bid * L

    decay_scale = -tl.load(decay_scales + hid)

    offs_b = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, KD)
    offs_v = tl.arange(0, VD)

    k_ptrs = (
        K
        + c0 * stride_k
        + hid * D
        + kid * KD
        + (offs_b[:, None] * stride_k + offs_k[None, :])
    )
    v_ptrs = (
        V
        + c0 * stride_v
        + hid * D
        + vid * VD
        + (offs_b[:, None] * stride_v + offs_v[None, :])
    )
    g_ptrs = (
        G
        + c0 * D * H
        + hid * D
        + vid * VD
        + (offs_b[:, None] * stride_g + offs_v[None, :])
    )
    dq_ptrs = (
        DQ
        + c0 * D * H
        + hid * D
        + kid * KD
        + (offs_b[:, None] * H * D + offs_k[None, :])
    )

    state = tl.zeros((KD, VD), dtype=tl.float32)
    mask = tl.exp((offs_b[:, None] - offs_b[None, :]) * decay_scale)
    mask = tl.where(offs_b[None, :] <= offs_b[:, None], mask, 0.0) * softmax_scale

    decay_offs = BLOCK - 1 - offs_b

    block_decay = tl.exp(decay_scale * BLOCK)
    decays = tl.exp(decay_scale * decay_offs)  # [0.01, 0.1, 1]

    n_steps = tl.cdiv(L, BLOCK)
    for i in range(n_steps):
        n = i * BLOCK
        n = tl.multiple_of(n, BLOCK)

        # q = tl.load(q_ptrs + n * stride_q)
        k = tl.load(k_ptrs + n * stride_k)
        v = tl.load(v_ptrs + n * stride_v)
        g = tl.load(g_ptrs + n * stride_g)

        # qk = tl.dot(q, tl.trans(k)) * mask
        # qk *= mask

        state = state * block_decay

        dqk = tl.dot(g, tl.trans(v)) * mask

        dq = (
            tl.dot(dqk.to(k.dtype), k)
            + tl.dot(g * decays[:, None], tl.trans(state)) * softmax_scale
        )

        if VD == D:
            tl.store(dq_ptrs + n * H * D, dq.to(DQ.dtype.element_ty))
        else:
            tl.atomic_add(
                dq_ptrs + n * H * D, dq.to(DQ.dtype.element_ty), sem="relaxed"
            )

        state = state + tl.dot((tl.trans(k) * decays[None, :]).to(v.dtype), v)


@triton.jit
def fp32_lightning_attention_kv_backward_kernel(
    Q,
    K,
    V,
    G,
    DK,
    DV,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_g,
    decay_scales,
    L,
    D: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    N = D // VD
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)

    c0 = bid * L

    decay_scale = -tl.load(decay_scales + hid)

    offs_b = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, KD)
    offs_v = tl.arange(0, VD)

    q_ptrs = (
        Q
        + c0 * stride_q
        + hid * D
        + kid * KD
        + (offs_b[:, None] * stride_q + offs_k[None, :])
    )
    k_ptrs = (
        K
        + c0 * stride_k
        + hid * D
        + kid * KD
        + (offs_b[:, None] * stride_k + offs_k[None, :])
    )
    v_ptrs = (
        V
        + c0 * stride_v
        + hid * D
        + vid * VD
        + (offs_b[:, None] * stride_v + offs_v[None, :])
    )
    g_ptrs = (
        G
        + c0 * D * H
        + hid * D
        + vid * VD
        + (offs_b[:, None] * stride_g + offs_v[None, :])
    )

    dk_ptrs = (
        DK
        + c0 * H * D
        + hid * D
        + kid * KD
        + (offs_b[:, None] * H * D + offs_k[None, :])
    )
    dv_ptrs = (
        DV
        + c0 * H * D
        + hid * D
        + vid * VD
        + (offs_b[:, None] * H * D + offs_v[None, :])
    )

    gs = tl.zeros((KD, VD), dtype=tl.float32)

    n_steps = tl.cdiv(L, BLOCK)
    for i in range(n_steps):
        n = (n_steps - i - 1) * BLOCK
        n = tl.multiple_of(n, BLOCK)

        q = tl.load(q_ptrs + n * stride_q).to(tl.float32)
        k = tl.load(k_ptrs + n * stride_k).to(tl.float32)
        v = tl.load(v_ptrs + n * stride_v).to(tl.float32)
        g = tl.load(g_ptrs + n * stride_g).to(tl.float32)
        b = BLOCK
        b_offs = b - 1 - offs_b

        block_decay = tl.exp(decay_scale * b)
        amps = tl.exp(-decay_scale * b_offs)  # [100, 10, 1]
        decays = tl.exp(decay_scale * b_offs)  # [0.01, 0.1, 1]

        qs = q * amps[:, None]  # [100, 10, 1]
        ks = k * decays[:, None]  # [0.01, 0.1, 1]
        qk = tl.dot(qs, tl.trans(ks)) * softmax_scale
        qk = tl.where(offs_b[None, :] <= offs_b[:, None], qk, 0.0)

        dv = tl.dot(tl.trans(qk), g)

        dv += tl.dot(ks, gs)

        dqk = tl.dot(g, tl.trans(v))
        dqk = tl.where(offs_b[None, :] <= offs_b[:, None], dqk, 0.0) * softmax_scale
        dk = tl.dot(tl.trans(dqk), qs)
        dk += tl.dot(v, tl.trans(gs))
        dk *= decays[:, None]

        gs *= block_decay
        gs += tl.dot(tl.trans(qs), g) * softmax_scale * block_decay

        if VD == D:
            tl.store(dk_ptrs + n * H * D, dk.to(DK.dtype.element_ty))
        else:
            tl.atomic_add(
                dk_ptrs + n * H * D, dk.to(DK.dtype.element_ty), sem="relaxed"
            )
        if KD == D:
            tl.store(dv_ptrs + n * H * D, dv.to(DV.dtype.element_ty))
        else:
            tl.atomic_add(
                dv_ptrs + n * H * D, dv.to(DV.dtype.element_ty), sem="relaxed"
            )


@triton.jit
def lightning_attention_kv_backward_kernel(
    Q,
    K,
    V,
    G,
    DK,
    DV,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_g,
    decay_scales,
    L,
    D: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    N = D // VD
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)

    c0 = bid * L

    decay_scale = -tl.load(decay_scales + hid)

    offs_b = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, KD)
    offs_v = tl.arange(0, VD)

    q_ptrs = (
        Q
        + c0 * stride_q
        + hid * D
        + kid * KD
        + (offs_b[:, None] * stride_q + offs_k[None, :])
    )
    k_ptrs = (
        K
        + c0 * stride_k
        + hid * D
        + kid * KD
        + (offs_b[:, None] * stride_k + offs_k[None, :])
    )
    v_ptrs = (
        V
        + c0 * stride_v
        + hid * D
        + vid * VD
        + (offs_b[:, None] * stride_v + offs_v[None, :])
    )
    g_ptrs = (
        G
        + c0 * D * H
        + hid * D
        + vid * VD
        + (offs_b[:, None] * stride_g + offs_v[None, :])
    )

    dk_ptrs = (
        DK
        + c0 * H * D
        + hid * D
        + kid * KD
        + (offs_b[:, None] * H * D + offs_k[None, :])
    )
    dv_ptrs = (
        DV
        + c0 * H * D
        + hid * D
        + vid * VD
        + (offs_b[:, None] * H * D + offs_v[None, :])
    )

    gs = tl.zeros((KD, VD), dtype=tl.float32)

    b_offs = BLOCK - 1 - offs_b

    block_decay = tl.exp(decay_scale * BLOCK)
    amps = tl.exp(-decay_scale * b_offs)  # [100, 10, 1]
    # decays = tl.exp(decay_scale * b_offs)  # [0.01, 0.1, 1]
    decays = 1 / amps  # [0.01, 0.1, 1]
    sd = softmax_scale * block_decay

    mask = tl.exp((offs_b[:, None] - offs_b[None, :]) * decay_scale)
    mask = tl.where(offs_b[None, :] <= offs_b[:, None], mask, 0.0) * softmax_scale

    n_steps = tl.cdiv(L, BLOCK)
    for i in range(n_steps):
        n = (n_steps - i - 1) * BLOCK
        n = tl.multiple_of(n, BLOCK)

        q = tl.load(q_ptrs + n * stride_q)
        k = tl.load(k_ptrs + n * stride_k)
        v = tl.load(v_ptrs + n * stride_v)
        g = tl.load(g_ptrs + n * stride_g)

        # qs = q * amps[:, None]   # [100, 10, 1]
        # ks = k * decays[:, None]  # [0.01, 0.1, 1]
        qk = tl.dot(q, tl.trans(k)) * mask

        dv = tl.dot(tl.trans(qk).to(g.dtype), g)

        dv += tl.dot(k * decays[:, None], gs)

        dqk = (tl.dot(g, tl.trans(v)) * mask).to(q.dtype)
        dk = tl.dot(tl.trans(dqk), (q * amps[:, None]).to(q.dtype))
        dk = tl.dot(v, tl.trans(gs.to(v.dtype)), dk)
        dk *= decays[:, None]

        gs *= block_decay
        gs += tl.dot(tl.trans(q * amps[:, None]).to(g.dtype), g) * sd

        if VD == D:
            tl.store(dk_ptrs + n * H * D, dk.to(DK.dtype.element_ty))
        else:
            tl.atomic_add(
                dk_ptrs + n * H * D, dk.to(DK.dtype.element_ty), sem="relaxed"
            )
        if KD == D:
            tl.store(dv_ptrs + n * H * D, dv.to(DV.dtype.element_ty))
        else:
            tl.atomic_add(
                dv_ptrs + n * H * D, dv.to(DV.dtype.element_ty), sem="relaxed"
            )


def triton_lightning_attention_backward(
    output_grad, q, k, v, decay_scales, softmax_scale=None, hpc=False, hp=False
):
    B, L, H, D = q.shape
    if softmax_scale is None:
        softmax_scale = D ** (-0.5)

    dtype = q.dtype
    device = q.device

    KD = 64
    VD = 128
    BLOCK = 32
    k_dim_block = D // KD
    v_dim_block = D // VD
    if v_dim_block > 1:
        dq = torch.zeros(
            (B, L, H, D), device=device, dtype=torch.float32 if hpc else dtype
        )
    else:
        dq = torch.empty((B, L, H, D), device=device, dtype=dtype)
    assert L % BLOCK == 0 and BLOCK <= 64
    grid = (B, H, k_dim_block * v_dim_block)
    num_warps = 4  # 2
    num_stages = 3  # 5
    kernel = (
        fp32_lightning_attention_q_backward_kernel
        if hp
        else lightning_attention_q_backward_kernel
    )
    kernel[grid](
        q,
        k,
        v,
        output_grad,
        dq,
        softmax_scale,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        output_grad.stride(1),
        decay_scales,
        L,
        D=D,
        KD=KD,
        VD=VD,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    KD = 32
    VD = 128
    BLOCK = 32
    k_dim_block = D // KD
    v_dim_block = D // VD
    if v_dim_block > 1:
        dk = torch.zeros(
            (B, L, H, D), device=device, dtype=torch.float32 if hpc else dtype
        )
    else:
        dk = torch.empty((B, L, H, D), device=device, dtype=dtype)
    if k_dim_block > 1:
        dv = torch.zeros(
            (B, L, H, D), device=device, dtype=torch.float32 if hpc else dtype
        )
    else:
        dv = torch.empty((B, L, H, D), device=device, dtype=dtype)
    num_warps = 4  # 4
    num_stages = 5  # 5
    grid = (B, H, k_dim_block * v_dim_block)
    kernel = (
        fp32_lightning_attention_kv_backward_kernel
        if hp
        else lightning_attention_kv_backward_kernel
    )
    kernel[grid](
        q,
        k,
        v,
        output_grad,
        dk,
        dv,
        softmax_scale,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        output_grad.stride(1),
        decay_scales,
        L,
        D=D,
        KD=KD,
        VD=VD,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if v_dim_block > 1 and hpc:
        dq = dq.to(dtype)
        dk = dk.to(dtype)
    if k_dim_block > 1 and hpc:
        dv = dv.to(dtype)
    return dq, dk, dv


@triton.jit
def fused_lightning_attention_backward_kernel(
    Q,
    K,
    V,
    S,
    G,
    DQ,
    DK,
    DV,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_g,
    decay_scales,
    L,
    D: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    N = D // VD
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)

    c0 = bid * L

    decay_scale = -tl.load(decay_scales + hid)

    offs_b = tl.arange(0, BLOCK)
    offs_k = tl.arange(0, KD)
    offs_v = tl.arange(0, VD)

    q_ptrs = (
        Q
        + c0 * stride_q
        + hid * D
        + kid * KD
        + (offs_b[:, None] * stride_q + offs_k[None, :])
    )
    k_ptrs = (
        K
        + c0 * stride_k
        + hid * D
        + kid * KD
        + (offs_b[:, None] * stride_k + offs_k[None, :])
    )
    v_ptrs = (
        V
        + c0 * stride_v
        + hid * D
        + vid * VD
        + (offs_b[:, None] * stride_v + offs_v[None, :])
    )
    g_ptrs = (
        G
        + c0 * D * H
        + hid * D
        + vid * VD
        + (offs_b[:, None] * stride_g + offs_v[None, :])
    )
    dq_ptrs = (
        DQ
        + c0 * D * H
        + hid * D
        + kid * KD
        + (offs_b[:, None] * H * D + offs_k[None, :])
    )
    dk_ptrs = (
        DK
        + c0 * H * D
        + hid * D
        + kid * KD
        + (offs_b[:, None] * H * D + offs_k[None, :])
    )
    dv_ptrs = (
        DV
        + c0 * H * D
        + hid * D
        + vid * VD
        + (offs_b[:, None] * H * D + offs_v[None, :])
    )
    s_ptrs = (
        S
        + bid * H * D * D
        + hid * D * D
        + kid * D * KD
        + vid * VD
        + (offs_k[:, None] * D + offs_v[None, :])
    )

    state = tl.load(s_ptrs).to(tl.float32)
    gs = tl.zeros((KD, VD), dtype=tl.float32)

    n_steps = tl.cdiv(L, BLOCK)
    for i in range(n_steps):
        n = (n_steps - i - 1) * BLOCK
        n = tl.multiple_of(n, BLOCK)

        q = tl.load(q_ptrs + n * stride_q).to(tl.float32)
        k = tl.load(k_ptrs + n * stride_k).to(tl.float32)
        v = tl.load(v_ptrs + n * stride_v).to(tl.float32)
        g = tl.load(g_ptrs + n * stride_g).to(tl.float32)
        b = BLOCK
        b_offs = b - 1 - offs_b

        block_decay = tl.exp(decay_scale * b)
        amps = tl.exp(-decay_scale * b_offs)  # [100, 10, 1]
        decays = tl.exp(decay_scale * b_offs)  # [0.01, 0.1, 1]

        qs = q * amps[:, None]  # [100, 10, 1]
        ks = k * decays[:, None]  # [0.01, 0.1, 1]
        qk = tl.dot(qs, tl.trans(ks)) * softmax_scale
        qk = tl.where(offs_b[None, :] <= offs_b[:, None], qk, 0.0)

        # state = state * block_decay
        # o = tl.dot(q * decays, state) * softmax_scale + tl.dot(qk, v)
        # state = state + tl.dot(tl.trans(k * decays[:, None]), v)

        state = state - tl.dot(tl.trans(ks), v)

        dv = tl.dot(tl.trans(qk), g)

        dv += tl.dot(ks, gs)

        dqk = tl.dot(g, tl.trans(v))
        dqk = tl.where(offs_b[None, :] <= offs_b[:, None], dqk, 0.0) * softmax_scale
        dk = tl.dot(tl.trans(dqk), qs)
        dk += tl.dot(v, tl.trans(gs))
        dk *= decays[:, None]

        dq = (
            tl.dot(dqk, ks)
            + tl.dot(g, tl.trans(state)) * softmax_scale * decays[:, None]
        )
        dq *= amps[:, None]
        state /= block_decay

        # dq = tl.dot(dqk, ks) + tl.dot(g, tl.trans(state)) * softmax_scale
        # dq *= amps[:, None]

        gs *= block_decay
        gs += tl.dot(tl.trans(qs), g) * softmax_scale * block_decay

        if VD == D:
            tl.store(dq_ptrs + n * H * D, dq.to(DQ.dtype.element_ty))
            tl.store(dk_ptrs + n * H * D, dk.to(DK.dtype.element_ty))
        else:
            tl.atomic_add(
                dq_ptrs + n * H * D, dq.to(DQ.dtype.element_ty), sem="relaxed"
            )
            tl.atomic_add(
                dk_ptrs + n * H * D, dk.to(DK.dtype.element_ty), sem="relaxed"
            )
        if KD == D:
            tl.store(dv_ptrs + n * H * D, dv.to(DV.dtype.element_ty))
        else:
            tl.atomic_add(
                dv_ptrs + n * H * D, dv.to(DV.dtype.element_ty), sem="relaxed"
            )


def triton_fused_lightning_attention_backward(
    output_grad, q, k, v, s, decay_scales, softmax_scale=None, hpc=False
):
    B, L, H, D = q.shape
    if softmax_scale is None:
        softmax_scale = D ** (-0.5)

    dtype = q.dtype
    device = q.device
    KD = 128
    VD = 32
    BLOCK = 32

    k_dim_block = D // KD
    v_dim_block = D // VD
    if v_dim_block > 1:
        dq = torch.zeros(
            (B, L, H, D), device=device, dtype=torch.float32 if hpc else dtype
        )
        dk = torch.zeros(
            (B, L, H, D), device=device, dtype=torch.float32 if hpc else dtype
        )
    else:
        dq = torch.empty((B, L, H, D), device=device, dtype=dtype)
        dk = torch.empty((B, L, H, D), device=device, dtype=dtype)
    if k_dim_block > 1:
        dv = torch.zeros(
            (B, L, H, D), device=device, dtype=torch.float32 if hpc else dtype
        )
    else:
        dv = torch.empty((B, L, H, D), device=device, dtype=dtype)

    assert L % BLOCK == 0 and BLOCK <= 64
    grid = (B, H, k_dim_block * v_dim_block)

    num_warps = 4  # 2
    num_stages = 3  # 3
    fused_lightning_attention_backward_kernel[grid](
        q,
        k,
        v,
        s,
        output_grad,
        dq,
        dk,
        dv,
        softmax_scale,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        output_grad.stride(1),
        decay_scales,
        L,
        D=D,
        KD=KD,
        VD=VD,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if hpc:
        if v_dim_block > 1:
            dq = dq.to(dtype)
            dk = dk.to(dtype)
        if k_dim_block > 1:
            dv = dv.to(dtype)
    return dq, dk, dv


# @triton.jit
# def varlen_lightning_attention_forward_kernel(
#     Q,
#     K,
#     V,
#     S,
#     Out,
#     softmax_scale,
#     stride_q,
#     stride_k,
#     stride_v,
#     stride_s,
#     stride_o,
#     CU,
#     PCU,
#     decay_scales,
#     D: tl.constexpr,
#     KD: tl.constexpr,
#     VD: tl.constexpr,
#     BLOCK: tl.constexpr,
#     EVEN: tl.constexpr,
#     PAD: tl.constexpr
# ):
#     bid = tl.program_id(0)
#     hid = tl.program_id(1)
#     kvid = tl.program_id(2)
#     N = D // VD
#     kid = kvid // N
#     vid = kvid % N
#     H = tl.num_programs(1)

#     if PAD:
#         c01 = tl.load(CU + bid + tl.arange(0, 2))
#         c0, c1 = tl.split(c01)
#         length = c1 - c0
#         pc01 = tl.load(PCU + bid + tl.arange(0, 2))
#         pc0, pc1 = tl.split(pc01)
#         padded_length = pc1 - pc0
#         c0 = pc0
#         if padded_length == 0:
#             return
#     else:
#         c01 = tl.load(CU + bid + tl.arange(0, 2))
#         c0, c1 = tl.split(c01)
#         length = c1 - c0
#         padded_length = length
#         if length == 0:
#             return

#     decay_scale = -tl.load(decay_scales + hid)

#     offs_b = tl.arange(0, BLOCK)
#     offs_k = tl.arange(0, KD)
#     offs_v = tl.arange(0, VD)

#     q_ptrs = (
#         Q
#         + c0 * stride_q
#         + hid * D
#         + kid * KD
#         + (offs_b[:, None] * stride_q + offs_k[None, :])
#     )
#     k_ptrs = (
#         K
#         + c0 * stride_k
#         + hid * D
#         + kid * KD
#         + (offs_b[:, None] * stride_k + offs_k[None, :])
#     )
#     v_ptrs = (
#         V
#         + c0 * stride_v
#         + hid * D
#         + vid * VD
#         + (offs_b[:, None] * stride_v + offs_v[None, :])
#     )
#     # (num_dim_block, length, qo_heads, d)
#     out_ptrs = (
#         Out
#         + kid * stride_o
#         + c0 * D * H
#         + hid * D
#         + vid * VD
#         + (offs_b[:, None] * H * D + offs_v[None, :])
#     )
#     s_ptrs = (
#         S
#         + bid * stride_s
#         + hid * D * D
#         + kid * D * KD
#         + vid * VD
#         + (offs_k[:, None] * D + offs_v[None, :])
#     )
#     state = tl.zeros((KD, VD), dtype=tl.float32)

#     for n in range(0, padded_length, BLOCK):
#         n = tl.multiple_of(n, BLOCK)

#         if EVEN:
#             q = tl.load(q_ptrs + n * stride_q).to(tl.float32)
#             k = tl.trans(tl.load(k_ptrs + n * stride_k)).to(tl.float32)
#             v = tl.load(v_ptrs + n * stride_v).to(tl.float32)
#             b = BLOCK
#             b_offs = b - 1 - offs_b
#             decays = tl.exp(decay_scale * b_offs)
#             inv_decays = 1 / decays
#         else:
#             q = tl.load(
#                 q_ptrs + n * stride_q, mask=(n + offs_b)[:, None] < length, other=0.0
#             ).to(tl.float32)
#             k = tl.trans(
#                 tl.load(
#                     k_ptrs + n * stride_k,
#                     mask=(n + offs_b)[:, None] < length,
#                     other=0.0,
#                 )
#             ).to(tl.float32)
#             v = tl.load(
#                 v_ptrs + n * stride_v, mask=(n + offs_b)[:, None] < length, other=0.0
#             ).to(tl.float32)
#             b = min(BLOCK, length - n)
#             b_offs = b - 1 - offs_b
#             block_decays = tl.exp(decay_scale * b_offs)
#             decays = tl.where(b_offs >= 0, block_decays, 0)
#             inv_decays = tl.where(b_offs >= 0, 1 / block_decays, 0)

#         q = q * inv_decays[:, None]
#         k = k * decays[None, :]
#         qk = tl.dot(q, k) * softmax_scale
#         qk = tl.where(offs_b[None, :] <= offs_b[:, None], qk, 0.0)
#         o = tl.dot(qk, v)

#         block_decay = tl.exp(decay_scale * b)
#         o = tl.dot(q, state) * block_decay * softmax_scale + o

#         state = state * block_decay + tl.dot(k, v)

#         if EVEN:
#             tl.store(out_ptrs + n * H * D, o.to(Out.dtype.element_ty))
#         else:
#             tl.store(
#                 out_ptrs + n * H * D,
#                 o.to(Out.dtype.element_ty),
#                 mask=(n + offs_b)[:, None] < length,
#             )
#     tl.store(s_ptrs, state.to(S.dtype.element_ty))


# def triton_varlen_lightning_attention_forward(q, k, v, decay_scales, cu_seqlens, padded_cu_seqlens, max_q_length, softmax_scale=None):
#     length, qo_heads, D = q.shape
#     _, kv_heads, _ = k.shape
#     bs = cu_seqlens.size(0) - 1
#     if softmax_scale is None:
#         softmax_scale = D ** (-0.5)

#     MAX_LENGTH = max_q_length

#     assert qo_heads == kv_heads, "triton_lightning_attention_forward does NOT support GQA currently"

#     KD = 32
#     VD = 32 if bs <= 2 else 64

#     num_warps = 2  # 2
#     num_stages = 3  # 3

#     k_dim_block = D // KD
#     v_dim_block = D // VD
#     tmp = torch.empty(
#         (k_dim_block, length, qo_heads, D), device=q.device, dtype=q.dtype
#     )
#     s = torch.empty(
#         (bs, qo_heads, D, D), device=q.device, dtype=torch.float32
#     )

#     # BLOCK should <= 64
#     BLOCK = 32
#     EVEN = MAX_LENGTH % BLOCK == 0 if bs == 1 else False
#     grid = (bs, kv_heads, k_dim_block * v_dim_block)
#     varlen_lightning_attention_forward_kernel[grid](
#         q,
#         k,
#         v,
#         s,
#         tmp,
#         softmax_scale,
#         q.stride(0),
#         k.stride(0),
#         v.stride(0),
#         s.stride(0),
#         tmp.stride(0),
#         cu_seqlens,
#         padded_cu_seqlens,
#         decay_scales,
#         D=D,
#         KD=KD,
#         VD=VD,
#         BLOCK=BLOCK,
#         EVEN=EVEN,
#         num_warps=num_warps,
#         num_stages=num_stages,
#     )

#     if k_dim_block > 1:
#         if length < 2048:
#             o = tmp.sum(0)
#         else:
#             o = torch.empty(
#                 (length, qo_heads, D), device=q.device, dtype=q.dtype
#             )
#             output_sum_kernel[(length,)](
#                 tmp,
#                 o,
#                 DIM=qo_heads * D,
#                 NUM_BLOCK=k_dim_block,
#                 num_warps=2,
#                 num_stages=3,
#             )
#     else:
#         o = tmp[0]

#     return o, s


# @triton.jit
# def varlen_lightning_attention_backward_kernel(
#     Q,
#     K,
#     V,
#     S,
#     G,
#     DQ,
#     DK,
#     DV,
#     softmax_scale,
#     stride_q,
#     stride_k,
#     stride_v,
#     stride_g,
#     CU,
#     PCU,
#     decay_scales,
#     D: tl.constexpr,
#     KD: tl.constexpr,
#     VD: tl.constexpr,
#     BLOCK: tl.constexpr,
#     EVEN: tl.constexpr,
#     PAD: tl.constexpr
# ):
#     bid = tl.program_id(0)
#     hid = tl.program_id(1)
#     kvid = tl.program_id(2)
#     N = D // VD
#     kid = kvid // N
#     vid = kvid % N
#     H = tl.num_programs(1)

#     if PAD:
#         c01 = tl.load(CU + bid + tl.arange(0, 2))
#         c0, c1 = tl.split(c01)
#         length = c1 - c0
#         pc01 = tl.load(PCU + bid + tl.arange(0, 2))
#         pc0, pc1 = tl.split(pc01)
#         padded_length = pc1 - pc0
#         c0 = pc0
#         if padded_length == 0:
#             return
#     else:
#         c01 = tl.load(CU + bid + tl.arange(0, 2))
#         c0, c1 = tl.split(c01)
#         length = c1 - c0
#         padded_length = length
#         if length == 0:
#             return

#     decay_scale = -tl.load(decay_scales + hid)

#     offs_b = tl.arange(0, BLOCK)
#     offs_k = tl.arange(0, KD)
#     offs_v = tl.arange(0, VD)

#     q_ptrs = (
#         Q
#         + c0 * stride_q
#         + hid * D
#         + kid * KD
#         + (offs_b[:, None] * stride_q + offs_k[None, :])
#     )
#     k_ptrs = (
#         K
#         + c0 * stride_k
#         + hid * D
#         + kid * KD
#         + (offs_b[:, None] * stride_k + offs_k[None, :])
#     )
#     v_ptrs = (
#         V
#         + c0 * stride_v
#         + hid * D
#         + vid * VD
#         + (offs_b[:, None] * stride_v + offs_v[None, :])
#     )
#     g_ptrs = (
#         G
#         + c0 * D * H
#         + hid * D
#         + kid * KD
#         + (offs_b[:, None] * stride_q + offs_k[None, :])
#     )
#     # (num_dim_block, length, qo_heads, d)
#     dq_ptrs = (
#         DQ
#         + c0 * D * H
#         + hid * D
#         + vid * VD
#         + (offs_b[:, None] * H * D + offs_k[None, :])
#     )
#     dk_ptrs = (
#         DK
#         + c0 * stride_k
#         + hid * D
#         + kid * KD
#         + (offs_b[:, None] * stride_k + offs_k[None, :])
#     )
#     dv_ptrs = (
#         DV
#         + c0 * stride_v
#         + hid * D
#         + vid * VD
#         + (offs_b[:, None] * stride_v + offs_v[None, :])
#     )
#     s_ptrs = (
#         S
#         + bid * H * D * D
#         + hid * D * D
#         + kid * D * KD
#         + vid * VD
#         + (offs_k[:, None] * D + offs_v[None, :])
#     )
#     state = tl.load(s_ptrs).to(tl.float32)
#     gs = tl.zeros((KD, VD), dtype=tl.float32)
#     n_steps = tl.cdiv(padded_length, BLOCK)
#     for i in range(n_steps):
#         n = (n_steps - i - 1) * BLOCK
#         n = tl.multiple_of(n, BLOCK)

#         if EVEN:
#             q = tl.load(q_ptrs + n * stride_q).to(tl.float32)
#             k = tl.trans(tl.load(k_ptrs + n * stride_k)).to(tl.float32)
#             v = tl.load(v_ptrs + n * stride_v).to(tl.float32)
#             g = tl.load(g_ptrs + n * stride_g).to(tl.float32)
#             b = BLOCK
#             b_offs = b - 1 - offs_b
#             decays = tl.exp(decay_scale * b_offs)
#             inv_decays = 1 / decays
#         else:
#             q = tl.load(
#                 q_ptrs + n * stride_q, mask=(n + offs_b)[:, None] < length, other=0.0
#             ).to(tl.float32)
#             k = tl.trans(
#                 tl.load(
#                     k_ptrs + n * stride_k,
#                     mask=(n + offs_b)[:, None] < length,
#                     other=0.0,
#                 )
#             ).to(tl.float32)
#             v = tl.load(
#                 v_ptrs + n * stride_v, mask=(n + offs_b)[:, None] < length, other=0.0
#             ).to(tl.float32)
#             g = tl.load(
#                 g_ptrs + n * stride_g, mask=(n + offs_b)[:, None] < length, other=0.0
#             ).to(tl.float32)
#             b = min(BLOCK, length - n)
#             b_offs = b - 1 - offs_b
#             block_decays = tl.exp(decay_scale * b_offs)
#             decays = tl.where(b_offs >= 0, block_decays, 0)
#             inv_decays = tl.where(b_offs >= 0, 1 / block_decays, 0)
#         block_decay = tl.exp(decay_scale * b)

#         q = q * inv_decays[:, None]
#         k = k * decays[None, :]
#         qk = tl.dot(q, k) * softmax_scale
#         qk = tl.where(offs_b[None, :] <= offs_b[:, None], qk, 0.0)
#         dv = tl.dot(tl.trans(qk), g)
#         dqk = tl.dot(g, tl.trans(v))
#         dqk = tl.where(offs_b[None, :] <= offs_b[:, None], dqk, 0.0)
#         dk = tl.dot(tl.trans(dqk), q)
#         dq = tl.dot(dqk, k)

#         o = tl.dot(q, state) * block_decay * softmax_scale + o

#         state = (state - tl.dot(k, v))/block_decay
#         dq += tl.dot(g, tl.trans(state)) * block_decay * softmax_scale
#         dv += tl.dot(k, gs)
#         dk += tl.dot(v, tl.trans(gs))
#         gs += tl.dot(tl.trans(q), g)

#         if EVEN:
#             tl.store(dq_ptrs + n * H * D, dq.to(DQ.dtype.element_ty))
#             tl.store(dk_ptrs + n * H * D, dk.to(DK.dtype.element_ty))
#             tl.store(dv_ptrs + n * H * D, dv.to(DV.dtype.element_ty))

#         else:
#             tl.store(
#                 dq_ptrs + n * H * D,
#                 dq.to(DQ.dtype.element_ty),
#                 mask=(n + offs_b)[:, None] < length,
#             )
#             tl.store(
#                 dk_ptrs + n * H * D,
#                 dk.to(DQ.dtype.element_ty),
#                 mask=(n + offs_b)[:, None] < length,
#             )
#             tl.store(
#                 dv_ptrs + n * H * D,
#                 dv.to(DQ.dtype.element_ty),
#                 mask=(n + offs_b)[:, None] < length,
#             )


# def triton_varlen_lightning_attention_backward(output_grad, q, k, v, s, decay_scales, cu_seqlens, padded_cu_seqlens, max_q_length, softmax_scale=None):
#     length, qo_heads, D = q.shape
#     _, kv_heads, _ = k.shape
#     bs = cu_seqlens.size(0) - 1
#     if softmax_scale is None:
#         softmax_scale = D ** (-0.5)

#     MAX_LENGTH = max_q_length

#     assert qo_heads == kv_heads, "triton_lightning_attention_forward does NOT support GQA currently"

#     KD = 128
#     VD = 128

#     num_warps = 2  # 2
#     num_stages = 3  # 3

#     k_dim_block = D // KD
#     v_dim_block = D // VD
#     dq = torch.empty(
#         (length, qo_heads, D), device=q.device, dtype=q.dtype
#     )
#     dk = torch.empty(
#         (length, qo_heads, D), device=q.device, dtype=q.dtype
#     )
#     dv = torch.empty(
#         (length, qo_heads, D), device=q.device, dtype=q.dtype
#     )

#     # BLOCK should <= 64
#     BLOCK = 32
#     EVEN = MAX_LENGTH % BLOCK == 0 if bs == 1 else False
#     grid = (bs, kv_heads, k_dim_block * v_dim_block)
#     varlen_lightning_attention_backward_kernel[grid](
#         q,
#         k,
#         v,
#         s,
#         output_grad,
#         dq,
#         dk,
#         dv,
#         softmax_scale,
#         q.stride(0),
#         k.stride(0),
#         v.stride(0),
#         output_grad.stride(0),
#         cu_seqlens,
#         padded_cu_seqlens,
#         decay_scales,
#         D=D,
#         KD=KD,
#         VD=VD,
#         BLOCK=BLOCK,
#         EVEN=EVEN,
#         num_warps=num_warps,
#         num_stages=num_stages,
#     )

#     return dq, dk, dv
