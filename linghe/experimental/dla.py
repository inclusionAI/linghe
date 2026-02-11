# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl

from linghe.experimental.symm_mem_barrier import symm_mem_sync

"""
context-parallel lightning attention
"""


@triton.jit
def cp_lightning_attention_forward_kernel(
    Q,
    K,
    V,
    S,
    Out,
    buffer_ptrs,
    signal_ptrs,
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
    SIZE: tl.constexpr,
    RANK: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    N = D // VD
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)

    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))

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
        + hid * 2 * D * D
        + kid * D * KD
        + vid * VD
        + (offs_k[:, None] * D + offs_v[None, :])
    )

    buffer_offs = (
        bid * stride_s
        + hid * 2 * D * D
        + kid * D * KD
        + vid * VD
        + (offs_k[:, None] * D + offs_v[None, :])
    )

    block_decay = tl.exp(decay_scale * BLOCK)
    mask = tl.exp(decay_scale * (offs_b[:, None] - offs_b[None, :]))
    mask = tl.where(offs_b[None, :] <= offs_b[:, None], mask, 0.0) * softmax_scale
    b_offs = BLOCK - 1 - offs_b
    decays = tl.exp(decay_scale * b_offs)
    amps = block_decay * softmax_scale / decays

    state0 = tl.zeros((KD, VD), dtype=tl.float32)

    # cid = 0
    for n in range(0, L // 2, BLOCK):
        n = tl.multiple_of(n, BLOCK)

        q = tl.load(q_ptrs + n * stride_q)
        k = tl.trans(tl.load(k_ptrs + n * stride_k))
        v = tl.load(v_ptrs + n * stride_v)

        qk = tl.dot(q, k) * mask
        o = tl.dot(qk.to(v.dtype), v)

        # o = tl.dot((q * amps[:, None]).to(q.dtype), state.to(q.dtype), o)
        o += tl.dot((q * amps[:, None]), state0)

        state0 *= block_decay
        # state = tl.dot((k * decays[None, :]).to(v.dtype), v, state)
        state0 += tl.dot((k * decays[None, :]), v.to(tl.float32))

        if KD == D:
            tl.store(out_ptrs + n * H * D, o)
        else:
            tl.atomic_add(out_ptrs + n * H * D, o, sem="relaxed")

    buffer_ptr = tl.load(buffer_ptrs + RANK).to(tl.pointer_type(tl.float32))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    tl.store(buffer_ptr + buffer_offs, state0)

    # c1
    state1 = tl.zeros((KD, VD), dtype=tl.float32)
    DD = D * D
    for n in range(L // 2, L, BLOCK):
        n = tl.multiple_of(n, BLOCK)

        q = tl.load(q_ptrs + n * stride_q)
        k = tl.trans(tl.load(k_ptrs + n * stride_k))
        v = tl.load(v_ptrs + n * stride_v)

        qk = tl.dot(q, k) * mask
        o = tl.dot(qk.to(v.dtype), v)

        # o = tl.dot((q * amps[:, None]).to(q.dtype), state.to(q.dtype), o)
        o += tl.dot((q * amps[:, None]), state1)

        state1 *= block_decay
        # state = tl.dot((k * decays[None, :]).to(v.dtype), v, state)
        state1 += tl.dot((k * decays[None, :]), v.to(tl.float32))

        if KD == D:
            tl.store(out_ptrs + n * H * D, o)
        else:
            tl.atomic_add(out_ptrs + n * H * D, o, sem="relaxed")

    tl.store(buffer_ptr + DD + buffer_offs, state1)
    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # accumulate c0
    gcid = RANK
    if (gcid + 1) % 2 == 0:
        pre_rank = RANK - 1
        chunk_decay = tl.exp(L // 2 * decay_scale)
        pre_buffer_ptr = tl.load(buffer_ptrs + pre_rank).to(tl.pointer_type(tl.float32))
        state0 += tl.load(pre_buffer_ptr + buffer_offs) * chunk_decay
        tl.store(buffer_ptr + buffer_offs, state0)

    # accumulate c1
    gcid = 2 * SIZE - 1 - RANK
    if (gcid + 1) % 2 == 0:
        pre_rank = RANK + 1
        chunk_decay = tl.exp(L // 2 * decay_scale)
        pre_buffer_ptr = tl.load(buffer_ptrs + pre_rank).to(tl.pointer_type(tl.float32))
        state1 += tl.load(pre_buffer_ptr + DD + buffer_offs) * chunk_decay
        tl.store(buffer_ptr + DD + buffer_offs, state1)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    if SIZE >= 4:
        gcid = RANK
        if (gcid + 1) % 4 == 0:
            pre_rank = RANK - 2
            chunk_decay = tl.exp(L * decay_scale)
            pre_buffer_ptr = tl.load(buffer_ptrs + pre_rank).to(
                tl.pointer_type(tl.float32)
            )
            state0 += tl.load(pre_buffer_ptr + buffer_offs) * chunk_decay
            tl.store(buffer_ptr + buffer_offs, state0)

        gcid = 2 * SIZE - 1 - RANK
        if (gcid + 1) % 4 == 0:
            pre_rank = RANK + 2
            chunk_decay = tl.exp(L * decay_scale)
            pre_buffer_ptr = tl.load(buffer_ptrs + pre_rank).to(
                tl.pointer_type(tl.float32)
            )
            state1 += tl.load(pre_buffer_ptr + DD + buffer_offs) * chunk_decay
            tl.store(buffer_ptr + DD + buffer_offs, state1)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    if SIZE >= 8:
        gcid = RANK
        if (gcid + 1) % 8 == 0:
            pre_rank = RANK - 4
            chunk_decay = tl.exp(L * 2 * decay_scale)
            pre_buffer_ptr = tl.load(buffer_ptrs + pre_rank).to(
                tl.pointer_type(tl.float32)
            )
            state0 += tl.load(pre_buffer_ptr + buffer_offs) * chunk_decay
            tl.store(buffer_ptr + buffer_offs, state0)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # scatter
    if SIZE >= 8:
        gcid = 2 * SIZE - 1 - RANK

        if (gcid + 1) % 8 == 4:
            pre_gcid = gcid - 4
            pre_rank = pre_gcid if pre_gcid < SIZE else 2 * SIZE - 1 - pre_gcid
            pre_offs = 0 if pre_gcid < SIZE else DD
            chunk_decay = tl.exp(L * decay_scale)
            pre_buffer_ptr = tl.load(buffer_ptrs + pre_rank).to(
                tl.pointer_type(tl.float32)
            )
            state1 += tl.load(pre_buffer_ptr + pre_offs + buffer_offs) * chunk_decay
            tl.store(buffer_ptr + buffer_offs, state1)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # accumulate c0

    if SIZE >= 4:
        gcid = RANK
        if ((gcid + 1) % 4 == 2) and (gcid > 4):
            pre_gcid = gcid - 2
            pre_rank = pre_gcid if pre_gcid < SIZE else 2 * SIZE - 1 - pre_gcid
            pre_offs = 0 if pre_gcid < SIZE else DD
            chunk_decay = tl.exp(L * decay_scale)
            pre_buffer_ptr = tl.load(buffer_ptrs + pre_rank).to(
                tl.pointer_type(tl.float32)
            )
            state0 += tl.load(pre_buffer_ptr + pre_offs + buffer_offs) * chunk_decay
            tl.store(buffer_ptr + buffer_offs, state0)

        gcid = 2 * SIZE - 1 - RANK
        if (gcid + 1) % 4 == 2:
            pre_gcid = gcid - 2
            pre_rank = pre_gcid if pre_gcid < SIZE else 2 * SIZE - 1 - pre_gcid
            pre_offs = 0 if pre_gcid < SIZE else DD
            chunk_decay = tl.exp(L * decay_scale)
            pre_buffer_ptr = tl.load(buffer_ptrs + pre_rank).to(
                tl.pointer_type(tl.float32)
            )
            state1 += tl.load(pre_buffer_ptr + pre_offs + buffer_offs) * chunk_decay
            tl.store(buffer_ptr + DD + buffer_offs, state1)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    gcid = RANK
    if ((gcid + 1) % 2 == 1) and (gcid > 0):
        chunk_decay = tl.exp(L // 2 * decay_scale)
        pre_gcid = gcid - 1
        pre_rank = pre_gcid if pre_gcid < SIZE else 2 * SIZE - 1 - pre_gcid
        pre_offs = 0 if pre_gcid < SIZE else DD
        pre_buffer_ptr = tl.load(buffer_ptrs + pre_rank).to(tl.pointer_type(tl.float32))
        state0 += tl.load(pre_buffer_ptr + pre_offs + buffer_offs) * chunk_decay
        tl.store(buffer_ptr + buffer_offs, state0)

    gcid = 2 * SIZE - 1 - RANK
    if (gcid + 1) % 2 == 1:
        chunk_decay = tl.exp(L // 2 * decay_scale)
        pre_gcid = gcid - 1
        pre_rank = pre_gcid if pre_gcid < SIZE else 2 * SIZE - 1 - pre_gcid
        pre_offs = 0 if pre_gcid < SIZE else DD
        pre_buffer_ptr = tl.load(buffer_ptrs + pre_rank).to(tl.pointer_type(tl.float32))
        state1 += tl.load(pre_buffer_ptr + pre_offs + buffer_offs) * chunk_decay
        tl.store(buffer_ptr + DD + buffer_offs, state1)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    tl.store(s_ptrs, state0)
    tl.store(s_ptrs + DD, state1)

    # read pre state
    # chunk_decay = tl.exp(L // 2 * decay_scale)
    if RANK > 0:
        gcid = RANK
        pre_gcid = gcid - 1
        pre_rank = pre_gcid if pre_gcid < SIZE else 2 * SIZE - 1 - pre_gcid
        pre_offs = 0 if pre_gcid < SIZE else DD
        buffer_ptr = tl.load(buffer_ptrs + pre_rank).to(tl.pointer_type(tl.float32))
        buffer_ptr = tl.multiple_of(buffer_ptr, 16)
        # pre_state = tl.load(buffer_ptr + pre_offs + buffer_offs).to(Q.dtype.element_ty)
        pre_state = tl.load(
            buffer_ptr + pre_offs + buffer_offs
        )  # .to(Q.dtype.element_ty)

        iter_amps = amps * block_decay
        for n in range(0, L // 2, BLOCK):
            n = tl.multiple_of(n, BLOCK)

            q = tl.load(q_ptrs + n * stride_q)
            # q = (q * iter_amps[:, None]).to(q.dtype)
            q = q * iter_amps[:, None]

            # o = tl.dot(q, pre_state.to(q.dtype))
            o = tl.dot(q, pre_state)

            iter_amps *= block_decay
            tl.atomic_add(out_ptrs + n * H * D, o, sem="relaxed")

    gcid = 2 * SIZE - 1 - RANK
    pre_gcid = gcid - 1
    pre_rank = pre_gcid if pre_gcid < SIZE else 2 * SIZE - 1 - pre_gcid
    pre_offs = 0 if pre_gcid < SIZE else DD
    buffer_ptr = tl.load(buffer_ptrs + pre_rank).to(tl.pointer_type(tl.float32))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    # pre_state = tl.load(buffer_ptr + pre_offs + buffer_offs).to(Q.dtype.element_ty)
    pre_state = tl.load(buffer_ptr + pre_offs + buffer_offs)  # .to(Q.dtype.element_ty)

    iter_amps = amps * block_decay
    for n in range(L // 2, L, BLOCK):
        n = tl.multiple_of(n, BLOCK)

        q = tl.load(q_ptrs + n * stride_q)
        # q = (q * iter_amps[:, None]).to(q.dtype)
        q = q * iter_amps[:, None]

        # o = tl.dot(q, pre_state.to(q.dtype))
        o = tl.dot(q, pre_state)

        iter_amps *= block_decay
        tl.atomic_add(out_ptrs + n * H * D, o, sem="relaxed")


def triton_cp_lightning_attention_forward(
    q, k, v, decay_scales, hdl, group, hpc=True, softmax_scale=None
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
        outputs = torch.empty(
            (B, L, H, D), device=device, dtype=torch.float32 if hpc else dtype
        )
    else:
        outputs = torch.zeros(
            (B, L, H, D), device=device, dtype=torch.float32 if hpc else dtype
        )

    s = torch.empty((B, H, 2, D, D), device=device, dtype=torch.float32)
    assert L % BLOCK == 0 and BLOCK <= 64
    group_size = hdl.world_size
    group_rank = hdl.rank

    grid = (B, H, k_dim_block * v_dim_block)
    cp_lightning_attention_forward_kernel[grid](
        q,
        k,
        v,
        s,
        outputs,
        hdl.buffer_ptrs_dev,
        hdl.signal_pad_ptrs_dev,
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
        SIZE=group_size,
        RANK=group_rank,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    o = outputs.to(dtype)

    return o, s


@triton.jit
def cp_lightning_attention_q_backward_kernel(
    Q,
    K,
    V,
    S,
    G,
    DQ,
    buffer_ptrs,
    signal_ptrs,
    softmax_scale,
    stride_q,
    stride_k,
    stride_v,
    stride_s,
    stride_g,
    decay_scales,
    L,
    D: tl.constexpr,
    KD: tl.constexpr,
    VD: tl.constexpr,
    BLOCK: tl.constexpr,
    SIZE: tl.constexpr,
    RANK: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    N = D // VD
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)
    DD = D * D

    c0 = bid * L

    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))
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
    s_ptrs = (
        S
        + bid * stride_s
        + hid * 2 * D * D
        + kid * D * KD
        + vid * VD
        + (offs_k[:, None] * D + offs_v[None, :])
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

    buffer_offs = (
        bid * stride_s
        + hid * 2 * D * D
        + kid * D * KD
        + vid * VD
        + (offs_k[:, None] * D + offs_v[None, :])
    )

    # store state to buffer
    state = tl.load(s_ptrs)
    buffer_ptr = tl.load(buffer_ptrs + RANK).to(tl.pointer_type(tl.float32))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    tl.store(buffer_ptr + buffer_offs, state)

    state = tl.load(s_ptrs + DD)
    buffer_ptr = tl.load(buffer_ptrs + RANK).to(tl.pointer_type(tl.float32))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    tl.store(buffer_ptr + DD + buffer_offs, state)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    mask = tl.exp((offs_b[:, None] - offs_b[None, :]) * decay_scale)
    mask = tl.where(offs_b[None, :] <= offs_b[:, None], mask, 0.0) * softmax_scale
    decay_offs = BLOCK - 1 - offs_b
    block_decay = tl.exp(decay_scale * BLOCK)
    decays = tl.exp(decay_scale * decay_offs)  # [0.01, 0.1, 1]

    # read state from buffer, c0
    state = tl.zeros((KD, VD), dtype=tl.float32)
    if RANK > 0:
        gcid = RANK
        pre_gcid = gcid - 1
        pre_rank = pre_gcid if pre_gcid < SIZE else 2 * SIZE - 1 - pre_gcid
        pre_offs = 0 if pre_gcid < SIZE else DD
        buffer_ptr = tl.load(buffer_ptrs + pre_rank).to(tl.pointer_type(tl.float32))
        buffer_ptr = tl.multiple_of(buffer_ptr, 16)
        state += tl.load(buffer_ptr + pre_offs + buffer_offs)  # .to(Q.dtype.element_ty)

    for n in range(0, L // 2, BLOCK):
        n = tl.multiple_of(n, BLOCK)

        k = tl.load(k_ptrs + n * stride_k)
        v = tl.load(v_ptrs + n * stride_v)
        g = tl.load(g_ptrs + n * stride_g)

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

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # c1
    state = tl.zeros((KD, VD), dtype=tl.float32)
    gcid = 2 * SIZE - 1 - RANK
    pre_gcid = gcid - 1
    pre_rank = pre_gcid if pre_gcid < SIZE else 2 * SIZE - 1 - pre_gcid
    pre_offs = 0 if pre_gcid < SIZE else DD
    buffer_ptr = tl.load(buffer_ptrs + pre_rank).to(tl.pointer_type(tl.float32))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    state += tl.load(buffer_ptr + pre_offs + buffer_offs)  # .to(Q.dtype.element_ty)

    for n in range(L // 2, L, BLOCK):
        n = tl.multiple_of(n, BLOCK)

        k = tl.load(k_ptrs + n * stride_k)
        v = tl.load(v_ptrs + n * stride_v)
        g = tl.load(g_ptrs + n * stride_g)

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
def cp_lightning_attention_kv_backward_kernel(
    Q,
    K,
    V,
    G,
    DK,
    DV,
    buffer_ptrs,
    signal_ptrs,
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
    SIZE: tl.constexpr,
    RANK: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kvid = tl.program_id(2)
    N = D // VD
    kid = kvid // N
    vid = kvid % N
    H = tl.num_programs(1)
    DD = D * D
    c0 = bid * L

    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))

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
    buffer_offs = (
        bid * H * 2 * D * D
        + hid * 2 * D * D
        + kid * D * KD
        + vid * VD
        + (offs_k[:, None] * D + offs_v[None, :])
    )

    b_offs = BLOCK - 1 - offs_b

    block_decay = tl.exp(decay_scale * BLOCK)
    amps = tl.exp(-decay_scale * b_offs)  # [100, 10, 1]
    decays = 1 / amps  # [0.01, 0.1, 1]
    sd = softmax_scale * block_decay

    mask = tl.exp((offs_b[:, None] - offs_b[None, :]) * decay_scale)
    mask = tl.where(offs_b[None, :] <= offs_b[:, None], mask, 0.0) * softmax_scale

    gs0 = tl.zeros((KD, VD), dtype=tl.float32)
    n_steps = tl.cdiv(L // 2, BLOCK)
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

        dv += tl.dot(k * decays[:, None], gs0)

        dqk = (tl.dot(g, tl.trans(v)) * mask).to(q.dtype)
        dk = tl.dot(tl.trans(dqk), (q * amps[:, None]).to(q.dtype))
        dk = tl.dot(v, tl.trans(gs0.to(v.dtype)), dk)
        dk *= decays[:, None]

        gs0 *= block_decay
        gs0 += tl.dot(tl.trans(q * amps[:, None]).to(g.dtype), g) * sd

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

    buffer_ptr = tl.load(buffer_ptrs + RANK).to(tl.pointer_type(tl.float32))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    tl.store(buffer_ptr + buffer_offs, gs0)

    gs1 = tl.zeros((KD, VD), dtype=tl.float32)
    n_steps = tl.cdiv(L // 2, BLOCK)
    for i in range(n_steps):
        n = (n_steps - i - 1) * BLOCK + L // 2
        n = tl.multiple_of(n, BLOCK)

        q = tl.load(q_ptrs + n * stride_q)
        k = tl.load(k_ptrs + n * stride_k)
        v = tl.load(v_ptrs + n * stride_v)
        g = tl.load(g_ptrs + n * stride_g)

        # qs = q * amps[:, None]   # [100, 10, 1]
        # ks = k * decays[:, None]  # [0.01, 0.1, 1]
        qk = tl.dot(q, tl.trans(k)) * mask

        dv = tl.dot(tl.trans(qk).to(g.dtype), g)

        dv += tl.dot(k * decays[:, None], gs1)

        dqk = (tl.dot(g, tl.trans(v)) * mask).to(q.dtype)
        dk = tl.dot(tl.trans(dqk), (q * amps[:, None]).to(q.dtype))
        dk = tl.dot(v, tl.trans(gs1.to(v.dtype)), dk)
        dk *= decays[:, None]

        gs1 *= block_decay
        gs1 += tl.dot(tl.trans(q * amps[:, None]).to(g.dtype), g) * sd

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

    tl.store(buffer_ptr + DD + buffer_offs, gs1)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # accumulate c0
    gcid = RANK
    if RANK > 0:
        if (gcid + 1) % 2 == 1:
            next_rank = RANK + 1
            chunk_decay = tl.exp(L // 2 * decay_scale)
            next_buffer_ptr = tl.load(buffer_ptrs + next_rank).to(
                tl.pointer_type(tl.float32)
            )
            gs0 += tl.load(next_buffer_ptr + buffer_offs) * chunk_decay
            tl.store(buffer_ptr + buffer_offs, gs0)

    # accumulate c1
    gcid = 2 * SIZE - 1 - RANK
    if (gcid + 1) % 2 == 1:
        next_rank = RANK - 1
        chunk_decay = tl.exp(L // 2 * decay_scale)
        next_buffer_ptr = tl.load(buffer_ptrs + next_rank).to(
            tl.pointer_type(tl.float32)
        )
        gs1 += tl.load(next_buffer_ptr + DD + buffer_offs) * chunk_decay
        tl.store(buffer_ptr + DD + buffer_offs, gs1)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    if RANK >= 4:
        # accumulate c0
        gcid = RANK
        if RANK > 0:
            if (gcid + 1) % 4 == 1:
                next_rank = RANK + 2
                chunk_decay = tl.exp(L * decay_scale)
                next_buffer_ptr = tl.load(buffer_ptrs + next_rank).to(
                    tl.pointer_type(tl.float32)
                )
                gs0 += tl.load(next_buffer_ptr + buffer_offs) * chunk_decay
                tl.store(buffer_ptr + buffer_offs, gs0)

        # accumulate c1
        gcid = 2 * SIZE - 1 - RANK
        if (gcid + 1) % 4 == 1:
            next_rank = RANK - 2
            chunk_decay = tl.exp(L * decay_scale)
            next_buffer_ptr = tl.load(buffer_ptrs + next_rank).to(
                tl.pointer_type(tl.float32)
            )
            gs1 += tl.load(next_buffer_ptr + DD + buffer_offs) * chunk_decay
            tl.store(buffer_ptr + DD + buffer_offs, gs1)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    if SIZE >= 8:
        # accumulate c1
        gcid = 2 * SIZE - 1 - RANK
        if (gcid + 1) % 8 == 1:
            next_rank = RANK - 4
            chunk_decay = tl.exp(L * 2 * decay_scale)
            next_buffer_ptr = tl.load(buffer_ptrs + next_rank).to(
                tl.pointer_type(tl.float32)
            )
            gs1 += tl.load(next_buffer_ptr + DD + buffer_offs) * chunk_decay
            tl.store(buffer_ptr + DD + buffer_offs, gs1)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # scatter
    if SIZE >= 8:
        # accumulate c0
        gcid = RANK
        if (gcid + 1) % 8 == 5:
            next_gcid = gcid + 4
            next_rank = next_gcid if next_gcid < SIZE else 2 * SIZE - 1 - next_gcid
            next_offs = 0 if next_gcid < SIZE else DD
            chunk_decay = tl.exp(L * 2 * decay_scale)
            next_buffer_ptr = tl.load(buffer_ptrs + next_rank).to(
                tl.pointer_type(tl.float32)
            )
            gs0 += tl.load(next_buffer_ptr + next_offs + buffer_offs) * chunk_decay
            tl.store(buffer_ptr + buffer_offs, gs0)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # scatter
    if SIZE >= 4:
        # accumulate c0
        gcid = RANK
        if (gcid + 1) % 4 == 3:
            next_gcid = gcid + 2
            next_rank = next_gcid if next_gcid < SIZE else 2 * SIZE - 1 - next_gcid
            next_offs = 0 if next_gcid < SIZE else DD
            chunk_decay = tl.exp(L * decay_scale)
            next_buffer_ptr = tl.load(buffer_ptrs + next_rank).to(
                tl.pointer_type(tl.float32)
            )
            gs0 += tl.load(next_buffer_ptr + next_offs + buffer_offs) * chunk_decay
            tl.store(buffer_ptr + buffer_offs, gs0)

        gcid = 2 * SIZE - 1 - RANK
        if ((gcid + 1) % 4 == 3) and (gcid < 14):
            next_gcid = gcid + 2
            next_rank = next_gcid if next_gcid < SIZE else 2 * SIZE - 1 - next_gcid
            next_offs = 0 if next_gcid < SIZE else DD
            chunk_decay = tl.exp(L * decay_scale)
            next_buffer_ptr = tl.load(buffer_ptrs + next_rank).to(
                tl.pointer_type(tl.float32)
            )
            gs1 += tl.load(next_buffer_ptr + next_offs + buffer_offs) * chunk_decay
            tl.store(buffer_ptr + buffer_offs, gs1)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    # accumulate c0
    gcid = RANK
    if (gcid + 1) % 2 == 0:
        next_gcid = gcid + 1
        next_rank = next_gcid if next_gcid < SIZE else 2 * SIZE - 1 - next_gcid
        next_offs = 0 if next_gcid < SIZE else DD
        chunk_decay = tl.exp(L // 2 * decay_scale)
        next_buffer_ptr = tl.load(buffer_ptrs + next_rank).to(
            tl.pointer_type(tl.float32)
        )
        gs0 += tl.load(next_buffer_ptr + next_offs + buffer_offs) * chunk_decay
        tl.store(buffer_ptr + buffer_offs, gs0)

    # accumulate c1
    gcid = 2 * SIZE - 1 - RANK
    if RANK > 0:
        if (gcid + 1) % 2 == 0:
            next_gcid = gcid + 1
            next_rank = next_gcid if next_gcid < SIZE else 2 * SIZE - 1 - next_gcid
            next_offs = 0 if next_gcid < SIZE else DD
            chunk_decay = tl.exp(L * decay_scale)
            next_buffer_ptr = tl.load(buffer_ptrs + next_rank).to(
                tl.pointer_type(tl.float32)
            )
            gs1 += tl.load(next_buffer_ptr + DD + buffer_offs) * chunk_decay
            tl.store(buffer_ptr + DD + buffer_offs, gs1)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )

    gcid = RANK
    next_gcid = gcid + 1
    next_rank = next_gcid if next_gcid < SIZE else 2 * SIZE - 1 - next_gcid
    next_offs = 0 if next_gcid < SIZE else DD
    chunk_decay = tl.exp(L // 2 * decay_scale)
    next_buffer_ptr = tl.load(buffer_ptrs + next_rank).to(tl.pointer_type(tl.float32))
    gs = tl.load(next_buffer_ptr + next_offs + buffer_offs)

    n_steps = tl.cdiv(L // 2, BLOCK)
    for i in range(n_steps):
        n = (n_steps - i - 1) * BLOCK
        n = tl.multiple_of(n, BLOCK)

        k = tl.load(k_ptrs + n * stride_k)
        v = tl.load(v_ptrs + n * stride_v)

        dv = tl.dot(k * decays[:, None], gs)

        dk = tl.dot(v.to(tl.float32), tl.trans(gs))

        gs *= block_decay

        tl.atomic_add(dk_ptrs + n * H * D, dk.to(DK.dtype.element_ty), sem="relaxed")
        tl.atomic_add(dv_ptrs + n * H * D, dv.to(DV.dtype.element_ty), sem="relaxed")

    if RANK > 0:
        gcid = 2 * SIZE - 1 - RANK
        next_gcid = gcid + 1
        next_rank = next_gcid if next_gcid < SIZE else 2 * SIZE - 1 - next_gcid
        next_offs = 0 if next_gcid < SIZE else DD
        chunk_decay = tl.exp(L // 2 * decay_scale)
        next_buffer_ptr = tl.load(buffer_ptrs + next_rank).to(
            tl.pointer_type(tl.float32)
        )
        gs = tl.load(next_buffer_ptr + next_offs + buffer_offs)

        n_steps = tl.cdiv(L // 2, BLOCK)
        for i in range(n_steps):
            n = (n_steps - i - 1) * BLOCK + L // 2
            n = tl.multiple_of(n, BLOCK)

            k = tl.load(k_ptrs + n * stride_k)
            v = tl.load(v_ptrs + n * stride_v)

            dv = tl.dot(k * decays[:, None], gs)

            dk = tl.dot(v, tl.trans(gs.to(v.dtype)))

            gs *= block_decay

            tl.atomic_add(
                dk_ptrs + n * H * D, dk.to(DK.dtype.element_ty), sem="relaxed"
            )
            tl.atomic_add(
                dv_ptrs + n * H * D, dv.to(DV.dtype.element_ty), sem="relaxed"
            )


def triton_cp_lightning_attention_backward(
    output_grad,
    q,
    k,
    v,
    s,
    decay_scales,
    hdl,
    group,
    softmax_scale=None,
    hpc=False,
    hp=False,
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
        dq = torch.empty(
            (B, L, H, D), device=device, dtype=torch.float32 if hpc else dtype
        )
    assert L % BLOCK == 0 and BLOCK <= 64
    group_size = hdl.world_size
    group_rank = hdl.rank

    grid = (B, H, k_dim_block * v_dim_block)
    num_warps = 4  # 2
    num_stages = 3  # 5
    cp_lightning_attention_q_backward_kernel[grid](
        q,
        k,
        v,
        s,
        output_grad,
        dq,
        hdl.buffer_ptrs_dev,
        hdl.signal_pad_ptrs_dev,
        softmax_scale,
        q.stride(1),
        k.stride(1),
        v.stride(1),
        s.stride(0),
        output_grad.stride(1),
        decay_scales,
        L,
        D=D,
        KD=KD,
        VD=VD,
        BLOCK=BLOCK,
        SIZE=group_size,
        RANK=group_rank,
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
        dk = torch.empty(
            (B, L, H, D), device=device, dtype=torch.float32 if hpc else dtype
        )
    if k_dim_block > 1:
        dv = torch.zeros(
            (B, L, H, D), device=device, dtype=torch.float32 if hpc else dtype
        )
    else:
        dv = torch.empty(
            (B, L, H, D), device=device, dtype=torch.float32 if hpc else dtype
        )
    num_warps = 4  # 4
    num_stages = 5  # 5
    grid = (B, H, k_dim_block * v_dim_block)
    cp_lightning_attention_kv_backward_kernel[grid](
        q,
        k,
        v,
        output_grad,
        dk,
        dv,
        hdl.buffer_ptrs_dev,
        hdl.signal_pad_ptrs_dev,
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
        SIZE=group_size,
        RANK=group_rank,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    dq = dq.to(dtype)
    dk = dk.to(dtype)
    dv = dv.to(dtype)
    return dq, dk, dv
