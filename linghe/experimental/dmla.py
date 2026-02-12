# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math

import torch
import triton
import triton.language as tl

from linghe.attn.mla import mla_ds_kernel
from linghe.experimental.symm_mem_barrier import symm_mem_sync

"""
context-parallel multi latent attention
"""


@triton.jit
def deprecated_cp_mla_forward_kernel(
        Q,
        K,
        V,
        Out,
        LSE,
        ML,
        buffer_ptrs,
        signal_ptrs,
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
        SIZE: tl.constexpr,
        RANK: tl.constexpr, ):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.num_programs(2) - tl.program_id(2) - 1
    H = tl.num_programs(1)
    cid = 0 if mid < L // 2 // M else 1

    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))

    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)

    offs_0 = tl.arange(0, 128)
    offs_1 = tl.arange(0, 64)

    # [B, L, H, 192】
    q_ptrs = (Q
              + (bid * L + mid * M) * stride_q
              + hid * 192
              + (offs_m[:, None] * stride_q + offs_1[None, :]))

    k_ptrs = (K
              + bid * L * stride_k
              + hid * 192
              + (offs_n[:, None] * stride_k + offs_1[None, :]))

    v_ptrs = (V
              + bid * L * stride_v
              + hid * 128
              + (offs_n[:, None] * stride_v + offs_0[None, :]))

    q0 = tl.load(q_ptrs)
    q1 = tl.load(q_ptrs + 64)
    q2 = tl.load(q_ptrs + 128)

    # (B, H, L, (192 + 128)*2 )
    DKV = (192 + 128) * 2
    buffer_ptr = tl.load(buffer_ptrs + RANK).to(tl.pointer_type(tl.bfloat16))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    kb_ptrs = (buffer_ptr +
               bid * H * L * DKV
               + hid * L * DKV
               + (offs_n[:, None] * DKV + offs_1[None, :]))
    vb_ptrs = (buffer_ptr +
               bid * H * L * DKV
               + hid * L * DKV
               + 192
               + (offs_n[:, None] * DKV + offs_0[None, :]))

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

        k0 = tl.load(k_ptrs + n * stride_k)
        k1 = tl.load(k_ptrs + n * stride_k + 64)
        k2 = tl.load(k_ptrs + n * stride_k + 128)

        qk = tl.dot(q0, tl.trans(k0))
        qk = tl.dot(q1, tl.trans(k1), qk)
        qk = tl.dot(q2, tl.trans(k2), qk)

        if tl.program_id(2) == 0:
            tl.store(kb_ptrs + n * DKV, k0)
            tl.store(kb_ptrs + n * DKV + 64, k1)
            tl.store(kb_ptrs + n * DKV + 128, k2)

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

        tl.store(vb_ptrs + n * DKV, v)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True, )

    for src_idx in range(SIZE):

        if src_idx != RANK:

            buffer_ptr = tl.load(buffer_ptrs + src_idx).to(tl.pointer_type(tl.bfloat16))
            buffer_ptr = tl.multiple_of(buffer_ptr, 16)
            kb_ptrs = (buffer_ptr +
                       bid * H * L * DKV
                       + hid * L * DKV
                       + (offs_n[:, None] * DKV + offs_1[None, :]))
            vb_ptrs = (buffer_ptr +
                       bid * H * L * DKV
                       + hid * L * DKV
                       + 192
                       + (offs_n[:, None] * DKV + offs_0[None, :]))

            # chunk 0
            if cid == 0:
                if src_idx < RANK:
                    for n in range(0, L // 2, N):
                        n = tl.multiple_of(n, N)

                        k0 = tl.load(kb_ptrs + n * DKV)
                        k1 = tl.load(kb_ptrs + n * DKV + 64)
                        k2 = tl.load(kb_ptrs + n * DKV + 128)

                        qk = tl.dot(q0, tl.trans(k0))
                        qk = tl.dot(q1, tl.trans(k1), qk)
                        qk = tl.dot(q2, tl.trans(k2), qk)

                        qk *= softmax_scale

                        if SAFE:
                            latest_max_logits = tl.maximum(max_logits,
                                                           tl.max(qk, 1))
                            p = tl.exp(qk - latest_max_logits[:, None])
                            rescale = tl.exp(max_logits - latest_max_logits)
                            lse = lse * rescale + tl.sum(p, 1)
                            v = tl.load(vb_ptrs + n * DKV)
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
                            v = tl.load(vb_ptrs + n * DKV)
                            p = p.to(V.dtype.element_ty)
                            acc_o = tl.dot(p, v, acc_o)

            # chunk 1
            if cid == 1:
                for n in range(0, L // 2, N):
                    n = tl.multiple_of(n, N)

                    k0 = tl.load(kb_ptrs + n * DKV)
                    k1 = tl.load(kb_ptrs + n * DKV + 64)
                    k2 = tl.load(kb_ptrs + n * DKV + 128)

                    qk = tl.dot(q0, tl.trans(k0))
                    qk = tl.dot(q1, tl.trans(k1), qk)
                    qk = tl.dot(q2, tl.trans(k2), qk)

                    qk *= softmax_scale

                    if SAFE:
                        latest_max_logits = tl.maximum(max_logits,
                                                       tl.max(qk, 1))
                        p = tl.exp(qk - latest_max_logits[:, None])
                        rescale = tl.exp(max_logits - latest_max_logits)
                        lse = lse * rescale + tl.sum(p, 1)
                        v = tl.load(vb_ptrs + n * DKV)
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
                        v = tl.load(vb_ptrs + n * DKV)
                        p = p.to(V.dtype.element_ty)
                        acc_o = tl.dot(p, v, acc_o)

                if src_idx > RANK:
                    for n in range(L // 2, L, N):
                        n = tl.multiple_of(n, N)

                        k0 = tl.load(kb_ptrs + n * DKV)
                        k1 = tl.load(kb_ptrs + n * DKV + 64)
                        k2 = tl.load(kb_ptrs + n * DKV + 128)

                        qk = tl.dot(q0, tl.trans(k0))
                        qk = tl.dot(q1, tl.trans(k1), qk)
                        qk = tl.dot(q2, tl.trans(k2), qk)

                        qk *= softmax_scale

                        if SAFE:
                            latest_max_logits = tl.maximum(max_logits,
                                                           tl.max(qk, 1))
                            p = tl.exp(qk - latest_max_logits[:, None])
                            rescale = tl.exp(max_logits - latest_max_logits)
                            lse = lse * rescale + tl.sum(p, 1)
                            v = tl.load(vb_ptrs + n * DKV)
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
                            v = tl.load(vb_ptrs + n * DKV)
                            p = p.to(V.dtype.element_ty)
                            acc_o = tl.dot(p, v, acc_o)

    acc_o = acc_o / lse[:, None]

    # [B, L, H, 128]
    out_ptrs = (Out
                + (bid * L + mid * M) * H * 128
                + hid * 128
                + (offs_m[:, None] * 128 * H + offs_0[None, :]))

    tl.store(out_ptrs, acc_o)
    tl.store(LSE + bid * H * L + hid * L + mid * M + tl.arange(0, M), lse)
    if SAFE:
        tl.store(ML + bid * H * L + hid * L + mid * M + tl.arange(0, M),
                 max_logits)


@triton.jit
def cp_mla_forward_kernel(
        Q,
        K,
        V,
        Out,
        LSE,
        ML,
        buffer_ptrs,
        signal_ptrs,
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
        SIZE: tl.constexpr,
        RANK: tl.constexpr, ):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    mid = tl.num_programs(2) - tl.program_id(2) - 1
    H = tl.num_programs(1)

    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))

    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)

    offs_0 = tl.arange(0, 128)
    offs_1 = tl.arange(0, 64)

    # [B, L, H, 192】
    q_ptrs = (Q
              + (bid * L + mid * M) * stride_q
              + hid * 192
              + (offs_m[:, None] * stride_q + offs_1[None, :]))

    k_ptrs = (K
              + bid * L * stride_k
              + hid * 192
              + (offs_n[:, None] * stride_k + offs_1[None, :]))

    v_ptrs = (V
              + bid * L * stride_v
              + hid * 128
              + (offs_n[:, None] * stride_v + offs_0[None, :]))

    # [B, L, H, 128]
    out_ptrs = (Out
                + (bid * L + mid * M) * H * 128
                + hid * 128
                + (offs_m[:, None] * 128 * H + offs_0[None, :]))
    lse_ptrs = LSE + bid * H * L + hid * L + mid * M + tl.arange(0, M)

    # (B, H, L, (192 + 128) * 2)
    # [q_buffer, o_buffer, lse_buffer]
    DKV = (192 + 128) * 2
    buffer_ptr = tl.load(buffer_ptrs + RANK).to(tl.pointer_type(tl.bfloat16))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    qb_ptrs = (buffer_ptr +
               bid * H * L * DKV
               + hid * L * DKV
               + mid * M * DKV
               + (offs_m[:, None] * DKV + offs_1[None, :]))
    lseb_ptrs = (buffer_ptr +
                 bid * H * L * DKV
                 + hid * L * DKV
                 + mid * DKV
                 + offs_m)

    # load q0
    q0 = tl.load(q_ptrs)
    q1 = tl.load(q_ptrs + 64)
    q2 = tl.load(q_ptrs + 128)

    # store q0 to buffer
    tl.store(qb_ptrs, q0)
    tl.store(qb_ptrs + 64, q1)
    tl.store(qb_ptrs + 128, q2)

    # reset o0 buffer to 0
    tl.store(qb_ptrs + 192, 0.0)
    tl.store(qb_ptrs + 256, 0.0)
    # reset lse0 buffer to 0
    tl.store(lseb_ptrs, 0.0)

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

    # q0 kv0
    for i in range(0, steps):
        n = i * N
        n = tl.multiple_of(n, N)

        k0 = tl.load(k_ptrs + n * stride_k)
        k1 = tl.load(k_ptrs + n * stride_k + 64)
        k2 = tl.load(k_ptrs + n * stride_k + 128)

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

    # store o0
    tl.store(out_ptrs, acc_o)
    tl.store(lse_ptrs, lse)
    if SAFE:
        tl.store(ML + bid * H * L + hid * L + mid * M + tl.arange(0, M),
                 max_logits)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True, )

    # q0 kv0
    for src_idx in range(SIZE):

        if src_idx > RANK:

            src_buffer_ptr = tl.load(buffer_ptrs + src_idx).to(tl.pointer_type(tl.bfloat16))
            src_buffer_ptr = tl.multiple_of(src_buffer_ptr, 16)
            src_qb_ptrs = (src_buffer_ptr +
                           bid * H * L * DKV
                           + hid * L * DKV
                           + mid * M * DKV
                           + (offs_m[:, None] * DKV + offs_1[None, :]))
            src_ob_ptrs = (src_buffer_ptr +
                           bid * H * L * DKV
                           + hid * L * DKV
                           + mid * M * DKV
                           + 192
                           + (offs_m[:, None] * DKV + offs_0[None, :]))
            # note that M <= 320
            src_lseb_ptrs = (src_buffer_ptr +
                             bid * H * L * DKV
                             + hid * L * DKV
                             + 320
                             + mid * DKV
                             + offs_m)
            q0 = tl.load(src_qb_ptrs)
            q1 = tl.load(src_qb_ptrs + 64)
            q2 = tl.load(src_qb_ptrs + 128)

            acc_o = tl.zeros((M, 128), dtype=tl.float32)
            if SAFE:
                max_logits = tl.zeros((M,), dtype=tl.float32) - 10000.0
                lse = tl.zeros((M,), dtype=tl.float32)
            else:
                lse = tl.zeros((M,), dtype=tl.float32) + 1e-30

            for n in range(0, L // 2, N):
                n = tl.multiple_of(n, N)

                k0 = tl.load(k_ptrs + n * stride_k)
                k1 = tl.load(k_ptrs + n * stride_k + 64)
                k2 = tl.load(k_ptrs + n * stride_k + 128)

                qk = tl.dot(q0, tl.trans(k0))
                qk = tl.dot(q1, tl.trans(k1), qk)
                qk = tl.dot(q2, tl.trans(k2), qk)

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
            tl.atomic_add(src_ob_ptrs, acc_o)
            tl.atomic_add(src_lseb_ptrs, lse)

    # load q1
    q0 = tl.load(q_ptrs + L // 2 * stride_q)
    q1 = tl.load(q_ptrs + L // 2 * stride_q + 64)
    q2 = tl.load(q_ptrs + L // 2 * stride_q + 128)

    # store q1 to buffer
    tl.store(qb_ptrs + L // 2 * DKV, q0)
    tl.store(qb_ptrs + L // 2 * DKV + 64, q1)
    tl.store(qb_ptrs + L // 2 * DKV + 128, q2)

    # reset o1 buffer to 0
    tl.store(qb_ptrs + L // 2 * DKV + 192, 0.0)
    tl.store(qb_ptrs + L // 2 * DKV + 256, 0.0)
    # reset lse1 buffer to 0
    tl.store(lseb_ptrs + L // 2 * DKV, 0.0)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True, )

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

    # q1 kv1
    for i in range(0, steps):
        n = L // 2 + i * N
        n = tl.multiple_of(n, N)

        k0 = tl.load(k_ptrs + n * stride_k)
        k1 = tl.load(k_ptrs + n * stride_k + 64)
        k2 = tl.load(k_ptrs + n * stride_k + 128)

        qk = tl.dot(q0, tl.trans(k0))
        qk = tl.dot(q1, tl.trans(k1), qk)
        qk = tl.dot(q2, tl.trans(k2), qk)

        if CAUSAL:
            qk += tl.where((L // 2 + mid * M + offs_m)[:, None] >= (n + offs_n)[None, :],
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

    # q1 kv0
    for n in range(0, L // 2, N):
        n = tl.multiple_of(n, N)

        k0 = tl.load(k_ptrs + n * stride_k)
        k1 = tl.load(k_ptrs + n * stride_k + 64)
        k2 = tl.load(k_ptrs + n * stride_k + 128)

        qk = tl.dot(q0, tl.trans(k0))
        qk = tl.dot(q1, tl.trans(k1), qk)
        qk = tl.dot(q2, tl.trans(k2), qk)

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

    # store o1
    tl.store(out_ptrs + L // 2 * H * 128, acc_o)
    tl.store(lse_ptrs + L // 2, lse)
    if SAFE:
        tl.store(ML + bid * H * L + hid * L + mid * M + tl.arange(0, M),
                 max_logits)

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

    # q1 
    for src_idx in range(SIZE):

        src_buffer_ptr = tl.load(buffer_ptrs + src_idx).to(tl.pointer_type(tl.bfloat16))
        src_buffer_ptr = tl.multiple_of(src_buffer_ptr, 16)
        src_qb_ptrs = (src_buffer_ptr +
                       bid * H * L * DKV
                       + hid * L * DKV
                       + L // 2 * DKV
                       + mid * M * DKV
                       + (offs_m[:, None] * DKV + offs_1[None, :]))
        src_ob_ptrs = (src_buffer_ptr +
                       bid * H * L * DKV
                       + hid * L * DKV
                       + L // 2 * DKV
                       + mid * M * DKV
                       + 192
                       + (offs_m[:, None] * DKV + offs_0[None, :]))
        src_lseb_ptrs = (src_buffer_ptr +
                         bid * H * L * DKV
                         + hid * L * DKV
                         + L // 2 * DKV
                         + 320
                         + mid * DKV
                         + offs_m)
        q0 = tl.load(src_qb_ptrs)
        q1 = tl.load(src_qb_ptrs + 64)
        q2 = tl.load(src_qb_ptrs + 128)

        acc_o = tl.zeros((M, 128), dtype=tl.float32)
        if SAFE:
            max_logits = tl.zeros((M,), dtype=tl.float32) - 10000.0
            lse = tl.zeros((M,), dtype=tl.float32)
        else:
            lse = tl.zeros((M,), dtype=tl.float32) + 1e-30

        # q1 kv0
        if src_idx != RANK:
            for n in range(0, L // 2, N):
                n = tl.multiple_of(n, N)

                k0 = tl.load(k_ptrs + n * stride_k)
                k1 = tl.load(k_ptrs + n * stride_k + 64)
                k2 = tl.load(k_ptrs + n * stride_k + 128)

                qk = tl.dot(q0, tl.trans(k0))
                qk = tl.dot(q1, tl.trans(k1), qk)
                qk = tl.dot(q2, tl.trans(k2), qk)

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

        # q1 kv1
        if src_idx < RANK:

            for n in range(L // 2, L, N):
                n = tl.multiple_of(n, N)

                k0 = tl.load(k_ptrs + n * stride_k)
                k1 = tl.load(k_ptrs + n * stride_k + 64)
                k2 = tl.load(k_ptrs + n * stride_k + 128)

                qk = tl.dot(q0, tl.trans(k0))
                qk = tl.dot(q1, tl.trans(k1), qk)
                qk = tl.dot(q2, tl.trans(k2), qk)

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

        tl.atomic_add(src_ob_ptrs, acc_o)
        tl.atomic_add(src_lseb_ptrs, lse)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True, )

    ob_ptrs = (buffer_ptr +
               bid * H * L * DKV
               + hid * L * DKV
               + mid * M * DKV
               + 192
               + (offs_m[:, None] * DKV + offs_0[None, :]))
    lseb_ptrs = (buffer_ptr +
                 bid * H * L * DKV
                 + hid * L * DKV
                 + 320
                 + mid * DKV
                 + offs_m)

    o = tl.load(out_ptrs)
    ob = tl.load(ob_ptrs)
    o += ob

    lse = tl.load(lse_ptrs)
    lseb = tl.load(lseb_ptrs)
    lse += lseb
    o = o / lse[:, None]

    tl.store(out_ptrs, o)
    tl.store(lse_ptrs, lse)

    o = tl.load(out_ptrs + L // 2 * H * 128)
    ob = tl.load(ob_ptrs + L // 2 * DKV)
    o += ob

    lse = tl.load(lse_ptrs + L // 2)
    lseb = tl.load(lseb_ptrs + L // 2 * DKV)
    lse += lseb
    o = o / lse[:, None]

    tl.store(out_ptrs + L // 2 * H * 128, o)
    tl.store(lse_ptrs + L // 2, lse)


def triton_cp_mla_forward(q, k, v, hdl, group, causal=True, safe=True,
                          clip_value=0.0):
    assert not safe  # TODO: use cross device online softmax
    B, L, H, D = q.shape
    h = k.shape[2]
    assert H == h, "triton_cp_mla_forward does NOT support GQA currently"
    assert D == 192

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

    clip = clip_value > 0.0
    clip_value = clip_value * softmax_scale if clip else 0.0

    if clip and clip_value + math.log(L) < 88.7:
        safe = False

    num_m_block = L // 2 // M
    num_stages = 2
    num_warps = 8
    group_size = group.size()
    group_rank = group.rank()

    grid = (B, H, num_m_block)
    cp_mla_forward_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        max_logits,
        hdl.buffer_ptrs_dev,
        hdl.signal_pad_ptrs_dev,
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
        group_size,
        group_rank,
        num_warps=num_warps,
        num_stages=num_stages, )

    return o, lse, max_logits


@triton.jit
def cp_mla_backward_kernel(
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
        buffer_ptrs,
        signal_ptrs,
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
        SIZE: tl.constexpr,
        RANK: tl.constexpr, ):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    nid = tl.program_id(2)
    H = tl.num_programs(1).to(tl.int64)

    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))

    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)

    offs_0 = tl.arange(0, 128)  # nope
    offs_1 = tl.arange(0, 64)  # pe

    # [B, L, H, 192]
    q0_ptrs = (Q
               + bid * L * stride_q
               + hid * 192
               + (offs_m[:, None] * stride_q + offs_0[None, :]))
    q1_ptrs = (Q
               + bid * L * stride_q
               + hid * 192
               + 128
               + (offs_m[:, None] * stride_q + offs_1[None, :]))

    k0_ptrs = (K
               + (bid * L + nid * N) * stride_k
               + hid * 192
               + (offs_n[:, None] * stride_k + offs_0[None, :]))
    k1_ptrs = (K
               + (bid * L + nid * N) * stride_k
               + hid * 192
               + 128
               + (offs_n[:, None] * stride_k + offs_1[None, :]))
    v_ptrs = (V
              + (bid * L + nid * N) * stride_v
              + hid * 128
              + (offs_n[:, None] * stride_v + offs_0[None, :]))

    k0 = tl.load(k0_ptrs)
    k1 = tl.load(k1_ptrs)
    v = tl.load(v_ptrs)

    DKV = (192 + 128) * 2
    buffer_ptr = tl.load(buffer_ptrs + RANK).to(tl.pointer_type(tl.bfloat16))
    buffer_ptr = tl.multiple_of(buffer_ptr, 16)
    kb0_ptrs = (buffer_ptr
                + bid * H * L * DKV
                + hid * L * DKV
                + nid * N * DKV
                + (offs_n[:, None] * DKV + offs_0[None, :]))
    kb1_ptrs = (buffer_ptr
                + bid * H * L * DKV
                + hid * L * DKV
                + nid * N * DKV
                + 128
                + (offs_n[:, None] * DKV + offs_1[None, :]))

    # store kv 0 to buffer
    tl.store(kb0_ptrs, k0)
    tl.store(kb1_ptrs, k1)
    tl.store(kb0_ptrs + 192, v)

    # reset grad 0 to zero
    tl.store(kb0_ptrs + 320, 0.0)
    tl.store(kb1_ptrs + 320, 0.0)
    tl.store(kb0_ptrs + 320 + 192, 0.0)

    # reset grad 1 to zero
    tl.store(kb0_ptrs + L // 2 * DKV + 320, 0.0)
    tl.store(kb1_ptrs + L // 2 * DKV + 320, 0.0)
    tl.store(kb0_ptrs + L // 2 * DKV + 320 + 192, 0.0)

    tl.debug_barrier()

    go_ptrs = (GO
               + bid * L * H * 128
               + hid * 128
               + (offs_m[:, None] * 128 * H + offs_0[None, :]))

    # [B, L, H, 192]
    dq0_ptrs = (GQ
                + bid * L * H * 192
                + hid * 192
                + (offs_m[:, None] * H * 192 + offs_0[None, :]))
    dq1_ptrs = (GQ
                + bid * L * H * 192
                + hid * 192
                + 128
                + (offs_m[:, None] * H * 192 + offs_1[None, :]))

    gk0_ptrs = (GK
                + (bid * L + nid * N) * H * 192
                + hid * 192
                + (offs_n[:, None] * 192 * H + offs_0[None, :]))

    gk1_ptrs = (GK
                + (bid * L + nid * N) * H * 192
                + hid * 192
                + 128
                + (offs_n[:, None] * 192 * H + offs_1[None, :]))

    gv_ptrs = (GV
               + (bid * L + nid * N) * H * 128
               + hid * 128
               + (offs_n[:, None] * 128 * H + offs_0[None, :]))

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
            max_logits = tl.load(ML + bid * H * L + hid * L + m + tl.arange(0, M))
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
        tl.atomic_add(dq0_ptrs + m * H * 192, dq0, sem='relaxed')
        tl.atomic_add(dq1_ptrs + m * H * 192, dq1, sem='relaxed')

        dp = tl.trans(dp)
        dk0 = tl.dot(dp, q0, dk0)  # [N, M]@[M, 128]=[N, 128]
        dk1 = tl.dot(dp, q1, dk1)  # [N, M]@[M, 64]=[N, 64]

    tl.atomic_add(gk0_ptrs, dk0, sem='relaxed')
    tl.atomic_add(gk1_ptrs, dk1, sem='relaxed')
    tl.atomic_add(gv_ptrs, dv, sem='relaxed')

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True, )

    # chunk 0
    for src_idx in range(SIZE):

        if src_idx != RANK:

            src_buffer_ptr = tl.load(buffer_ptrs + src_idx).to(tl.pointer_type(tl.bfloat16))
            src_buffer_ptr = tl.multiple_of(src_buffer_ptr, 16)
            src_kb0_ptrs = (src_buffer_ptr
                            + bid * H * L * DKV
                            + hid * L * DKV
                            + nid * N * DKV
                            + (offs_n[:, None] * DKV + offs_0[None, :]))
            src_kb1_ptrs = (src_buffer_ptr
                            + bid * H * L * DKV
                            + hid * L * DKV
                            + nid * N * DKV
                            + 128
                            + (offs_n[:, None] * DKV + offs_1[None, :]))

            dk0 = tl.zeros((N, 128), dtype=tl.float32)
            dk1 = tl.zeros((N, 64), dtype=tl.float32)
            dv = tl.zeros((N, 128), dtype=tl.float32)

            k0 = tl.load(src_kb0_ptrs)
            k1 = tl.load(src_kb1_ptrs)
            v = tl.load(src_kb0_ptrs + 192)

            # kv: chunk 0
            # q: chunk 0
            if src_idx < RANK:
                for m in range(0, L // 2, M):
                    lse = 1 / tl.load(LSE + bid * H * L + hid * L + m + tl.arange(0, M))
                    if SAFE:
                        max_logits = tl.load(ML + bid * H * L + hid * L + m + tl.arange(0, M))
                    ds = tl.load(DS + bid * H * L + hid * L + m + tl.arange(0, M))

                    q0 = tl.load(q0_ptrs + m * stride_q)
                    q1 = tl.load(q1_ptrs + m * stride_q)
                    go = tl.load(go_ptrs + m * H * 128)

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
                    tl.atomic_add(dq0_ptrs + m * H * 192, dq0, sem='relaxed')
                    tl.atomic_add(dq1_ptrs + m * H * 192, dq1, sem='relaxed')

                    dp = tl.trans(dp)
                    dk0 = tl.dot(dp, q0, dk0)  # [N, M]@[M, 128]=[N, 128]
                    dk1 = tl.dot(dp, q1, dk1)  # [N, M]@[M, 64]=[N, 64]

            # kv: chunk 0
            # q: chunk 1
            for m in range(L // 2, L, M):
                lse = 1 / tl.load(LSE + bid * H * L + hid * L + m + tl.arange(0, M))
                if SAFE:
                    max_logits = tl.load(ML + bid * H * L + hid * L + m + tl.arange(0, M))
                ds = tl.load(DS + bid * H * L + hid * L + m + tl.arange(0, M))

                q0 = tl.load(q0_ptrs + m * stride_q)
                q1 = tl.load(q1_ptrs + m * stride_q)

                go = tl.load(go_ptrs + m * H * 128)

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
                tl.atomic_add(dq0_ptrs + m * H * 192, dq0, sem='relaxed')
                tl.atomic_add(dq1_ptrs + m * H * 192, dq1, sem='relaxed')

                dp = tl.trans(dp)
                dk0 = tl.dot(dp, q0, dk0)  # [N, M]@[M, 128]=[N, 128]
                dk1 = tl.dot(dp, q1, dk1)  # [N, M]@[M, 64]=[N, 64]

            tl.atomic_add(src_kb0_ptrs + 320, dk0, sem='relaxed')
            tl.atomic_add(src_kb1_ptrs + 320, dk1, sem='relaxed')
            tl.atomic_add(src_kb0_ptrs + 192 + 320, dv, sem='relaxed')

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True, )

    tl.debug_barrier()
    src_dk0 = tl.load(kb0_ptrs + 320)
    tl.atomic_add(gk0_ptrs, src_dk0)

    src_dk1 = tl.load(kb1_ptrs + 320)
    tl.atomic_add(gk1_ptrs, src_dk1)

    src_dv = tl.load(kb0_ptrs + 192 + 320)
    tl.atomic_add(gv_ptrs, src_dv)

    tl.debug_barrier()

    # kv 1
    k0 = tl.load(k0_ptrs + L // 2 * stride_k)
    k1 = tl.load(k1_ptrs + L // 2 * stride_k)
    v = tl.load(v_ptrs + L // 2 * stride_v)

    # store kv 1 to buffer
    tl.store(kb0_ptrs + L // 2 * DKV, k0)
    tl.store(kb1_ptrs + L // 2 * DKV, k1)
    tl.store(kb0_ptrs + L // 2 * DKV + 192, v)

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True, )

    dv = tl.zeros((N, 128), dtype=tl.float32)
    dk0 = tl.zeros((N, 128), dtype=tl.float32)
    dk1 = tl.zeros((N, 64), dtype=tl.float32)
    if CAUSAL:
        step = nid * N + L // 2
        n_steps = tl.cdiv(L - step, M)
    else:
        step = 0
        n_steps = tl.cdiv(L, M)

    for i in range(n_steps):
        m = step + (n_steps - 1 - i) * M
        lse = 1 / tl.load(LSE + bid * H * L + hid * L + m + tl.arange(0, M))
        if SAFE:
            max_logits = tl.load(ML + bid * H * L + hid * L + m + tl.arange(0, M))
        ds = tl.load(DS + bid * H * L + hid * L + m + tl.arange(0, M))

        q0 = tl.load(q0_ptrs + m * stride_q)
        q1 = tl.load(q1_ptrs + m * stride_q)
        go = tl.load(go_ptrs + m * H * 128)

        if CAUSAL:
            qk = tl.where((m + offs_m)[:, None] >= (L // 2 + nid * N + offs_n)[None, :],
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
        tl.atomic_add(dq0_ptrs + m * H * 192, dq0, sem='relaxed')
        tl.atomic_add(dq1_ptrs + m * H * 192, dq1, sem='relaxed')

        dp = tl.trans(dp)
        dk0 = tl.dot(dp, q0, dk0)  # [N, M]@[M, 128]=[N, 128]
        dk1 = tl.dot(dp, q1, dk1)  # [N, M]@[M, 64]=[N, 64]

    tl.atomic_add(gk0_ptrs + L // 2 * H * 192, dk0, sem='relaxed')
    tl.atomic_add(gk1_ptrs + L // 2 * H * 192, dk1, sem='relaxed')
    tl.atomic_add(gv_ptrs + L // 2 * H * 128, dv, sem='relaxed')

    for src_idx in range(SIZE):
        src_buffer_ptr = tl.load(buffer_ptrs + src_idx).to(tl.pointer_type(tl.bfloat16))
        src_buffer_ptr = tl.multiple_of(src_buffer_ptr, 16)
        src_kb0_ptrs = (src_buffer_ptr
                        + bid * H * L * DKV
                        + hid * L * DKV
                        + L // 2 * DKV
                        + nid * N * DKV
                        + (offs_n[:, None] * DKV + offs_0[None, :]))
        src_kb1_ptrs = (src_buffer_ptr
                        + bid * H * L * DKV
                        + hid * L * DKV
                        + L // 2 * DKV
                        + nid * N * DKV
                        + 128
                        + (offs_n[:, None] * DKV + offs_1[None, :]))

        if src_idx != RANK:

            dk0 = tl.zeros((N, 128), dtype=tl.float32)
            dk1 = tl.zeros((N, 64), dtype=tl.float32)
            dv = tl.zeros((N, 128), dtype=tl.float32)

            k0 = tl.load(src_kb0_ptrs)
            k1 = tl.load(src_kb1_ptrs)
            v = tl.load(src_kb0_ptrs + 192)

            # second chunk
            if src_idx > RANK:
                for m in range(L // 2, L, M):
                    lse = 1 / tl.load(LSE + bid * H * L + hid * L + m + tl.arange(0, M))
                    if SAFE:
                        max_logits = tl.load(ML + bid * H * L + hid * L + m + tl.arange(0, M))
                    ds = tl.load(DS + bid * H * L + hid * L + m + tl.arange(0, M))

                    q0 = tl.load(q0_ptrs + m * stride_q)
                    q1 = tl.load(q1_ptrs + m * stride_q)
                    go = tl.load(go_ptrs + m * H * 128)

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
                    tl.atomic_add(dq0_ptrs + m * H * 192, dq0, sem='relaxed')
                    tl.atomic_add(dq1_ptrs + m * H * 192, dq1, sem='relaxed')

                    dp = tl.trans(dp)
                    dk0 = tl.dot(dp, q0, dk0)  # [N, M]@[M, 128]=[N, 128]
                    dk1 = tl.dot(dp, q1, dk1)  # [N, M]@[M, 64]=[N, 64]

                tl.atomic_add(src_kb0_ptrs + 320, dk0, sem='relaxed')
                tl.atomic_add(src_kb1_ptrs + 320, dk1, sem='relaxed')
                tl.atomic_add(src_kb0_ptrs + 192 + 320, dv, sem='relaxed')

    symm_mem_sync(
        signal_ptrs,
        None,
        RANK,
        SIZE,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True, )

    src_dk0 = tl.load(kb0_ptrs + L // 2 * DKV + 320)
    tl.atomic_add(gk0_ptrs + L // 2 * H * 192, src_dk0)

    src_dk1 = tl.load(kb1_ptrs + L // 2 * DKV + 320)
    tl.atomic_add(gk1_ptrs + L // 2 * H * 192, src_dk1)

    src_dv = tl.load(kb0_ptrs + L // 2 * DKV + 192 + 320)
    tl.atomic_add(gv_ptrs + L // 2 * H * 128, src_dv)


def triton_cp_mla_backward(go, o, q, k, v, lse, max_logits, hdl, group,
                           causal=True, safe=True,
                           hpc=False, clip_value=0.0):
    # q: [B, L, H, 192]
    # k: [B, L, H, 192]
    # v: [B, L, H, 128]
    B, L, H, _ = q.shape
    assert k.size(1) == L

    device = q.device
    dtype = q.dtype

    ds = torch.empty((B, H, L), dtype=torch.float32, device=device)

    softmax_scale = 128 ** (-0.5)
    clip = clip_value > 0.0
    clip_value = clip_value * softmax_scale if clip else 0.0

    M = 64
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
        num_stages=num_stages)

    M = 32
    N = 128
    gq = torch.zeros((B, L, H, 192), dtype=torch.float32 if hpc else dtype,
                     device=device)
    gk = torch.zeros((B, L, H, 192), dtype=torch.float32 if hpc else dtype,
                     device=device)
    gv = torch.zeros((B, L, H, 128), dtype=torch.float32 if hpc else dtype,
                     device=device)
    assert L % M == 0
    assert L % N == 0
    assert N >= M
    group_size = group.size()
    group_rank = group.rank()
    num_n_block = L // 2 // N
    num_warps = 8
    num_stages = 5
    grid = (B, H, num_n_block)
    cp_mla_backward_kernel[grid](
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
        hdl.buffer_ptrs_dev,
        hdl.signal_pad_ptrs_dev,
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
        group_size,
        group_rank,
        num_warps=num_warps,
        num_stages=num_stages, )

    gq = gq.to(q.dtype)
    gk = gk.to(q.dtype)
    gv = gv.to(v.dtype)
    return gq, gk, gv
