# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math

import torch
import triton
import triton.language as tl


"""
deepseek R1 MLA kernel
head_num: 128

q_lora: 1536
q_nope: 128
q_rope: 64

kv_lora: 512
kv_nope: 128
kv_rope: 64

"""


@triton.jit
def single_seg_mla_kernel(
    Q,
    KV,
    Out,
    softmax_scale,
    stride_q,
    stride_k,
    stride_o,
    q_offsets,
    k_offsets,
    q_lengths,
    k_lengths,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):

    bid = tl.program_id(0)
    mid = tl.num_programs(1) - tl.program_id(1) - 1
    q_length = tl.load(q_lengths + bid)
    if mid >= q_length:
        return

    q_offset = tl.load(q_offsets + bid)
    k_offset = tl.load(k_offsets + bid)
    k_length = tl.load(k_lengths + bid)
    offs_n = tl.arange(0, BLOCK)
    offs_h = tl.arange(0, 128)
    offs_v = tl.arange(0, 512)
    offs_p = tl.arange(0, 64)

    max_n = k_length - q_length + mid + 1

    q0_ptrs = (
        Q
        + q_offset * stride_q
        + mid * stride_q
        + (offs_h[:, None] * 576 + offs_v[None, :])
    )
    q1_ptrs = (
        Q
        + q_offset * stride_q
        + mid * stride_q
        + 512
        + (offs_h[:, None] * 576 + offs_p[None, :])
    )
    k0_ptrs = (KV + k_offset * stride_k) + (
        offs_n[:, None] * stride_k + offs_v[None, :]
    )
    k1_ptrs = (KV + k_offset * stride_k + 512) + (
        offs_n[:, None] * stride_k + offs_p[None, :]
    )

    lse = tl.zeros([128], dtype=tl.float32)
    acc_o = tl.zeros([128, 512], dtype=tl.float32)

    q0 = tl.load(q0_ptrs)
    q1 = tl.load(q1_ptrs)
    steps = tl.cdiv(max_n, BLOCK)

    for i in range(0, steps):
        n = i * BLOCK

        if EVEN:
            k0 = tl.load(k0_ptrs + n * stride_k)
        else:
            k0 = tl.load(
                k0_ptrs + n * stride_k, mask=(n + offs_n)[:, None] < max_n, other=0.0
            )
        qk = tl.dot(q0, tl.trans(k0))

        if EVEN:
            k1 = tl.load(k1_ptrs + n * stride_k)
        else:
            k1 = tl.load(
                k1_ptrs + n * stride_k, mask=(n + offs_n)[:, None] < max_n, other=0.0
            )

        qk = tl.dot(q1, tl.trans(k1), qk)

        # pad mask
        qk += tl.where((n + offs_n)[None, :] < max_n, 0.0, float("-inf"))

        p = tl.exp(qk * softmax_scale)
        lse += tl.sum(p, 1)

        p = p.to(KV.dtype.element_ty)
        acc_o = tl.dot(p, k0, acc_o)

    acc_o = acc_o / lse[:, None]

    out_ptrs = (
        Out
        + q_offset * stride_o
        + mid * stride_o
        + (offs_h[:, None] * 512 + offs_v[None, :])
    )

    tl.store(out_ptrs, acc_o)

def seg_mla_fwd(q, kv, meta):
    # q: [q_length, 128, 576]
    # kv: [cache_size, 576]
    q_length, n_heads, q_dim = q.shape
    batch = meta.batch_size
    # assert n_heads == 128 and q_dim == 576
    softmax_scale = 1.0 / math.sqrt(128)

    o = torch.empty((q_length, n_heads, 512), device=q.device, dtype=q.dtype)

    assert meta.max_seg == 1

    BLOCK = 64

    if isinstance(meta.kls[0], (list, tuple)):
        EVEN = all([all([y % BLOCK == 0 for y in x]) for x in meta.kls])
    else:
        EVEN = all([x % BLOCK == 0 for x in meta.kls])

    num_m_block = meta.max_q_length
    num_warps = 8
    num_stages = 1

    grid = lambda META: (batch, num_m_block)
    single_seg_mla_kernel[grid](
        q,
        kv,
        o,
        softmax_scale,
        q.stride(0),
        kv.stride(0),
        o.stride(0),
        meta.q_offsets,
        meta.k_offsets,
        meta.q_lengths,
        meta.k_lengths,
        BLOCK=BLOCK,
        EVEN=EVEN,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o
