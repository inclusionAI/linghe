# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math

import torch

from linghe.attn.la import (triton_lightning_attention_forward,
                            triton_lightning_attention_backward,
                            triton_fused_lightning_attention_backward)
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check


def torch_la(q, k, v, s, decay_scales):
    dtype = q.dtype

    q = q.double()
    k = k.double()
    v = v.double()
    s = s.double()
    bs, q_len, q_heads, head_dim = q.shape
    k_heads = k.shape[2]
    k_len = k.shape[1]
    assert q_len == k_len
    softmax_scale = 1.0 / math.sqrt(head_dim)
    query = q.transpose(1, 2)  # [bs, head, len, dim]
    key = torch.permute(k, (0, 2, 3, 1))  # [bs, head, dim, len]
    value = v.transpose(1, 2)  # [bs, head, len, dim]
    if k_heads != q_heads:
        g = q_heads // k_heads
        key = torch.repeat_interleave(key, g, dim=1)
        value = torch.repeat_interleave(value, g, dim=1)

    arr = torch.arange(q_len, dtype=torch.float64, device=q.device)
    decay_matrix = arr.view(-1, 1) - arr.view(1, -1)
    decay_matrix = torch.exp(-decay_scales[:, None, None] * decay_matrix[None])
    decay_matrix = torch.tril(decay_matrix, 0)

    score = torch.matmul(query, key) * softmax_scale
    score *= decay_matrix[None]
    att = torch.matmul(score, value)

    decay_arr = torch.exp(-decay_scales[:, None, None] * (arr[:, None] + 1))
    att = att + torch.matmul(query * decay_arr, s)

    att = torch.reshape(att.transpose(1, 2),
                        [bs, q_len, q_heads, head_dim]).contiguous()

    decay_key = key * torch.exp(
        -decay_scales[:, None, None] * (q_len - 1 - arr))
    state = decay_key @ value + s * torch.exp(-decay_scales[:, None, None])

    return att.to(dtype), state.to(torch.float32)


def torch_varlen_torch_la(q, k, v, s, cu_seqlens, padded_cu_seqlens,
                          decay_scales):
    pass


def make_varlen_input(qo_heads=16, kv_heads=16, dim=128, qls=[1024, 1024],
                      kls=[1024, 1024]):
    device = torch.device('cuda:0')
    dtype = torch.bfloat16
    qs = []
    ks = []
    vs = []
    for i, ql in enumerate(qls):
        kvl = kls[i]
        q = torch.randn(ql, qo_heads, dim, dtype=dtype, device=device)
        k = torch.randn(kvl, kv_heads, dim, dtype=dtype, device=device)
        v = torch.randn(kvl, kv_heads, dim, dtype=dtype, device=device)
        qs.append(q)
        ks.append(k)
        vs.append(v)

    q = torch.cat(qs, 0)
    k = torch.cat(ks, 0)
    v = torch.cat(vs, 0)

    return q, k, v


def test_la(bs=1, length=4096, qo_heads=16, kv_heads=16, dim=128, digest=False,
            bench=False):
    device = torch.device('cuda:0')
    dtype = torch.bfloat16

    q = torch.randn(bs, length, qo_heads, dim, dtype=dtype,
                    device=device) ** 3 * 0.1
    q = q.requires_grad_()

    k = torch.randn(bs, length, kv_heads, dim, dtype=dtype,
                    device=device) ** 3 * 0.1
    k = k.requires_grad_()

    v = torch.randn(bs, length, kv_heads, dim, dtype=dtype,
                    device=device) ** 3 * 0.1
    v = v.requires_grad_()

    g = torch.randn(bs, length, qo_heads, dim, dtype=dtype, device=device)

    s = torch.zeros(bs, kv_heads, dim, dim, dtype=torch.float32, device=device)

    decay_scales = 2 ** (
                -0.5 * torch.arange(1, qo_heads + 1, dtype=torch.float32,
                                    device=device))
    # decay_scales = 0.0 * torch.ones(qo_heads, dtype=torch.float32, device=device)
    output_ref, state_ref = torch_la(q, k, v, s, decay_scales)
    output_ref.backward(g)
    dq_ref = q.grad
    dk_ref = k.grad
    dv_ref = v.grad
    q.grad = None
    k.grad = None
    v.grad = None

    output, state = triton_lightning_attention_forward(q, k, v, decay_scales)

    output_check(output_ref, output, name='output', rtol=0.1, atol=0.2)
    output_check(state_ref, state, name='state', rtol=0.1, atol=0.2)

    dq, dk, dv = triton_lightning_attention_backward(g, q, k, v, decay_scales)

    output_check(dq_ref, dq, name='dq', rtol=-0.1, atol=1.0)
    output_check(dk_ref, dk, name='dk', rtol=-0.1, atol=1.0)
    output_check(dv_ref, dv, name='dv', rtol=-0.1, atol=1.0)

    max_decay_scale = decay_scales.max().item()
    if max_decay_scale < 0.1:
        dq, dk, dv = triton_fused_lightning_attention_backward(g, q, k, v,
                                                               state,
                                                               decay_scales)
        output_check(dq_ref, dq, name='dq', rtol=-0.1, atol=0.1)
        output_check(dk_ref, dk, name='dk', rtol=-0.1, atol=0.1)
        output_check(dv_ref, dv, name='dv', rtol=-0.1, atol=0.1)

    if bench:
        ref_bytes = bs * length * qo_heads * dim * 8 + bs * qo_heads * dim * dim * 8
        benchmark_func(triton_lightning_attention_forward, q, k, v,
                       decay_scales, ref_bytes=ref_bytes)
        benchmark_func(triton_lightning_attention_backward, g, q, k, v,
                       decay_scales, ref_bytes=ref_bytes)
        benchmark_func(triton_fused_lightning_attention_backward, g, q, k, v,
                       state, decay_scales, ref_bytes=ref_bytes)


# def test_varlen_la(qls=[1024,1024], qo_heads=16, kv_heads=16, dim=128, digest=False, bench=False):
#     device = torch.device('cuda:0')
#     dtype = torch.bfloat16
#     kls = qls

#     assert all([x<=kls[i] for i,x in enumerate(qls)])
#     bs = len(qls)

#     ref_bytes = sum(qls) * qo_heads * dim * 8 + bs * qo_heads * dim * dim * 8

#     q, k, v = make_input(qo_heads=qo_heads, kv_heads=kv_heads, dim=dim, qls=qls, kls=kls)

#     s = torch.zeros(bs, kv_heads, dim, dim, dtype=torch.float32, device=device)

#     decay_scales = 2**(-0.5 * torch.arange(1, qo_heads+1, dtype=torch.float32, device=device))
#     # decay_scales = 2**(-0.5 * torch.ones(qo_head, dtype=torch.float32, device=device))
#     lengths = torch.tensor([0] + qls, device=device, dtype=torch.long)
#     cu_seqlens = torch.cumsum(lengths, 0)
#     padded_cu_seqlens = cu_seqlens

#     output_ref, state_ref = torch_varlen_linear_attn(q, k, v, s, decay_scales, cu_seqlens, padded_cu_seqlens)

#     max_q_length = max(qls)
#     output = triton_lightning_attention_forward(q, k, v, decay_scales, cu_seqlens, padded_cu_seqlens, max_q_length)

#     output_check(output_ref, output, name='output', rtol=0.1, atol=0.1)
#     output_check(state_ref, s, name='state', rtol=0.01, atol=0.01)

#     if digest:
#         print(
#             f"output_ref max:{torch.max(output_ref).item():.3f} min:{torch.min(output_ref).item():.3f}")
#         print(
#             f"output max:{torch.max(output).item():.3f} min:{torch.min(output).item():.3f}")

#         print("output_ref[:,0,0]", output_ref[:, 0, 0])
#         print("output[:,0,0]", output[:, 0, 0])

#         print("output_ref[0,:,0]", output_ref[0, :, 0])
#         print("output[0,:,0]", output[0, :, 0])

#         print("output_ref[0,0,:]", output_ref[0, 0, :])
#         print("output[0,0,:]", output[0, 0, :])

#     if bench:
#         benchmark_func(triton_lightning_attention_forward, q, k, v, decay_scales, cu_seqlens, padded_cu_seqlens, max_q_length, ref_bytes=ref_bytes)


if __name__ == '__main__':
    test_la(bs=1, length=8192, qo_heads=64, kv_heads=64, dim=128, digest=False,
            bench=True)
