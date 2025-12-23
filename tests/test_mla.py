# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math
import torch

from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.attn.mla import triton_mla_forward, triton_mla_backward, triton_mp_mla_forward


def torch_attn(q, k, v, causal=True, mask=None, hp=False):
    dtype = q.dtype
    if hp:
        q = q.float()
        k = k.float()
        v = v.float()
    bs, q_len, q_head, q_head_dim = q.shape
    v_head_dim = v.shape[-1]
    k_head = k.shape[2]
    k_len = k.shape[1]
    if mask is None:
        if causal:
            mask = -10000 * torch.triu(
                torch.ones((q_len, k_len), dtype=q.dtype, device="cuda:0"),
                k_len - q_len + 1,
            )
        else:
            mask = torch.zeros((q_len, k_len), dtype=q.dtype, device="cuda:0")

    query = q.transpose(1, 2)
    key = torch.permute(k, (0, 2, 3, 1))
    value = v.transpose(1, 2)
    if k_head != q_head:
        g = q_head // k_head
        key = torch.repeat_interleave(key, g, dim=1)
        value = torch.repeat_interleave(value, g, dim=1)
    score = torch.matmul(query, key) / math.sqrt(v_head_dim) + mask
    lse = torch.sum(torch.exp(score), -1)
    max_logits = torch.amax(score, -1)
    prob = torch.softmax(score, dim=-1, dtype=torch.float32) 
    if not hp:
        prob = prob.to(dtype)
    att = torch.matmul(prob, value)
    att = torch.reshape(att.transpose(1, 2), [bs, q_len, q_head, v_head_dim]).contiguous()
    return att.to(dtype), lse, max_logits


def torch_softmax(x):
    prob = torch.softmax(x, dim=-1, dtype=torch.float32)
    return prob.to(x.dtype)

def torch_softmax_backward(x, g):
    p = torch.softmax(x, dim=-1, dtype=torch.float32)
    # (dp * p - p * tl.sum(p * dp, 1)[:,None])
    gi =  g * p - p * torch.sum(p * g, 1)[:, None]
    return gi.to(x.dtype)

def head_wise_quant(x):
    x = x.float()
    maxs = x.abs().amax(-1)
    scales = torch.maximum(maxs/448, maxs*0.0+1e-30)
    x_q = (x/scales[...,None]).to(torch.float8_e4m3fn)
    x_s = scales.permute(0,2,1).contiguous()
    return x_q, x_s


def test_softmax(M=128, N=128):
    x = torch.randn((N, N), dtype=torch.bfloat16, device='cuda:0', requires_grad=True)
    g = torch.randn((N, N), dtype=torch.bfloat16, device='cuda:0')
    y_ref = torch_softmax(x)
    y_ref.backward(g)
    grad_ref = x.grad 
    grad = torch_softmax_backward(x, g)
    output_check(grad_ref, grad, atol=10, name='grad')

def test_dot_sum(M=128, N=128, D=128):
    p = torch.randn((M, N), dtype=torch.float32)
    v = torch.randn((N, D), dtype=torch.float32)
    g = torch.randn((M, D), dtype=torch.float32)
    ds_ref = ((g@v.T)*p).sum(1)
    ds = ((p@v)*g).sum(1)
    output_check(ds_ref, ds, atol=10, name='dot_sum')



def test_mla(B=2, L=4096, H=16, causal=True, hpc=False, safe=True, coef=1.0, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'
    q = (torch.randn((B, L, H, 192), device=device, dtype=dtype)*coef).requires_grad_()
    k = torch.randn((B, L, H, 192), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((B, L, H, 128), device=device, dtype=dtype, requires_grad=True)
    g = torch.randn((B, L, H, 128), device=device, dtype=dtype, requires_grad=True)

    output_ref, lse_ref, max_logits_ref = torch_attn(q, k, v, causal=causal, hp=True)
    output_ref.backward(g, retain_graph=False)
    gq_ref = q.grad 
    gk_ref = k.grad 
    gv_ref = v.grad

    q.grad = None
    k.grad = None
    v.grad = None

    output, lse, max_logits = triton_mla_forward(q, k, v, causal=causal, safe=safe)
    output_check(output_ref, output, atol=0.05, rtol=0.05, name='output')
    # output_check(lse_ref.float(), lse, atol=0.05, rtol=0.05, name='lse')
    # output_check(max_logits_ref, max_logits, atol=0.01, rtol=0.03, name='max_logits')

    gq, gk, gv = triton_mla_backward(g, output, q, k, v, lse, max_logits, causal=causal, hpc=hpc, safe=safe)
    output_check(gv_ref, gv, atol=0.05, rtol=0.05, name='gv')
    output_check(gk_ref, gk, atol=0.05 * coef, rtol=0.05, name='gk')
    output_check(gq_ref, gq, atol=0.05 * coef, rtol=0.05, name='gq')

    if bench:
        ref_flops = B * L * L * H * (192 + 128) * (1 if causal else 2)
        benchmark_func(triton_mla_forward, q, k, v, causal=causal, safe=safe, ref_flops=ref_flops)
        ref_flops = B * L * L * H * (192 + 128 * 2 + 192 * 2) * (1 if causal else 2)
        benchmark_func(triton_mla_backward, g, output, q, k, v, lse, max_logits, 
                       causal=causal, hpc=hpc, safe=safe,
                       ref_flops=ref_flops, 
                       n_profile=0)


def test_mp_mla(B=2, L=4096, H=16, causal=True, hpc=False, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'
    q = torch.randn((B, L, H, 192), device=device, dtype=dtype, requires_grad=True)
    k = torch.randn((B, L, H, 192), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((B, L, H, 128), device=device, dtype=dtype, requires_grad=True)

    q_q, q_s = head_wise_quant(q)
    k_q, k_s = head_wise_quant(k)

    output_ref, lse_ref, max_logits_ref = torch_attn(q, k, v, causal=causal, hp=True)

    output, lse, max_logits = triton_mp_mla_forward(q_q, k_q, v, q_s, k_s, causal=causal)
    output_check(output_ref, output, atol=0.2, rtol=0.5, name='mp.output')
    output_check(lse_ref.float(), lse, atol=0.2, rtol=0.5, name='mp.lse')

    if bench:
        ref_flops = B * L * L * H * (192 + 128) * (1 if causal else 2)
        benchmark_func(triton_mp_mla_forward, q_q, k_q, v, q_s, k_s, causal=causal, ref_flops=ref_flops)


if __name__ == "__main__":
    test_softmax(M=128, N=128)
    test_dot_sum(M=128, N=128, D=128)
    test_mla(B=1, L=8192, H=64, causal=True, hpc=True, safe=True, bench=True)
    test_mla(B=1, L=8192, H=64, causal=True, hpc=False, safe=False, coef=1.0, bench=True)
    test_mla(B=1, L=8192, H=64, causal=True, hpc=False, safe=True, coef=100.0, bench=True)
    test_mla(B=1, L=4096, H=64, causal=True, hpc=False, safe=False, bench=True)
    test_mla(B=1, L=4096, H=64, causal=False, hpc=False, safe=False, bench=True)
    test_mla(B=1, L=8192, H=64, causal=False, hpc=False, safe=False, bench=True)
    test_mp_mla(B=1, L=8192, H=64, causal=True, hpc=False, bench=True)

