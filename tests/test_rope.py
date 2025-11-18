# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.tools.benchmark import benchmark_func
from linghe.tools.util import output_check
from linghe.utils.rope import triton_half_rope_forward, \
    triton_half_rope_backward, triton_mla_rope_forward, triton_qk_norm_and_half_rope_forward, \
    triton_qk_norm_and_half_rope_backward, \
    triton_mla_rope_forward, triton_mla_rope_backward, \
    triton_varlen_qk_norm_and_half_rope_forward, triton_varlen_qk_norm_and_half_rope_backward


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    if cos.ndim == 2:
        cos = cos[position_ids][:, :, None]
        sin = sin[position_ids][:, :, None]
    elif cos.ndim == 4:
        cos = cos[:,0,0][position_ids][:, :, None]
        sin = sin[:,0,0][position_ids][:, :, None]
    else:
        raise ValueError('unsupported ndim=3')
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rope_freqs(length, dim, rope_theta=10000.0):
    inv_freq = 1.0 / (rope_theta ** (
                torch.arange(0, dim, 2, device='cuda:0').float() / dim))
    t = torch.arange(length, device='cuda:0', dtype=torch.int64).float()
    freqs = torch.outer(t, inv_freq)
    return freqs


def torch_half_rope(q, k, freqs, transposed=True):
    if transposed:
        L, B, H, D = q.shape
    else:
        B, L, H, D = q.shape
    d = D // 2
    cos = freqs.cos().to(q.dtype)
    sin = freqs.sin().to(q.dtype)
    if transposed:
        position_ids = torch.arange(L, device='cuda:0')[:, None].expand(-1, B)
    else:
        position_ids = torch.arange(L, device='cuda:0')[None, :].expand(B, -1)
    qr, kr = apply_rotary_pos_emb(q[:, :, :, :d], k[:, :, :, :d], cos, sin,
                                  position_ids)
    qo = torch.cat([qr, q[:, :, :, d:]], dim=-1)
    ko = torch.cat([kr, k[:, :, :, d:]], dim=-1)
    return qo, ko


def torch_qk_norm(q, k, qw, kw, eps=1e-6, transposed=True):
    dtype = q.dtype
    if transposed:
        L, B, H, D = q.shape
    else:
        B, L, H, D = q.shape
    rms = torch.sqrt(q.float().square().mean(-1) + eps)
    q = q / rms[:, :, :, None]
    q = q * qw
    rms = torch.sqrt(k.float().square().mean(-1) + eps)
    k = k / rms[:, :, :, None]
    k = k * kw
    return q.to(dtype), k.to(dtype)

def torch_mla_rope(q, kv, k_pos_emb, freqs, mscale=1.0):

    L, B, H, _ = q.shape
    q_no_pe, q_pos_emb = torch.split(
        q, [128, 64], dim=-1
    )

    k_no_pe, value = torch.split(
        kv, [128, 128], dim=-1
    )

    cos = freqs.cos().to(q.dtype) * mscale
    sin = freqs.sin().to(q.dtype) * mscale
    position_ids = torch.arange(L, device='cuda:0')[:, None].expand(-1, B)

    q_pos_emb = torch.cat([q_pos_emb[:,:,:,0::2], q_pos_emb[:,:,:,1::2]], -1)
    k_pos_emb = torch.cat([k_pos_emb[:,:,:,0::2], k_pos_emb[:,:,:,1::2]], -1)

    q_pos_emb, k_pos_emb = apply_rotary_pos_emb(q_pos_emb, k_pos_emb, cos, sin,
                                  position_ids)

    query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

    k_pos_emb = k_pos_emb.expand(-1, -1, H, -1)

    key = torch.cat([k_no_pe, k_pos_emb], dim=-1) 

    return query, key, value.contiguous()


def torch_varlen_mla_rope(qs, kvs, k_pos_embs, freqs, lengths, mscale=1.0, cp_size=1, cp_rank=0):

    ls = [x//cp_size for x in lengths]
    
    B = len(lengths)
    N, H, _ = qs.shape
    dtype = qs.dtype

    qss = qs.split(ls, 0)
    kvss = kvs.split(ls, 0)
    k_pos_embss = k_pos_embs.split(ls, 0)

    seg_size = 2 * cp_size
    qoss = [] 
    koss = []
    voss = []
    for i in range(B):
        q01 = qss[i][:,None].split([ls[i]//2]*2, 0)
        kv01 = kvss[i][:,None].split([ls[i]//2]*2, 0)
        pos01 = k_pos_embss[i][:,None].split([ls[i]//2]*2, 0)

        for j in range(2):
            q_no_pe, q_pos_emb = torch.split(
                q01[j], [128, 64], dim=-1
            )

            k_no_pe, value = torch.split(
                kv01[j], [128, 128], dim=-1
            )

            cos = freqs.cos().to(dtype) * mscale
            sin = freqs.sin().to(dtype) * mscale
            if j == 0:
                p = cp_rank * lengths[i] // seg_size
            else:
                p = (cp_size * 2 - cp_rank - 1) * lengths[i] // seg_size
            position_ids = p + torch.arange(lengths[i]//seg_size, device='cuda:0')[:, None]

            q_pos_emb = torch.cat([q_pos_emb[:,:,:,0::2], q_pos_emb[:,:,:,1::2]], -1)
            k_pos_emb = torch.cat([pos01[j][:,:,:,0::2], pos01[j][:,:,:,1::2]], -1)

            q_pos_emb, k_pos_emb = apply_rotary_pos_emb(q_pos_emb, k_pos_emb, cos, sin,
                                        position_ids)

            query = torch.cat([q_no_pe, q_pos_emb], dim=-1)

            k_pos_emb = k_pos_emb.expand(-1, -1, H, -1)

            key = torch.cat([k_no_pe, k_pos_emb], dim=-1) 

            qoss.append(query)
            koss.append(key)
            voss.append(value)

    qoss = torch.cat(qoss, 0)[:,0]
    koss = torch.cat(koss, 0)[:,0]
    voss = torch.cat(voss, 0)[:,0]

    return qoss, koss, voss


def torch_qk_norm_and_half_rope(qkv, qw, kw, freqs, H=32,
                                h=4, eps=1e-6, interleaved=True, transposed=True, silu=False):
    if transposed:
        length, bs, dim = qkv.shape
    else:
        bs, length, dim = qkv.shape
    qkv = qkv.float()
    qw = qw.float()
    kw = kw.float()

    if silu:
        qkv = torch.nn.functional.silu(qkv)

    D = dim // (H + 2 * h)
    if interleaved:
        if transposed:
            qkv = qkv.view(length, bs, h, (2 + H // h) * D)
            q, k, v = torch.split(qkv, [H // h * D, D, D], 3)
            q = torch.reshape(q, (length, bs, H, D))
        else:
            qkv = qkv.view(bs, length, h, (2 + H // h) * D)
            q, k, v = torch.split(qkv, [H // h * D, D, D], 3)
            q = torch.reshape(q, (bs, length, H, D))
    else:
        if transposed:
            qkv = qkv.view(length, bs, H + 2 * h, D)
            q, k, v = torch.split(qkv, [H, h, h], dim=2)
        else:
            qkv = qkv.view(bs, length, H + 2 * h, D)
            q, k, v = torch.split(qkv, [H, h, h], dim=2)
    q, k = torch_qk_norm(q, k, qw, kw, eps=eps, transposed=transposed)
    q, k = torch_half_rope(q, k, freqs, transposed=transposed)
    if transposed:
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
    return q, k, v



def torch_varlen_qk_norm_and_half_rope(qkvs, qw, kw, freqs, lengths, H=32, h=4, interleaved=True, silu=False, eps=1e-6, mscale=1.0, cp_size=1, cp_rank=0):
    ls = [x//cp_size for x in lengths]
    D = qkvs.size(1)

    B = len(lengths)

    qkvss = qkvs.split(ls, 0)

    seg_size = 2 * cp_size
    qoss = [] 
    koss = []
    voss = []
    for i in range(B):
        qkv01 = qkvss[i].split([ls[i]//2]*2, 0)

        for j in range(2):
            qkv = qkv01[j].view(1, ls[i]//2, D)

            if j == 0:
                p = cp_rank * lengths[i] // seg_size
            else:
                p = (cp_size * 2 - cp_rank - 1) * lengths[i] // seg_size
            position_ids = p + torch.arange(lengths[i]//seg_size, device='cuda:0')
            fs = freqs[position_ids]
            query, key, value = torch_qk_norm_and_half_rope(qkv, qw, kw, fs, H=H,
                                h=h, eps=eps, interleaved=interleaved, transposed=False, silu=silu)
            qoss.append(query)
            koss.append(key)
            voss.append(value)

    qoss = torch.cat(qoss, 1)[0]
    koss = torch.cat(koss, 1)[0]
    voss = torch.cat(voss, 1)[0]

    return qoss, koss, voss


def test_half_rope(B=2, L=4096, H=32, h=8, D=128, rope_theta=10000.0,
                    transposed=True,
                   bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'
    q = torch.randn(L, B, H, D, dtype=dtype, device=device)
    k = torch.randn(L, B, h, D, dtype=dtype, device=device)
    freqs = rope_freqs(L, D // 2, rope_theta=rope_theta)
    freqs = torch.cat([freqs, freqs], -1)

    q_ref, k_ref = torch_half_rope(q, k, freqs, transposed=transposed)
    qo, ko = triton_half_rope_forward(q, k, freqs, transposed=transposed)
    output_check(q_ref, qo, mode='q')
    output_check(k_ref, ko, mode='k')

    q_grad = torch.randn(L, B, H, D, dtype=dtype, device=device)
    k_grad = torch.randn(L, B, h, D, dtype=dtype, device=device)
    q_ref = q.detach().clone().requires_grad_()
    k_ref = k.detach().clone().requires_grad_()
    qo_ref, ko_ref = torch_half_rope(q_ref, k_ref, freqs, transposed=transposed)
    qo_ref.backward(gradient=q_grad)
    ko_ref.backward(gradient=k_grad)
    dq_ref = q_ref.grad
    dk_ref = k_ref.grad

    dq, dk = triton_half_rope_backward(q_grad, k_grad, freqs, inplace=True, transposed=transposed)
    output_check(dq_ref, dq, mode='dq')
    output_check(dk_ref, dk, mode='dk')

    if bench:
        benchmark_func(triton_half_rope_forward, q, k, freqs,
                       ref_bytes=L * B * (H + h) * D * 4,
                       n_profile=0)


def test_qk_norm_and_half_rope(B=2, L=4096, H=32, h=8, D=128,
                               rope_theta=10000.0, 
                               interleaved=True,
                               transposed=True,
                               silu=False,
                               bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'
    if transposed:
        qkv = torch.randn(L, B, (H + 2 * h) * D, dtype=dtype, device=device)
    else:
        qkv = torch.randn(B, L, (H + 2 * h) * D, dtype=dtype, device=device)
    qw = torch.randn(D, dtype=dtype, device=device)
    kw = torch.randn(D, dtype=dtype, device=device)
    freqs = rope_freqs(L, D // 2, rope_theta=rope_theta)
    freqs = torch.cat([freqs, freqs], -1)
    q_ref, k_ref, v_ref = torch_qk_norm_and_half_rope(qkv, qw, kw, freqs,
                                                      H=H, h=h,
                                                      eps=1e-6,
                                                      transposed=transposed,
                                                      interleaved=interleaved,
                                                      silu=silu)
    qo, ko, vo = triton_qk_norm_and_half_rope_forward(qkv, qw, kw, freqs, H=H,
                                                      h=h, eps=1e-6,
                                                      transposed=transposed,
                                                      interleaved=interleaved,
                                                      silu=silu)
    output_check(q_ref, qo, mode='q')
    output_check(k_ref, ko, mode='k')
    output_check(v_ref, vo, mode='v')

    q_grad = torch.randn(B, L, H, D, dtype=dtype, device=device)
    k_grad = torch.randn(B, L, h, D, dtype=dtype, device=device)
    v_grad = torch.randn(B, L, h, D, dtype=dtype, device=device)
    qkv_ref = qkv.detach().clone().requires_grad_()
    qw_ref = qw.detach().clone().requires_grad_()
    kw_ref = kw.detach().clone().requires_grad_()
    qo_ref, ko_ref, vo_ref = torch_qk_norm_and_half_rope(qkv_ref, qw_ref,
                                                         kw_ref, freqs,
                                                         H=H, h=h, eps=1e-6,
                                                         transposed=transposed,
                                                         interleaved=interleaved,
                                                         silu=silu)
    qo_ref.backward(gradient=q_grad, retain_graph=True)
    ko_ref.backward(gradient=k_grad, retain_graph=True)
    vo_ref.backward(gradient=v_grad, retain_graph=True)

    dqkv_ref = qkv_ref.grad
    dqw_ref = qw_ref.grad
    dkw_ref = kw_ref.grad

    dqkv, dqw, dkw = triton_qk_norm_and_half_rope_backward(q_grad, k_grad,
                                                           v_grad, qkv, qw, kw,
                                                           freqs, eps=1e-6,
                                                           transposed=transposed,
                                                           interleaved=interleaved,
                                                           silu=silu)
    output_check(dqkv_ref, dqkv, mode='dqkv')
    output_check(dqw_ref, dqw, mode='dqw')
    output_check(dkw_ref, dkw, mode='dkw')

    if bench:
        benchmark_func(triton_qk_norm_and_half_rope_forward, qkv, qw, kw, freqs,
                       H=H, h=h, eps=1e-6, transposed=transposed, interleaved=interleaved,
                       ref_bytes=L * B * (H + 2 * h) * D * 4,
                       n_profile=0)
        benchmark_func(triton_qk_norm_and_half_rope_backward, q_grad, k_grad,
                       v_grad, qkv, qw, kw, freqs, eps=1e-6, transpose=transposed, interleaved=interleaved,
                       ref_bytes=L * B * (H + 2 * h) * D * 6,
                       n_profile=0)



def test_varlen_qk_norm_and_half_rope(lengths=[2048,2048], H=32, h=4, dim=128, rope_theta=10000.0, silu=False,
                    interleaved=True,
                   bench=False, cp_size=1, cp_rank=0):
    dtype = torch.bfloat16
    device = 'cuda:0'
    S = cp_size * 2
    N = sum(lengths) // cp_size
    qkv = torch.randn(N, (H + 2 * h) * dim, dtype=dtype, device=device)
    cu_seqlens_q = torch.cumsum(torch.tensor([0]+lengths, device=device, dtype=torch.int32), 0).to(torch.int32)
    cu_seqlens_kv = cu_seqlens_q

    freqs = rope_freqs(max(lengths), dim//2, rope_theta=rope_theta)
    freqs = torch.cat([freqs, freqs], -1)

    mscale = 1.0
    qw = torch.randn(dim, dtype=dtype, device=device).requires_grad_()
    kw = torch.randn(dim, dtype=dtype, device=device).requires_grad_()

    q_grad = torch.randn(sum(lengths)//cp_size, H, dim, dtype=dtype, device=device)
    k_grad = torch.randn(sum(lengths)//cp_size, h, dim, dtype=dtype, device=device)
    v_grad = torch.randn(sum(lengths)//cp_size, h, dim, dtype=dtype, device=device)
    qkv = qkv.detach().clone().requires_grad_()
    qo_ref, ko_ref, vo_ref = torch_varlen_qk_norm_and_half_rope(qkv, qw, kw, freqs, lengths, H=H, h=h, interleaved=interleaved, silu=silu, mscale=mscale, cp_size=cp_size, cp_rank=cp_rank)
    
    qo_ref.backward(gradient=q_grad, retain_graph=True)
    ko_ref.backward(gradient=k_grad, retain_graph=True)
    vo_ref.backward(gradient=v_grad, retain_graph=True)

    dqkv_ref = qkv.grad
    dqw_ref = qw.grad
    dkw_ref = kw.grad
    
    qo, ko, vo = triton_varlen_qk_norm_and_half_rope_forward(qkv, qw, kw, freqs, cu_seqlens_q, cu_seqlens_kv, interleaved=interleaved, silu=silu, mscale=mscale, cp_size=cp_size, cp_rank=cp_rank)
    output_check(qo_ref, qo, mode='q')
    output_check(ko_ref, ko, mode='k')
    output_check(vo_ref, vo, mode='v')


    dqkv, dqw, dkw = triton_varlen_qk_norm_and_half_rope_backward(q_grad, k_grad, v_grad, qkv, qw, kw, freqs, cu_seqlens_q, cu_seqlens_kv, mscale=mscale, interleaved=interleaved, silu=silu, cp_size=cp_size, cp_rank=cp_rank)
    output_check(dqkv_ref, dqkv, mode='dqkv')
    output_check(dqw_ref, dqw, mode='dqw')
    output_check(dkw_ref, dkw, mode='dkv')


    if bench:
        lbh = sum(lengths)//cp_size*H
        benchmark_func(triton_varlen_qk_norm_and_half_rope_forward, qkv, qw, kw, freqs, cu_seqlens_q, cu_seqlens_kv, interleaved=interleaved, silu=silu, mscale=mscale, cp_size=cp_size, cp_rank=cp_rank,
                       ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2),
                       n_profile=0)
        benchmark_func(triton_varlen_qk_norm_and_half_rope_backward, q_grad, k_grad, v_grad, qkv, qw, kw, freqs, cu_seqlens_q, cu_seqlens_kv, mscale=mscale, interleaved=interleaved, silu=silu, cp_size=cp_size, cp_rank=cp_rank,
                       ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2),
                       n_profile=0)



def test_mla_rope(B=2, L=4096, H=32, rope_theta=10000.0,
                   bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'
    q = torch.randn(L, B, H, 192, dtype=dtype, device=device)
    kv = torch.randn(L, B, H, 256, dtype=dtype, device=device)
    k_pos_emb = torch.randn(L, B, 1, 64+512, dtype=dtype, device=device)[:,:,:,:64]
    freqs = rope_freqs(L, 64, rope_theta=rope_theta)
    freqs = torch.cat([freqs, freqs], -1)
    freqs = freqs[:,None,None]

    mscale = 1.0
    q_ref, k_ref, v_ref = torch_mla_rope(q.clone().detach(), kv, k_pos_emb, freqs, mscale=mscale)
    qo, ko, vo = triton_mla_rope_forward(q.clone().detach(), kv, k_pos_emb, freqs, mscale=mscale)
    output_check(q_ref, qo, mode='q')
    output_check(k_ref, ko, mode='k')
    output_check(v_ref, vo, mode='v')


    q_grad = torch.randn(L, B, H, 192, dtype=dtype, device=device)
    k_grad = torch.randn(L, B, H, 192, dtype=dtype, device=device)
    v_grad = torch.randn(L, B, H, 128, dtype=dtype, device=device)
    q_ref = q.detach().clone().requires_grad_()
    kv_ref = kv.detach().clone().requires_grad_()
    k_pos_emb_ref = k_pos_emb.detach().clone().requires_grad_()
    qo_ref, ko_ref, vo_ref = torch_mla_rope(q_ref, kv_ref, k_pos_emb_ref, freqs, mscale=mscale)
    qo_ref.backward(gradient=q_grad, retain_graph=True)
    ko_ref.backward(gradient=k_grad, retain_graph=True)
    vo_ref.backward(gradient=v_grad, retain_graph=True)
    dq_ref = q_ref.grad
    dkv_ref = kv_ref.grad
    dp_ref = k_pos_emb_ref.grad
    dq, dkv, dp = triton_mla_rope_backward(q_grad, k_grad, v_grad, freqs, mscale=mscale)
    output_check(dq_ref, dq, mode='dq')
    output_check(dkv_ref, dkv, mode='dkv')
    output_check(dp_ref, dp, mode='dp')



    if False:
        from megatron.core.fusions.fused_mla_yarn_rope_apply import (
            fused_apply_mla_rope_for_kv,
            fused_apply_mla_rope_for_q,
        )
        rotary_pos_cos = freqs.cos()
        rotary_pos_sin = freqs.sin()
        q_ref = q.detach().clone().requires_grad_()
        kv_ref = kv.detach().clone().requires_grad_()
        k_pos_emb_ref = k_pos_emb.detach().clone().requires_grad_()
        query_ref = fused_apply_mla_rope_for_q(
            q_ref,
            rotary_pos_cos,
            rotary_pos_sin,
            128,
            64,
            cu_seqlens_q=None,
            cp_rank=0,
            cp_size=1,
        )
        key_ref, value_ref = fused_apply_mla_rope_for_kv(
            kv_ref,
            k_pos_emb_ref,
            rotary_pos_cos,
            rotary_pos_sin,
            64,
            128,
            128,
            cu_seqlens_kv=None,
            cp_rank=0,
            cp_size=1
        )
        output_check(query_ref, qo, mode='q')
        output_check(key_ref, ko, mode='k')
        output_check(value_ref, vo, mode='v')

        query_ref.backward(gradient=q_grad, retain_graph=True)
        key_ref.backward(gradient=k_grad, retain_graph=True)
        value_ref.backward(gradient=v_grad, retain_graph=True)
        dq_ref = q_ref.grad
        dkv_ref = kv_ref.grad
        dp_ref = k_pos_emb_ref.grad
        output_check(dq_ref, dq, mode='dq')
        output_check(dkv_ref, dkv, mode='dkv')
        output_check(dp_ref, dp, mode='dp')


    if bench:
        lbh = L*B*H
        benchmark_func(triton_mla_rope_forward, q, kv, k_pos_emb, freqs,
                       ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2),
                       n_profile=0)
        benchmark_func(triton_mla_rope_backward, q_grad, k_grad, v_grad, freqs,
                       ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2),
                       n_profile=0)


def test_varlen_mla_rope(lengths=[2048,2048], H=32, rope_theta=10000.0,
                   bench=False, cp_size=1, cp_rank=0):
    dtype = torch.bfloat16
    device = 'cuda:0'
    S = cp_size * 2
    qcs = []
    kvcs = []
    k_pos_embcs = []
    for i, N in enumerate(lengths):
        q = torch.randn(N, H, 192, dtype=dtype, device=device)
        kv = torch.randn(N, H, 256, dtype=dtype, device=device)
        k_pos_emb = torch.randn(N, 1, 64, dtype=dtype, device=device)
        qc = q.split([N//S]*S, 0)
        qcs.append(qc[cp_rank])
        qcs.append(qc[cp_size * 2 - cp_rank - 1])
        kvc = kv.split([N//S]*S, 0)
        kvcs.append(kvc[cp_rank])
        kvcs.append(kvc[cp_size * 2 - cp_rank - 1])
        k_pos_embc = k_pos_emb.split([N//S]*S, 0)
        k_pos_embcs.append(k_pos_embc[cp_rank])
        k_pos_embcs.append(k_pos_embc[cp_size * 2 - cp_rank - 1])
    qc = torch.cat(qcs, 0)
    kvc = torch.cat(kvcs, 0)
    k_pos_embc = torch.cat(k_pos_embcs, 0)
    k_pos_embc = torch.cat([k_pos_embc, k_pos_embc], -1)[:,:,:64]

    cu_seqlens_q = torch.cumsum(torch.tensor([0]+lengths, device=device, dtype=torch.int32), 0).to(torch.int32)
    cu_seqlens_kv = cu_seqlens_q

    freqs = rope_freqs(max(lengths), 64, rope_theta=rope_theta)
    freqs = torch.cat([freqs, freqs], -1)

    mscale = 1.0
    q_ref, k_ref, v_ref = torch_varlen_mla_rope(qc, kvc, k_pos_embc, freqs, lengths, mscale=mscale, cp_size=cp_size, cp_rank=cp_rank)
    qo, ko, vo = triton_mla_rope_forward(qc.detach().clone(), kvc, k_pos_embc, freqs, mscale=mscale, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, cp_size=cp_size, cp_rank=cp_rank)
    output_check(q_ref, qo, mode='q')
    output_check(k_ref, ko, mode='k')
    output_check(v_ref, vo, mode='v')


    q_grad = torch.randn(sum(lengths)//cp_size, H, 192, dtype=dtype, device=device)
    k_grad = torch.randn(sum(lengths)//cp_size, H, 192, dtype=dtype, device=device)
    v_grad = torch.randn(sum(lengths)//cp_size, H, 128, dtype=dtype, device=device)
    q_i = qc.detach().clone().requires_grad_()
    kv_i = kvc.detach().clone().requires_grad_()
    k_pos_emb_i = k_pos_embc.detach().clone().requires_grad_()
    qo_ref, ko_ref, vo_ref = torch_varlen_mla_rope(q_i, kv_i, k_pos_emb_i, freqs, lengths, mscale=mscale, cp_size=cp_size, cp_rank=cp_rank)
    qo_ref.backward(gradient=q_grad, retain_graph=True)
    ko_ref.backward(gradient=k_grad, retain_graph=True)
    vo_ref.backward(gradient=v_grad, retain_graph=True)
    dq_ref = q_i.grad
    dkv_ref = kv_i.grad
    dp_ref = k_pos_emb_i.grad
    dq, dkv, dp = triton_mla_rope_backward(q_grad, k_grad, v_grad, freqs, mscale=mscale, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, cp_size=cp_size, cp_rank=cp_rank)
    output_check(dq_ref, dq, mode='dq')
    output_check(dkv_ref, dkv, mode='dkv')
    output_check(dp_ref, dp, mode='dp')


    if False:
        from megatron.core.fusions.fused_mla_yarn_rope_apply import (
            fused_apply_mla_rope_for_kv,
            fused_apply_mla_rope_for_q,
        )
        rotary_pos_cos = freqs.cos()
        rotary_pos_sin = freqs.sin()
        q_i = qc.detach().clone().requires_grad_()
        kv_i = kvc.detach().clone().requires_grad_()
        k_pos_emb_i = k_pos_embc.detach().clone().requires_grad_()
        query_ref = fused_apply_mla_rope_for_q(
            q_i,
            rotary_pos_cos,
            rotary_pos_sin,
            128,
            64,
            cu_seqlens_q,
            cp_rank,
            cp_size,
        )
        key_ref, value_ref = fused_apply_mla_rope_for_kv(
            kv_i,
            k_pos_emb_i,
            rotary_pos_cos,
            rotary_pos_sin,
            64,
            128,
            128,
            cu_seqlens_kv,
            cp_rank,
            cp_size,
        )
        output_check(query_ref, qo, mode='q')
        output_check(key_ref, ko, mode='k')
        output_check(value_ref, vo, mode='v')
        query_ref.backward(gradient=q_grad, retain_graph=True)
        key_ref.backward(gradient=k_grad, retain_graph=True)
        value_ref.backward(gradient=v_grad, retain_graph=True)
        dq_ref = q_i.grad
        dkv_ref = kv_i.grad
        dp_ref = k_pos_emb_i.grad
        output_check(dq_ref, dq, mode='dq')
        output_check(dkv_ref, dkv, mode='dkv')
        output_check(dp_ref, dp, mode='dp')


    if bench:
        lbh = sum(lengths)//cp_size*H
        benchmark_func(triton_mla_rope_forward, qc, kvc, k_pos_embc, freqs, mscale=mscale, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, cp_size=cp_size, cp_rank=cp_rank,
                       ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2),
                       n_profile=0)
        benchmark_func(triton_mla_rope_backward, q_grad, k_grad, v_grad, freqs, mscale=mscale, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, cp_size=cp_size, cp_rank=cp_rank,
                       ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2),
                       n_profile=0)



if __name__ == '__main__':
    test_half_rope(B=2, L=4096, H=32, h=8, D=128, rope_theta=10000.0, transposed=True,
                   bench=False)
    test_half_rope(B=2, L=4096, H=32, h=8, D=128, rope_theta=10000.0, transposed=False,
                   bench=False)
    test_qk_norm_and_half_rope(B=1, L=4096, H=16, h=16, D=128,
                               rope_theta=10000.0, interleaved=False, transposed=True, bench=False)
    test_qk_norm_and_half_rope(B=2, L=4096, H=16, h=4, D=128,
                               rope_theta=10000.0, interleaved=False, transposed=True, bench=False)
    test_qk_norm_and_half_rope(B=4, L=4096, H=16, h=4, D=128,
                               rope_theta=10000.0, interleaved=True, transposed=True, silu=True, bench=False)
    test_qk_norm_and_half_rope(B=4, L=4096, H=16, h=4, D=128,
                               rope_theta=10000.0, interleaved=True, transposed=True, silu=False, bench=False)
    test_qk_norm_and_half_rope(B=4, L=4096, H=32, h=8, D=128,
                               rope_theta=10000.0, interleaved=True, transposed=True, bench=False)
    test_qk_norm_and_half_rope(B=4, L=4096, H=32, h=8, D=128,
                               rope_theta=10000.0, interleaved=False, transposed=True, 
                               bench=False)
    test_mla_rope(B=4, L=4096, H=16, rope_theta=10000.0,
                   bench=False)
    test_varlen_mla_rope(lengths=[4096,4096], H=32, rope_theta=10000.0,
                   bench=False, cp_size=1, cp_rank=0)
    test_varlen_mla_rope(lengths=[4096*4,2048*4,2048*4], H=32, rope_theta=10000.0,
                   bench=False, cp_size=4, cp_rank=0)
    test_varlen_mla_rope(lengths=[4096*4,2048*4,2048*4], H=32, rope_theta=10000.0,
                   bench=False, cp_size=4, cp_rank=1)
    test_varlen_qk_norm_and_half_rope(lengths=[2048,2048], H=32, h=4, dim=128, rope_theta=10000.0, silu=False,
                    interleaved=True,
                   bench=False, cp_size=1, cp_rank=0)
    test_varlen_qk_norm_and_half_rope(lengths=[2048,4096,4096], H=32, h=4, dim=128, rope_theta=10000.0, silu=False,
                    interleaved=True,
                   bench=False, cp_size=1, cp_rank=0)
    test_varlen_qk_norm_and_half_rope(lengths=[2048,4096,4096], H=32, h=4, dim=128, rope_theta=10000.0, silu=True,
                    interleaved=True,
                   bench=False, cp_size=1, cp_rank=0)
    test_varlen_qk_norm_and_half_rope(lengths=[2048,4096,4096], H=32, h=4, dim=128, rope_theta=10000.0, silu=True,
                    interleaved=True,
                   bench=False, cp_size=4, cp_rank=0)
    test_varlen_qk_norm_and_half_rope(lengths=[2048,4096,4096], H=32, h=4, dim=128, rope_theta=10000.0, silu=True,
                    interleaved=False,
                   bench=False, cp_size=4, cp_rank=0)