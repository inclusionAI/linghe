# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.infer.rope import triton_varlen_qk_norm_and_half_rope


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
        cos = cos[:, 0, 0][position_ids][:, :, None]
        sin = sin[:, 0, 0][position_ids][:, :, None]
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


def torch_half_rope(q, k, freqs, position_ids):
    dtype = q.dtype
    B, L, H, D = q.shape
    d = D // 2
    cos = freqs.cos()
    sin = freqs.sin()
    qr, kr = apply_rotary_pos_emb(q[:, :, :, :d], k[:, :, :, :d], cos, sin,
                                  position_ids)
    qo = torch.cat([qr, q[:, :, :, d:]], dim=-1)
    ko = torch.cat([kr, k[:, :, :, d:]], dim=-1)
    return qo.to(dtype), ko.to(dtype)


def torch_qk_norm(q, k, qw, kw, eps=1e-6):
    dtype = q.dtype
    B, L, H, D = q.shape
    rms = torch.sqrt(q.float().square().mean(-1) + eps)
    q = q / rms[:, :, :, None]
    q = q * qw
    rms = torch.sqrt(k.float().square().mean(-1) + eps)
    k = k / rms[:, :, :, None]
    k = k * kw
    return q.to(dtype), k.to(dtype)



def torch_qk_norm_and_half_rope(qkv, qw, kw, freqs,
                                position_ids,
                                H=32,
                                h=4,
                                eps=1e-6,
                                interleaved=False,
                                linear_scale=False,
                                scaling=1.0,
                                silu=False):

    bs, length, dim = qkv.shape
    dtype = qkv.dtype
    qkv = qkv.float()
    qw = qw.float()
    kw = kw.float()

    if silu:
        qkv = torch.nn.functional.silu(qkv)

    D = dim // (H + 2 * h)

    if interleaved:
        qkv = qkv.view(bs, length, h, (2 + H // h) * D)
        q, k, v = torch.split(qkv, [H // h * D, D, D], 3)
        q = torch.reshape(q, (bs, length, H, D))
    else:
        qkv = qkv.view(bs, length, H + 2 * h, D)
        q, k, v = torch.split(qkv, [H, h, h], dim=2)
    q, k = torch_qk_norm(q, k, qw, kw, eps=eps)
    q, k = torch_half_rope(q, k, freqs, position_ids)
    if linear_scale:
        q = q * scaling
    return q.to(dtype), k.to(dtype), v.to(dtype)


def torch_varlen_qk_norm_and_half_rope(qkvs, qw, kw,
                                       freqs, 
                                       position_ids,
                                       lengths,
                                       H=32,
                                       h=4,
                                       eps=1e-6,
                                       interleaved=False,
                                       silu=False, 
                                       linear_scale=False,
                                       scaling=1.0):

    B = len(lengths)

    qkvs = qkvs.split(lengths, 0)
    pids = position_ids.split(lengths, 0)

    qoss = []
    koss = []
    voss = []
    for i in range(B):
        qkv = qkvs[i][None]
        pid = pids[i][None]
        query, key, value = torch_qk_norm_and_half_rope(qkv, qw, kw, freqs,
                                                        pid,
                                                        H=H,
                                                        h=h, eps=eps,
                                                        interleaved=interleaved,
                                                        silu=silu,
                                                        linear_scale=linear_scale,
                                                        scaling=scaling)
        qoss.append(query)
        koss.append(key)
        voss.append(value)

    qoss = torch.cat(qoss, 1)[0]
    koss = torch.cat(koss, 1)[0]
    voss = torch.cat(voss, 1)[0]

    return qoss, koss, voss


def test_varlen_qk_norm_and_half_rope(lengths=[2048, 2048], H=32, h=4, dim=128,
                                      rope_theta=10000.0, silu=False,
                                      interleaved=False, eps=1e-6,
                                      linear_scale=False, scaling=1.0,
                                      bench=False):
    # weight grad of torch impl has great error with large seq nums
    dtype = torch.bfloat16 if len(lengths) < 8 else torch.float32
    device = 'cuda:0'
    N = sum(lengths)
    qkv = torch.randn(N, (H + 2 * h) * dim, dtype=dtype, device=device)
    
    position_ids = torch.cat([torch.arange(l, dtype=torch.int64, device=device) for l in lengths], 0)

    freq = rope_freqs(max(lengths), dim // 2, rope_theta=rope_theta)
    freqs = torch.cat([freq, freq], -1)
    cache = torch.cat([freq.cos(), freq.sin()], -1)

    qw = torch.randn(dim, dtype=dtype, device=device)
    kw = torch.randn(dim, dtype=dtype, device=device)

    qo_ref, ko_ref, vo_ref = torch_varlen_qk_norm_and_half_rope(qkv, qw, kw,
                                                                freqs,
                                                                position_ids,
                                                                lengths,
                                                                H=H, h=h,
                                                                eps=eps,
                                                                interleaved=interleaved,
                                                                silu=silu,
                                                                linear_scale=False,
                                                                scaling=1.0)

    qo, ko, vo = triton_varlen_qk_norm_and_half_rope(qkv, qw, kw, cache, position_ids, 
                                                     H=H,
                                                     h=h,
                                                     eps=eps,
                                                     interleaved=interleaved,
                                                     silu=silu,
                                                     linear_scale=False,
                                                     scaling=1.0)
    output_check(qo_ref, qo, name='q', atol=-1)
    output_check(ko_ref, ko, name='k', atol=-1)
    output_check(vo_ref, vo, name='v', atol=-1)

    if bench:
        lbh = sum(lengths) * H
        benchmark_func(triton_varlen_qk_norm_and_half_rope, qkv, qw, kw,
                       cache, position_ids,
                       interleaved=interleaved,
                       H=H, h=h,
                       silu=silu,
                       linear_scale=False,
                       scaling=1.0,
                       ref_bytes=lbh * (
                                   64 * 2 + 256 * 2 + 64 * 2 + 192 * 2 + 128 * 2),
                       n_profile=0)
        


if __name__ == '__main__':
    test_varlen_qk_norm_and_half_rope(lengths=[2048], H=32, h=8, dim=128,
                                      rope_theta=10000.0, silu=False,
                                      interleaved=False,
                                      bench=False)
    test_varlen_qk_norm_and_half_rope(lengths=[1024, 4096, 4096, 568], H=24,
                                      h=6, dim=128, rope_theta=10000.0,
                                      silu=False,
                                      interleaved=False,
                                      bench=False)
    test_varlen_qk_norm_and_half_rope(lengths=[16] * 512, H=32, h=4,
                                      dim=128, rope_theta=10000.0, silu=True,
                                      interleaved=True,
                                      bench=False,
                                      linear_scale=True, scaling=2.0)
    test_varlen_qk_norm_and_half_rope(lengths=[16] * 512, H=24, h=6,
                                      dim=128, rope_theta=10000.0, silu=True,
                                      interleaved=False,
                                      bench=False,
                                      linear_scale=True, scaling=2.0)
    test_varlen_qk_norm_and_half_rope(lengths=[1] * 1, H=32, h=4,
                                      dim=128, rope_theta=10000.0, silu=True,
                                      interleaved=False,
                                      bench=False,
                                      linear_scale=True, scaling=2.0)
    test_varlen_qk_norm_and_half_rope(lengths=[1] * 1, H=32, h=4,
                                      dim=128, rope_theta=10000.0, silu=True,
                                      interleaved=True,
                                      bench=False,
                                      linear_scale=True, scaling=2.0)
    test_varlen_qk_norm_and_half_rope(lengths=[1] * 1, H=24, h=6,
                                      dim=128, rope_theta=10000.0, silu=True,
                                      interleaved=True,
                                      bench=False,
                                      linear_scale=True, scaling=2.0)