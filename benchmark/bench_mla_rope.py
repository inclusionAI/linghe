import torch
from megatron.core.fusions.fused_mla_yarn_rope_apply import (
    fused_apply_mla_rope_for_kv,
    fused_apply_mla_rope_for_q,
)

from linghe.facade.rope import mla_rope
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check


def rope_freqs(length, dim, rope_theta=10000.0):
    inv_freq = 1.0 / (rope_theta ** (
            torch.arange(0, dim, 2, device='cuda:0').float() / dim))
    t = torch.arange(length, device='cuda:0', dtype=torch.int64).float()
    freqs = torch.outer(t, inv_freq)
    return freqs


def bench_mla_rope(B=2, L=4096, H=32, rope_theta=10000.0, transpose=True):
    dtype = torch.bfloat16
    device = 'cuda:0'
    q = torch.randn(L, B, H, 192, dtype=dtype, device=device).requires_grad_()
    kv = torch.randn(L, B, H, 256, dtype=dtype, device=device).requires_grad_()
    k_pos_emb = torch.randn(L, B, 64 + 512, dtype=dtype, device=device)[:, :,
                :64].view(L, B, 1, 64).requires_grad_()
    freqs = rope_freqs(L, 64, rope_theta=rope_theta)
    freqs = torch.cat([freqs, freqs], -1)
    freqs = freqs[:, None, None]
    if transpose:
        q_grad = torch.randn(B, L, H, 192, dtype=dtype, device=device)
        k_grad = torch.randn(B, L, H, 192, dtype=dtype, device=device)
        v_grad = torch.randn(B, L, H, 128, dtype=dtype, device=device)
    else:
        q_grad = torch.randn(L, B, H, 192, dtype=dtype, device=device)
        k_grad = torch.randn(L, B, H, 192, dtype=dtype, device=device)
        v_grad = torch.randn(L, B, H, 128, dtype=dtype, device=device)

    mscale = 1.0

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
    if transpose:
        query_ref = query_ref.transpose(0, 1)
        key_ref = key_ref.transpose(0, 1)
        value_ref = value_ref.transpose(0, 1)

    query_ref.backward(gradient=q_grad.detach().clone(), retain_graph=True)
    key_ref.backward(gradient=k_grad.detach().clone(), retain_graph=True)
    value_ref.backward(gradient=v_grad.detach().clone(), retain_graph=True)
    dq_ref = q_ref.grad
    dkv_ref = kv_ref.grad
    dp_ref = k_pos_emb_ref.grad

    qo, ko, vo = mla_rope(q,
                          kv,
                          k_pos_emb,
                          freqs,
                          cu_seqlens_q=None,
                          cu_seqlens_kv=None,
                          mscale=mscale,
                          cp_size=1,
                          cp_rank=0,
                          transpose=transpose)
    qo.backward(gradient=q_grad, retain_graph=True)
    ko.backward(gradient=k_grad, retain_graph=True)
    vo.backward(gradient=v_grad, retain_graph=True)
    dq = q.grad
    dkv = kv.grad
    dp = k_pos_emb.grad

    output_check(query_ref, qo, name='q')
    output_check(key_ref, ko, name='k')
    output_check(value_ref, vo, name='v')

    output_check(dq_ref, dq, name='dq')
    output_check(dkv_ref, dkv, name='dkv')
    output_check(dp_ref, dp, name='dp', atol=0.1, rtol=0.02)

    lbh = L * B * H
    benchmark_func(fused_apply_mla_rope_for_q, q, rotary_pos_cos,
                   rotary_pos_sin,
                   128, 64, cu_seqlens_q=None, cp_rank=0, cp_size=1,
                   ref_bytes=lbh * (
                               64 * 2 + 256 * 2 + 64 * 2 + 192 * 2 + 128 * 2),
                   n_profile=0)
    benchmark_func(fused_apply_mla_rope_for_kv, kv, k_pos_emb, rotary_pos_cos,
                   rotary_pos_sin,
                   64, 128, 128, cu_seqlens_kv=None, cp_rank=0, cp_size=1,
                   ref_bytes=lbh * (
                               64 * 2 + 256 * 2 + 64 * 2 + 192 * 2 + 128 * 2),
                   n_profile=0)
    benchmark_func(mla_rope, q, kv, k_pos_emb, freqs,
                   ref_bytes=lbh * (
                               64 * 2 + 256 * 2 + 64 * 2 + 192 * 2 + 128 * 2),
                   n_profile=0)


def bench_varlen_mla_rope(lengths=[2048, 2048], H=32, rope_theta=10000.0,
                          cp_size=1, cp_rank=0, stride=True):
    dtype = torch.bfloat16
    device = 'cuda:0'
    q = torch.randn(sum(lengths) // cp_size, H, 192, dtype=dtype,
                    device=device).requires_grad_()
    kv = torch.randn(sum(lengths) // cp_size, H, 256, dtype=dtype,
                     device=device).requires_grad_()
    if stride:
        k_pos_emb = torch.randn(sum(lengths) // cp_size, 576, dtype=dtype,
                                device=device)
        k_pos_emb = k_pos_emb[:, 512:].view(sum(lengths) // cp_size, 1,
                                            64).requires_grad_()
    else:
        k_pos_emb = torch.randn(sum(lengths) // cp_size, 1, 64, dtype=dtype,
                                device=device).requires_grad_()
    cu_seqlens_q = torch.cumsum(
        torch.tensor([0] + lengths, device=device, dtype=torch.int32), 0).to(
        torch.int32)
    cu_seqlens_kv = cu_seqlens_q

    freqs = rope_freqs((max(lengths) - 1) // 32 * 32 + 32, 64,
                       rope_theta=rope_theta)
    freqs = torch.cat([freqs, freqs], -1)[:, None, None]

    q_grad = torch.randn(sum(lengths) // cp_size, H, 192, dtype=dtype,
                         device=device)
    k_grad = torch.randn(sum(lengths) // cp_size, H, 192, dtype=dtype,
                         device=device)
    v_grad = torch.randn(sum(lengths) // cp_size, H, 128, dtype=dtype,
                         device=device)

    mscale = 1.0

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
        cu_seqlens_q,
        cp_rank,
        cp_size,
    )
    key_ref, value_ref = fused_apply_mla_rope_for_kv(
        kv_ref,
        k_pos_emb_ref,
        rotary_pos_cos,
        rotary_pos_sin,
        64,
        128,
        128,
        cu_seqlens_kv,
        cp_rank,
        cp_size,
    )

    query_ref.backward(gradient=q_grad.clone().detach(), retain_graph=True)
    key_ref.backward(gradient=k_grad.clone().detach(), retain_graph=True)
    value_ref.backward(gradient=v_grad.clone().detach(), retain_graph=True)
    dq_ref = q_ref.grad
    dkv_ref = kv_ref.grad
    dp_ref = k_pos_emb_ref.grad

    qo, ko, vo = mla_rope(q,
                          kv,
                          k_pos_emb,
                          freqs,
                          cu_seqlens_q=cu_seqlens_q,
                          cu_seqlens_kv=cu_seqlens_kv,
                          mscale=mscale,
                          cp_size=cp_size,
                          cp_rank=cp_rank,
                          transpose=False)

    qo.backward(gradient=q_grad.clone().detach(), retain_graph=True)
    ko.backward(gradient=k_grad, retain_graph=True)
    vo.backward(gradient=v_grad, retain_graph=True)
    dq = q.grad
    dkv = kv.grad
    dp = k_pos_emb.grad

    output_check(query_ref, qo, name='q')
    output_check(key_ref, ko, name='k')
    output_check(value_ref, vo, name='v')

    output_check(dq_ref, dq, name='dq')
    output_check(dkv_ref, dkv, name='dkv')
    output_check(dp_ref, dp, name='dp', atol=0.1, rtol=0.02)

    lbh = sum(lengths) // cp_size * H
    benchmark_func(fused_apply_mla_rope_for_q, q, rotary_pos_cos,
                   rotary_pos_sin, 128, 64,
                   cu_seqlens_q, cp_size=cp_size, cp_rank=cp_rank,
                   ref_bytes=lbh * (
                               64 * 2 + 256 * 2 + 64 * 2 + 192 * 2 + 128 * 2),
                   n_profile=0)
    benchmark_func(fused_apply_mla_rope_for_kv, kv, k_pos_emb, rotary_pos_cos,
                   rotary_pos_sin,
                   64, 128, 128, cu_seqlens_kv, cp_size=cp_size,
                   cp_rank=cp_rank,
                   ref_bytes=lbh * (
                               64 * 2 + 256 * 2 + 64 * 2 + 192 * 2 + 128 * 2),
                   n_profile=0)
    benchmark_func(mla_rope, q, kv, k_pos_emb, freqs, mscale=mscale,
                   cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv,
                   cp_size=cp_size, cp_rank=cp_rank,
                   ref_bytes=lbh * (
                               64 * 2 + 256 * 2 + 64 * 2 + 192 * 2 + 128 * 2),
                   n_profile=0)

    benchmark_func(query_ref.backward, q_grad, retain_graph=True,
                   ref_bytes=lbh * (
                               64 * 2 + 256 * 2 + 64 * 2 + 192 * 2 + 128 * 2),
                   n_profile=0)
    benchmark_func(key_ref.backward, k_grad, retain_graph=True,
                   ref_bytes=lbh * (
                               64 * 2 + 256 * 2 + 64 * 2 + 192 * 2 + 128 * 2),
                   n_profile=0)
    benchmark_func(qo.backward, q_grad, retain_graph=True,
                   ref_bytes=lbh * (
                               64 * 2 + 256 * 2 + 64 * 2 + 192 * 2 + 128 * 2),
                   n_profile=0)


if __name__ == '__main__':
    # bench_mla_rope(L=4096, B=2, H=32, transpose=True)
    # bench_mla_rope(L=4096, B=2, H=16, transpose=True)
    # bench_mla_rope(L=4096, B=2, H=64, transpose=True)
    # bench_mla_rope(L=4096, B=2, H=16, transpose=True)
    # bench_mla_rope(L=4096, B=1, H=16, transpose=True)
    # bench_mla_rope(L=4096, B=1, H=16, transpose=False)
    bench_varlen_mla_rope(lengths=[444, 503, 434, 433, 472, 483, 557, 770],
                          H=16, rope_theta=10000.0,
                          cp_size=1, cp_rank=0, stride=True)
    bench_varlen_mla_rope(lengths=[444, 503, 434, 433, 472, 483, 557, 770],
                          H=32, rope_theta=10000.0,
                          cp_size=2, cp_rank=0, stride=False)
    bench_varlen_mla_rope(lengths=[444, 503, 434, 433, 472, 483, 557, 770],
                          H=32, rope_theta=10000.0,
                          cp_size=2, cp_rank=1, stride=False)
    bench_varlen_mla_rope(lengths=[444, 503, 434, 433, 472, 483, 557, 770],
                          H=32, rope_theta=10000.0,
                          cp_size=4, cp_rank=3, stride=False)
