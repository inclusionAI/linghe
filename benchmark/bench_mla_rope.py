
import random
import torch
from linghe.tools.benchmark import benchmark_func
from linghe.tools.util import output_check
from linghe.utils.rope import (triton_mla_rope_forward, 
                               triton_mla_rope_backward)


from megatron.core.fusions.fused_mla_yarn_rope_apply import (
    fused_apply_mla_rope_for_kv,
    fused_apply_mla_rope_for_q,
)

def rope_freqs(length, dim, rope_theta=10000.0):
    inv_freq = 1.0 / (rope_theta ** (
                torch.arange(0, dim, 2, device='cuda:0').float() / dim))
    t = torch.arange(length, device='cuda:0', dtype=torch.int64).float()
    freqs = torch.outer(t, inv_freq)
    return freqs



def bench_mla_rope(B=2, L=4096, H=32, rope_theta=10000.0):
    dtype = torch.bfloat16
    device = 'cuda:0'
    q = torch.randn(L, B, H, 192, dtype=dtype, device=device)
    kv = torch.randn(L, B, H, 256, dtype=dtype, device=device)
    k_pos_emb = torch.randn(L, B, 1, 64+512, dtype=dtype, device=device)[:,:,:,:64]
    freqs = rope_freqs(L, 64, rope_theta=rope_theta)
    freqs = torch.cat([freqs, freqs], -1)
    freqs = freqs[:,None,None]
    q_grad = torch.randn(L, B, H, 192, dtype=dtype, device=device)
    k_grad = torch.randn(L, B, H, 192, dtype=dtype, device=device)
    v_grad = torch.randn(L, B, H, 128, dtype=dtype, device=device)

    mscale = 1.0
    qo, ko, vo = triton_mla_rope_forward(q.clone().detach(), kv, k_pos_emb, freqs, mscale=mscale)
    dq, dkv, dp = triton_mla_rope_backward(q_grad, k_grad, v_grad, freqs, mscale=mscale)

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


    lbh = L*B*H
    benchmark_func(fused_apply_mla_rope_for_q, q, rotary_pos_cos, rotary_pos_sin, 
                128, 64, cu_seqlens_q=None, cp_rank=0, cp_size=1,
                    ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2),
                    n_profile=0)
    benchmark_func(fused_apply_mla_rope_for_kv, kv, k_pos_emb, rotary_pos_cos, rotary_pos_sin, 
                64, 128, 128, cu_seqlens_kv=None, cp_rank=0, cp_size=1,
                    ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2),
                    n_profile=0)
    benchmark_func(query_ref.backward, 
                            gradient=query_ref.clone().detach(), 
                            retain_graph=True, 
                            ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2))
    benchmark_func(key_ref.backward, 
                            gradient=key_ref.clone().detach(), 
                            retain_graph=True, 
                            ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2))
    benchmark_func(value_ref.backward, 
                            gradient=value_ref.clone().detach(), 
                            retain_graph=True, 
                            ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2))
    benchmark_func(triton_mla_rope_forward, q, kv, k_pos_emb, freqs,
                    ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2),
                    n_profile=0)
    benchmark_func(triton_mla_rope_backward, q_grad, k_grad, v_grad, freqs,
                    ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2),
                    n_profile=0)


def bench_varlen_mla_rope(lengths=[2048,2048], H=32, rope_theta=10000.0,
                   cp_size=1, cp_rank=0):
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

    q_grad = torch.randn(sum(lengths)//cp_size, H, 192, dtype=dtype, device=device)
    k_grad = torch.randn(sum(lengths)//cp_size, H, 192, dtype=dtype, device=device)
    v_grad = torch.randn(sum(lengths)//cp_size, H, 128, dtype=dtype, device=device)

    mscale = 1.0
    qo, ko, vo = triton_mla_rope_forward(qc.detach().clone(), kvc, k_pos_embc, freqs, mscale=mscale, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, cp_size=cp_size, cp_rank=cp_rank)

    dq, dkv, dp = triton_mla_rope_backward(q_grad, k_grad, v_grad, freqs, mscale=mscale, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, cp_size=cp_size, cp_rank=cp_rank)

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


    lbh = sum(lengths)//cp_size*H
    benchmark_func(fused_apply_mla_rope_for_q, qc, rotary_pos_cos, rotary_pos_sin, 128, 64,
                cu_seqlens_q,  cp_size=cp_size, cp_rank=cp_rank,
                    ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2),
                    n_profile=0)
    benchmark_func(fused_apply_mla_rope_for_kv, kvc, k_pos_embc, rotary_pos_cos, rotary_pos_sin,
                    64, 128, 128, cu_seqlens_kv, cp_size=cp_size, cp_rank=cp_rank,
                    ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2),
                    n_profile=0)
    benchmark_func(query_ref.backward, 
                            gradient=query_ref.clone().detach(), 
                            retain_graph=True, 
                            ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2))
    benchmark_func(key_ref.backward, 
                            gradient=key_ref.clone().detach(), 
                            retain_graph=True, 
                            ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2))
    benchmark_func(value_ref.backward, 
                            gradient=value_ref.clone().detach(), 
                            retain_graph=True, 
                            ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2))
    benchmark_func(triton_mla_rope_forward, qc, kvc, k_pos_embc, freqs, mscale=mscale, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, cp_size=cp_size, cp_rank=cp_rank,
                    ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2),
                    n_profile=0)
    benchmark_func(triton_mla_rope_backward, q_grad, k_grad, v_grad, freqs, mscale=mscale, cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv, cp_size=cp_size, cp_rank=cp_rank,
                    ref_bytes=lbh * (64*2 + 256*2 + 64*2 + 192*2 + 128*2),
                    n_profile=0)



if __name__ == '__main__':
    bench_mla_rope(L=4096, B=4, H=32)
    bench_varlen_mla_rope(lengths=[2048,2048], H=32, rope_theta=10000.0,
                   cp_size=1, cp_rank=0)
