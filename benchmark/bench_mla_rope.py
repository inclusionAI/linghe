
import random
import torch
from linghe.tools.benchmark import benchmark_func
from linghe.tools.util import output_check

from megatron.core.fusions.fused_mla_yarn_rope_apply import (
    fused_apply_mla_rope_for_kv,
    fused_apply_mla_rope_for_q,
)



def bench_mla_rope(L=4096, B=2, H=16):
    dtype = torch.bfloat16
    device = 'cuda:0'
    q = torch.randn((L, B, H, 192), dtype=dtype, device=device).requires_grad_()
    rotary_pos_cos = torch.randn((L, 1, 1, 64), dtype=dtype, device=device)
    rotary_pos_sin = torch.randn((L, 1, 1, 64), dtype=dtype, device=device)
    kv = torch.randn((L, B, H, 256), dtype=dtype, device=device).requires_grad_()
    k_pos_emb = torch.randn((L, B, 1, 64), dtype=dtype, device=device).requires_grad_()

    cp_rank = 0
    cp_size = 1
    query = fused_apply_mla_rope_for_q(
        q,
        rotary_pos_cos,
        rotary_pos_sin,
        128,
        64,
        cu_seqlens_q=None,
        cp_rank=cp_rank,
        cp_size=cp_size
    )
    key, value = fused_apply_mla_rope_for_kv(
        kv,
        k_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        64,
        128,
        128,
        cu_seqlens_kv=None,
        cp_rank=cp_rank,
        cp_size=cp_size
    )
    ref_bytes = L*B*H*64*4
    ref_time = benchmark_func(fused_apply_mla_rope_for_q, 
                             q, 
                             rotary_pos_cos, 
                             rotary_pos_sin, 
                             128,
                             64, 
                            ref_bytes=ref_bytes)
    ref_time = benchmark_func(query.backward, 
                             gradient=query.clone().detach(), 
                             retain_graph=True, 
                            ref_bytes=ref_bytes)

    ref_bytes = L*B*H*256*2 + L*B*H*320*2 + L*B*H*64*2
    ref_time = benchmark_func(fused_apply_mla_rope_for_kv, 
                             kv, 
                             k_pos_emb,
                             rotary_pos_cos, 
                             rotary_pos_sin, 
                             64,
                             128,
                             128, 
                            ref_bytes=ref_bytes)
    ref_time = benchmark_func(key.backward, 
                             gradient=key.clone().detach(), 
                             retain_graph=True, 
                            ref_bytes=ref_bytes)
    ref_time = benchmark_func(value.backward, 
                             gradient=value.clone().detach(), 
                             retain_graph=True, 
                            ref_bytes=ref_bytes)

if __name__ == '__main__':
    bench_mla_rope(L=4096, B=4, H=32)
