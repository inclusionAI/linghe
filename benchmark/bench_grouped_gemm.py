# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.infer.grouped_gemm import triton_fp8_grouped_gemm
from linghe.tools.util import (torch_group_quant,
                               torch_block_quant,
                               torch_group_dequant,
                               torch_blockwise_dequant)

def torch_fp64_matmul(x, w):
    return torch.nn.functional.linear(x.to(torch.float64),
                                      w.to(torch.float64)).to(torch.float32)


def torch_fp32_matmul(x, w):
    return torch.nn.functional.linear(x.to(torch.float32), w.to(torch.float32))


def torch_group_gemm(xq,
                     wq,
                     xs,
                     ws,
                     token_ids,
                     expert_ids,
                     token_count,
                     c=None,
                     padding_size=16,
                     padding_value=9,
                     topk=9,
                     ):
    x = torch_group_dequant(xq, xs)
    weights = []
    for i in range(wq.size(0)):
        weights.append(torch_blockwise_dequant(wq[i], ws[i]))
    tids = torch.split(token_ids, padding_size, dim=0)
    expert_ids = expert_ids.tolist()
    token_count = token_count.item()

    for i, eid in enumerate(expert_ids):
        if i * padding_size >= token_count:
            continue
        weight = weights[eid]
        token_ids = tids[i]
        token_ids = token_ids[token_ids != padding_value]
        a = x[token_ids // topk]
        y = a@weight.t()
        c[token_ids] = y.to(c.dtype)
    return c



def bench_fp8_group_gemm(M=2048, N=256, K=8192, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    # x = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)
    # w = (torch.randn(N, K, dtype=dtype, device=device) * 0.1).requires_grad_()
    # dy = torch.randn(M, N, dtype=torch.float32, device=device)

    # y_ref = torch_fp32_matmul(x, w)

    states = torch.load('/ossfs/workspace/script/moe_decode_fc2.bin',weights_only=False)

    A = states['A']
    B = states['B']
    C = states['C']
    A_scale = states['A_scale']
    B_scale = states['B_scale']
    bias = None
    topk_weights = states['topk_weights']
    topk_ids = states['topk_ids']
    sorted_token_ids = states['sorted_token_ids']
    expert_ids = states['expert_ids']
    num_tokens_post_padded = states['num_tokens_post_padded']
    block_shape = states['block_shape']
    mul_routed_weight = states['mul_routed_weight']
    top_k = states['top_k']
    compute_type = states['compute_type']
    use_fp8_w8a8 = states['use_fp8_w8a8']
    use_int8_w8a8 = states['use_int8_w8a8']
    use_int8_w8a16 = states['use_int8_w8a16']
    per_channel_quant = states['per_channel_quant']
    even_Ks = states['even_Ks']
    config = states['config']
    padded_size = 0
    use_int4_w4a16 = False
    B_zp = None
    
    nt = num_tokens_post_padded.item()
    A = A[:nt]
    A_scale = A_scale[:nt]
    sorted_token_ids = sorted_token_ids[:nt]
    padding_size = config['BLOCK_SIZE_M']
    padding_value = topk_ids.numel()
    topk = topk_ids.size(1)
    M = topk_ids.numel()
    N, K = B.shape[1:]
    y_ref = torch.zeros(M, N, dtype=dtype, device=device)

    from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels import invoke_fused_moe_kernel
    y_ref = invoke_fused_moe_kernel(
            A,
            B,
            bias,
            y_ref,
            A_scale,
            B_scale,
            B_zp,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape,
            no_combine = False)
    y = torch.zeros(M, N, dtype=dtype, device=device)
    y = triton_fp8_grouped_gemm(A,
                     B,
                     A_scale,
                    #  A_scale.t().contiguous().t(),
                     B_scale,
                     sorted_token_ids,
                     expert_ids,
                     num_tokens_post_padded,
                     c=y,
                     padding_size=padding_size,
                     padding_value=padding_value,
                     topk=topk
    )

    output_check(y_ref, y, name='y', atol=5e-3, rtol=2e-3)


    if bench:
        M = sorted_token_ids.size(0)
        act_exp = num_tokens_post_padded.item() // padding_size
        ref_bytes = M * K + N * K * act_exp + M * N * 2
        ref_flops = 2 * M * N * K
        
        benchmark_func(triton_fp8_grouped_gemm, 
                        A,
                        B,
                        A_scale,
                        # A_scale.t().contiguous().t(),
                        B_scale,
                        sorted_token_ids,
                        expert_ids,
                        num_tokens_post_padded,
                        c=y,
                        padding_size=padding_size,
                        padding_value=padding_value,
                        topk=topk,
                        ref_bytes=ref_bytes,
                        ref_flops=ref_flops,
                        n_profile=10)


        benchmark_func(invoke_fused_moe_kernel,
            A,
            B,
            bias,
            C,
            A_scale,
            B_scale,
            B_zp,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape,
            no_combine = False,
            n_profile=10
        )


if __name__ == '__main__':
    bench_fp8_group_gemm(M=4096, N=256, K=8192, bench=True)

