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
                     block_size_m=16,
                     padding_value=9,
                     topk=9,
                     ):
    x = torch_group_dequant(xq, xs)
    weights = []
    for i in range(wq.size(0)):
        weights.append(torch_blockwise_dequant(wq[i], ws[i]))
    tids = torch.split(token_ids, block_size_m, dim=0)
    expert_ids = expert_ids.tolist()
    token_count = token_count.item()

    for i, eid in enumerate(expert_ids):
        if i * block_size_m >= token_count:
            continue
        weight = weights[eid]
        token_ids = tids[i]
        token_ids = token_ids[token_ids != padding_value]
        a = x[token_ids // topk]
        y = a@weight.t()
        c[token_ids] = y.to(c.dtype)
    return c



def test_fp8_group_gemm(M=4, N=512, K=4096, n_experts=257, topk=9, block_size_m=16, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)
    w = torch.randn(n_experts * N, K, dtype=dtype, device=device) * 0.1
    A, A_scale = torch_group_quant(x)
    wq, ws = torch_block_quant(w)
    B = wq.view(n_experts, N, K)
    B_scale = ws.view(n_experts, N//128, K//128)

    padding_value = M * topk

    top_indices = (torch.rand(M, topk, dtype=dtype) * (n_experts - 1)).to(torch.int32)
    top_indices[:, topk-1] = n_experts-1
    top_indices = top_indices.view(-1)
    counts = torch.bincount(top_indices)
    top_indices = top_indices.tolist()
    sorted_token_ids = []
    expert_ids = []
    for i, count in enumerate(counts):
        if count == 0:
            continue
        for j, k in enumerate(top_indices):
            if k == i:
                sorted_token_ids.append(j)
        pad = (block_size_m - count % block_size_m) % block_size_m
        if pad > 0:
            sorted_token_ids.extend([padding_value] * pad)
        ec = (count - 1)//block_size_m + 1
        expert_ids.extend([i] * ec)

    num_tokens_post_padded = torch.tensor([len(sorted_token_ids)], dtype=torch.int32, device=device)
    sorted_token_ids = torch.tensor(sorted_token_ids, device=device, dtype=torch.int32)
    expert_ids = torch.tensor(expert_ids, device=device, dtype=torch.int32)
    y_ref = torch.zeros(padding_value, N, dtype=dtype, device=device)
    y_ref = torch_group_gemm(A,
                     B,
                     A_scale,
                     B_scale,
                     sorted_token_ids,
                     expert_ids,
                     num_tokens_post_padded,
                     c=y_ref,
                     block_size_m=block_size_m,
                     padding_value=padding_value,
                     topk=topk)
    y = torch.zeros(padding_value, N, dtype=dtype, device=device)
    y = triton_fp8_grouped_gemm(A,
                     B,
                     A_scale,
                    #  A_scale.t().contiguous().t(),
                     B_scale,
                     sorted_token_ids,
                     expert_ids,
                     num_tokens_post_padded,
                     c=y,
                     block_size_m=block_size_m,
                     padding_value=padding_value,
                     topk=topk
    )

    output_check(y_ref, y, name='y', atol=5e-2, rtol=2e-2)


    if bench:
        M = sorted_token_ids.size(0)
        act_exp = num_tokens_post_padded.item() // block_size_m
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
                        block_size_m=block_size_m,
                        padding_value=padding_value,
                        topk=topk,
                        ref_bytes=ref_bytes,
                        ref_flops=ref_flops,
                        n_profile=10)


if __name__ == '__main__':
    test_fp8_group_gemm(M=4, N=512, K=4096, n_experts=257, topk=9, block_size_m=16, bench=True)
    test_fp8_group_gemm(M=4, N=512, K=4096, n_experts=257, topk=9, block_size_m=32, bench=True)
    test_fp8_group_gemm(M=255, N=512, K=4096, n_experts=257, topk=9, block_size_m=32, bench=True)

