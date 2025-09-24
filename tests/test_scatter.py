# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from flops.tools.benchmark import benchmark_func
from flops.tools.util import (output_check,
                              torch_make_indices)
from flops.utils.scatter import (triton_scatter_add,
                                 triton_unpermute_with_mask_map
                                 )


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def torch_fp16_scatter_add(x, outputs, indices, weights):
    if weights is not None:
        x = x * weights[:, None]
    dim = x.size(1)
    outputs.scatter_add_(0, indices.unsqueeze(1).expand(-1, dim), x)
    return outputs


def test_scatter(M=4098, N=4096, n_experts=32, topk=2, bias=0.0, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    logits = torch.randn((M, n_experts), dtype=torch.float32, device=device)
    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
        logits, topk=topk, bias=bias)

    token_count_per_expert_list = token_count_per_expert.tolist()
    out_tokens = sum(token_count_per_expert_list)

    x = torch.randn(out_tokens, N, dtype=dtype, device=device)

    outputs = torch.zeros((M, N), dtype=dtype, device=device)

    sums_ref = torch_fp16_scatter_add(x, outputs.clone(), indices, None)
    counts = mask_map.sum(1)
    unpermuted_prob = probs.T.contiguous().masked_select(
        mask_map.T.contiguous())

    sums_unpermute, output_prob = triton_unpermute_with_mask_map(x, row_id_map,
                                                                 unpermuted_prob)
    output_check(sums_ref, sums_unpermute, 'unpermute_data')
    output_check(probs, output_prob, 'unpermute_prob')

    if bench:
        n_repeat = 100
        # ref_time = benchmark_func(torch_fp16_scatter_add,x, outputs, indices, weights,n_repeat=n_repeat)
        # benchmark_func(triton_aligned_scatter_add,x, outputs, indices, weights=weights, n_repeat=n_repeat,ref_time=ref_time)
        ref_time = benchmark_func(triton_scatter_add, x, outputs, indices,
                                  n_repeat=n_repeat)
        benchmark_func(triton_unpermute_with_mask_map, x, row_id_map, probs,
                       n_repeat=n_repeat, ref_time=ref_time)


if __name__ == '__main__':
    test_scatter(M=4098, N=4096, n_experts=32, topk=2, bias=0.0)
    test_scatter(M=2467, N=4096, n_experts=32, topk=2, bias=-0.1)
