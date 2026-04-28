# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import pytest
import torch

from linghe.tools.check import output_check
from linghe.tools.util import torch_make_indices
from linghe.utils.scatter import triton_scatter_add, triton_unpermute_with_mask_map


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def torch_scatter_add(x, outputs, indices, weights):
    dtype = x.dtype
    x = x.float()
    if weights is not None:
        x = x * weights[:, None]
    dim = x.size(1)
    outputs = outputs.float()
    outputs.scatter_add_(0, indices.unsqueeze(1).expand(-1, dim), x)
    return outputs.to(dtype)


@pytest.mark.parametrize(
    "M,N,bias",
    [
        (4098, 4096, 0.0),
        (2467, 4096, -0.1),
        (2467, 1536, -0.1),
    ],
)
def test_scatter(M, N, bias, benchmark, n_experts=32, topk=2):
    dtype = torch.bfloat16
    device = "cuda:0"

    logits = torch.randn((M, n_experts), dtype=torch.float32, device=device)
    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
        logits, topk=topk, bias=bias
    )

    token_count_per_expert_list = token_count_per_expert.tolist()
    out_tokens = sum(token_count_per_expert_list)

    x = torch.randn(out_tokens, N, dtype=dtype, device=device)

    outputs = torch.zeros((M, N), dtype=dtype, device=device)

    sums_ref = torch_scatter_add(x, outputs.clone(), indices, None)
    unpermuted_prob = probs.T.contiguous().masked_select(mask_map.T.contiguous())

    sums_unpermute, output_prob = triton_unpermute_with_mask_map(
        x, row_id_map, unpermuted_prob
    )
    output_check(sums_ref, sums_unpermute, "unpermute_data")
    output_check(probs, output_prob, "unpermute_prob")

    n_repeat = 100
    ref_time = benchmark(triton_scatter_add, x, outputs, indices, n_repeat=n_repeat)
    benchmark(
        triton_unpermute_with_mask_map,
        x,
        row_id_map,
        probs,
        n_repeat=n_repeat,
        ref_time=ref_time,
    )
