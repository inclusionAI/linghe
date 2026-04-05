# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.infer.topk import triton_group_topk_score


def group_limited_topk(
        scores: torch.Tensor,
        topk: int,
        num_tokens: int,
        num_experts: int,
        num_groups: int,
        group_topk: int,
):
    # Organize the experts into groups
    # Select groups based on sum of top-(topk/group_topk) routing scores within each group
    group_scores = (
        scores.view(num_tokens, num_groups, -1).topk(topk // group_topk,
                                                     dim=-1)[0].sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=group_topk, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    # Mask the experts based on selection groups
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_tokens, num_groups, num_experts // num_groups)
        .reshape(num_tokens, -1)
    )

    masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
    probs, top_indices = torch.topk(masked_scores, k=topk, dim=-1)

    return probs, top_indices


def torch_group_topk_score(logits, expert_bias, num_experts=256, topk=8,
                           num_groups=32, group_topk=4, num_shared_experts=1,
                           scaling_factor=1.0,
                           eps=1e-20):
    num_tokens, num_experts = logits.shape
    device = logits.device
    scores = torch.sigmoid(logits.to(torch.float64))
    expert_bias = expert_bias.to(torch.float64)
    scores_for_routing = scores + expert_bias - torch.arange(0, num_experts,
                                                                device=device).to(
        torch.float64) * 1e-12
    _, top_indices = group_limited_topk(scores_for_routing, topk,
                                        num_tokens, num_experts, num_groups,
                                        group_topk)
    # should output the score before biased
    top_scores = torch.gather(scores, dim=1, index=top_indices)

    top_scores = top_scores / (
                top_scores.sum(dim=-1, keepdim=True) + eps) 

    if num_shared_experts == 1:
        top_scores = torch.cat([top_scores, torch.ones((num_tokens, 1), device=device, dtype=torch.float64)/scaling_factor], 1)
        top_indices = torch.cat([top_indices, num_experts*torch.ones((num_tokens, 1), device=device, dtype=top_indices.dtype)], 1)

    return top_scores.to(torch.float32), top_indices.to(torch.int32)


def test_group_topk_score(M=4096, N=256, k=8, num_groups=32, group_topk=4, num_shared_experts=1,
                          scaling_factor=1.0, bias_coef=0.01, equal=False, 
                          bench=False):
    dtype = torch.float32
    device = 'cuda:0'

    x = torch.randn(M, N, dtype=dtype, device=device)

    expert_bias = torch.randn(N, dtype=dtype, device=device) * bias_coef
    if equal:
        x[:] = 1.0
        expert_bias[:] = 0.0

    score_ref, indices_ref = torch_group_topk_score(x,
                                                    expert_bias,
                                                    num_experts=N,
                                                    topk=k,
                                                    num_groups=num_groups,
                                                    group_topk=group_topk,
                                                    scaling_factor=scaling_factor,
                                                    num_shared_experts=num_shared_experts)

    score, indices = triton_group_topk_score(x, 
                                             k,
                                             expert_bias,
                                             num_groups=num_groups,
                                             group_topk=group_topk,
                                             scaling_factor=scaling_factor,
                                             num_shared_experts=num_shared_experts)
    ref_sort_indices = torch.argsort(indices_ref)
    indices_ref = indices_ref.gather(-1, ref_sort_indices)
    score_ref = score_ref.gather(-1, ref_sort_indices)
    sort_indices = torch.argsort(indices)
    indices = indices.gather(-1, sort_indices)
    score = score.gather(-1, sort_indices)
    if (indices_ref-indices).abs().sum().item() > 0:
        diff =  (indices_ref-indices).abs().sum(-1)
        row = diff.argmax().item()
        ref_idx = indices_ref[row,:k]
        idx = indices[row,:k]
        print(f'ref {x[row][ref_idx].sort(descending=True)}')
        print(f'ours {x[row][idx].sort(descending=True)}')
    output_check(score_ref, score, 'score')
    output_check(indices_ref, indices, 'indices')


    if bench:
        ref_time = benchmark_func(torch_group_topk_score, x,
                                  expert_bias, num_experts=N, topk=k,
                                  num_groups=num_groups, group_topk=group_topk,
                                  scaling_factor=scaling_factor,
                                  num_shared_experts=num_shared_experts)
        benchmark_func(triton_group_topk_score, x, k,
                       expert_bias=expert_bias, num_groups=num_groups,
                       group_topk=group_topk, scaling_factor=scaling_factor,
                       num_shared_experts=num_shared_experts,
                       ref_time=ref_time,
                       n_profile=10)


if __name__ == '__main__':
    test_group_topk_score(M=4, N=256, k=8, num_groups=8, group_topk=4,
                          scaling_factor=2.5, equal=False, num_shared_experts=1,
                          bench=True)
    test_group_topk_score(M=8192, N=256, k=8, num_groups=8, group_topk=4,
                          scaling_factor=2.5, equal=False, num_shared_experts=1,
                          bench=False)
    test_group_topk_score(M=8192, N=256, k=8, num_groups=8, group_topk=4,
                          scaling_factor=2.5, equal=False, num_shared_experts=1,
                          bias_coef=1.0,
                          bench=False)
    test_group_topk_score(M=8192, N=256, k=8, num_groups=8, group_topk=4,
                          scaling_factor=2.5, equal=False, num_shared_experts=0,
                          bench=False)
    test_group_topk_score(M=8192, N=256, k=8, num_groups=8, group_topk=4,
                          scaling_factor=2.5, equal=True,
                          bench=False)
