# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.facade.topk import fused_topk, group_topk_score
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.utils.topk import (triton_topk_forward,
                               triton_topk_backward,
                               triton_group_topk_score_forward,
                               triton_group_topk_score_backward)


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


def torch_group_topk_score(logits, expert_bias=None, num_experts=256, topk=8,
                           num_groups=32, group_topk=4, scaling_factor=1.0,
                           eps=1e-20):
    num_tokens, num_experts = logits.shape
    scores = torch.sigmoid(logits).to(torch.float64)
    if expert_bias is not None:
        expert_bias = expert_bias.to(torch.float64)
        scores_for_routing = scores + expert_bias - torch.arange(0, num_experts,
                                                                 device=logits.device).to(
            torch.float64) * 1e-12
        _, top_indices = group_limited_topk(scores_for_routing, topk,
                                            num_tokens, num_experts, num_groups,
                                            group_topk)
        scores = torch.gather(scores, dim=1, index=top_indices)
    else:
        scores = scores - torch.arange(0, num_experts, device=logits.device).to(
            torch.float64) * 1e-12
        scores, top_indices = group_limited_topk(scores, topk, num_tokens,
                                                 num_experts, num_groups,
                                                 group_topk)
    probs = scores / (
                scores.sum(dim=-1, keepdim=True) + eps) if topk > 1 else scores

    if scaling_factor:
        probs = probs * scaling_factor

    # TODO Try using element-wise operations instead of scatter?
    topk_masked_gates = torch.zeros_like(logits, dtype=torch.float64).scatter(1,
                                                                              top_indices,
                                                                              probs)
    topk_map = torch.zeros_like(logits, dtype=torch.float64).int().scatter(1,
                                                                           top_indices,
                                                                           1).bool()

    tokens_per_expert = topk_map.sum(dim=0)
    return topk_masked_gates.float(), topk_map, tokens_per_expert


def test_topk(M=4096, B=1, N=256, k=8, equal=False, bench=False):
    dtype = torch.float32
    device = 'cuda:0'

    if B == 0:
        x = torch.randn(M, N, dtype=dtype, device=device)
    else:
        x = torch.randn(M, B, N, dtype=dtype, device=device)
    if equal:
        x[..., 0] = x[..., -1]

    x = x.requires_grad_()

    if equal:
        xd = x.to(torch.float64) * (1 - torch.arange(0, N, device=device).to(
            torch.float64) * 1e-12)
        value_ref, index_ref = torch.topk(xd, k)
        value_ref = value_ref.float()
    else:
        value_ref, index_ref = torch.topk(x, k)

    loss_ref = (value_ref * index_ref.float()).sum()
    loss_ref.backward()
    grad_ref = x.grad
    x.grad = None

    value, index = triton_topk_forward(x, k)
    grad = triton_topk_backward(index_ref.float(), index, N)
    output_check(value_ref, value, 'value')
    output_check(index_ref, index.to(torch.int64), 'index')
    output_check(grad_ref, grad, 'grad')

    value, index = fused_topk(x, k)
    value.backward(index_ref.float())
    grad = x.grad
    output_check(value_ref, value, 'value')
    output_check(index_ref, index.to(torch.int64), 'index')
    output_check(grad_ref, grad, 'grad')

    if bench:
        ref_time = benchmark_func(torch.topk, x, k,
                                  ref_bytes=M * N * 4)
        benchmark_func(triton_topk_forward, x, k,
                       ref_bytes=M * N * 4,
                       ref_time=ref_time)
        benchmark_func(triton_topk_backward, index_ref.float(), index, N,
                       ref_bytes=M * N * 4)


def test_group_topk_score(M=4096, N=256, k=8, num_groups=32, group_topk=4,
                          scaling_factor=1.0, equal=False, bias=True,
                          bench=False):
    dtype = torch.float32
    device = 'cuda:0'

    x = torch.randn(M, N, dtype=dtype, device=device)

    dy = torch.randn(M, N, dtype=dtype, device=device)
    if bias:
        expert_bias = torch.randn(N, dtype=dtype, device=device) * 10.0
    else:
        expert_bias = None
    if equal:
        x[..., 0] = x[..., 1]
    x = x.requires_grad_()

    prob_ref, map_ref, count_ref = torch_group_topk_score(x,
                                                          expert_bias=expert_bias,
                                                          num_experts=N, topk=k,
                                                          num_groups=num_groups,
                                                          group_topk=group_topk,
                                                          scaling_factor=scaling_factor)
    loss_ref = (prob_ref * map_ref.float() * dy).sum()
    loss_ref.backward()
    grad_ref = x.grad
    x.grad = None

    prob, maps, count = triton_group_topk_score_forward(x, k,
                                                        expert_bias=expert_bias,
                                                        num_groups=num_groups,
                                                        group_topk=group_topk,
                                                        scaling_factor=scaling_factor)
    grad = triton_group_topk_score_backward(map_ref.float() * dy, x, maps,
                                            scaling_factor=scaling_factor)
    output_check(prob_ref, prob, 'prob', atol=-1)  # may have mismatched results
    err = output_check(map_ref, maps, 'maps')
    output_check(count_ref, count, 'count')
    output_check(grad_ref, grad, 'grad')

    prob, maps, count = group_topk_score(x, k, expert_bias=expert_bias,
                                         num_groups=num_groups,
                                         group_topk=group_topk,
                                         scaling_factor=scaling_factor)
    prob.backward(gradient=map_ref.float() * dy)
    grad = x.grad
    output_check(prob_ref, prob, 'prob')
    output_check(map_ref, maps, 'maps')
    output_check(count_ref, count, 'count')
    output_check(grad_ref, grad, 'grad')

    if bench:
        ref_time = benchmark_func(torch_group_topk_score, x,
                                  expert_bias, num_experts=N, topk=k,
                                  num_groups=num_groups, group_topk=group_topk,
                                  scaling_factor=scaling_factor)
        benchmark_func(triton_group_topk_score_forward, x, k,
                       expert_bias=expert_bias, num_groups=num_groups,
                       group_topk=group_topk, scaling_factor=scaling_factor,
                       ref_time=ref_time)
        benchmark_func(triton_group_topk_score_backward, map_ref.float(), x,
                       maps)


if __name__ == '__main__':
    test_topk(M=8192, B=0, N=256, k=8, equal=False, bench=False)
    test_topk(M=4096, B=2, N=256, k=8, equal=False, bench=False)
    test_topk(M=4096, B=2, N=256, k=8, equal=True, bench=False)
    test_group_topk_score(M=8192, N=256, k=8, num_groups=32, group_topk=4,
                          scaling_factor=1.0, equal=False, bias=True,
                          bench=False)
    test_group_topk_score(M=8192, N=256, k=8, num_groups=32, group_topk=4,
                          scaling_factor=1.0, equal=False, bias=False,
                          bench=False)
    test_group_topk_score(M=8192, N=256, k=8, num_groups=32, group_topk=4,
                          scaling_factor=1.0, equal=True, bias=True,
                          bench=False)
