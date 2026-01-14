# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
from megatron.core.transformer.moe.moe_utils import topk_softmax_with_capacity

from linghe.facade.topk import fused_topk, group_topk_score
from linghe.tools.benchmark import benchmark_func


def bench_topk(M=4096, N=256, k=8):
    device = 'cuda:0'
    logits = torch.randn((M, N), dtype=torch.float32, device=device)
    logits = logits.detach().clone().requires_grad_()

    values_ref, indices_ref = torch.topk(
        logits,
        k)
    # print(f'{indices_ref[0]}')
    values, indices = fused_topk(logits,
                                 k)
    # print(f'{indices[0]}')

    ref_time = benchmark_func(torch.topk,
                              logits,
                              k,
                              ref_bytes=M * N * 4)
    benchmark_func(fused_topk,
                   logits,
                   k,
                   ref_time=ref_time,
                   ref_bytes=M * N * 4)


def bench_group_topk_score(M=4096, N=256, k=8):
    device = 'cuda:0'
    # logits = torch.randn((M, N), dtype=torch.float32, device=device)
    logits = torch.zeros((M, N), dtype=torch.float32, device=device) + 1.0
    logits = logits.detach().clone().requires_grad_()

    input_grad = 1 / M * torch.randn((M, N), dtype=torch.float32, device=device)
    num_groups = 32
    group_topk = 4
    scaling_factor = 2.5
    deterministic_mode = True
    score_function = 'sigmoid'
    expert_bias = torch.randn((N,), dtype=torch.float32, device=device)
    moe_router_fusion = True

    probs_ref, routing_map_ref, tokens_per_expert_ref = topk_softmax_with_capacity(
        logits,
        k,
        None,
        None,
        None,
        False,
        num_groups,
        group_topk,
        scaling_factor,
        deterministic_mode,
        score_function,
        expert_bias,
        moe_router_fusion)
    # print((-routing_map_ref[0].float()).argsort(0))

    probs, routing_map, tokens_per_expert = group_topk_score(
        logits,
        k,
        expert_bias,
        num_groups=num_groups,
        group_topk=group_topk,
        scaling_factor=scaling_factor,
        score_function=score_function)
    # print((-routing_map[0].float()).argsort(0))

    ref_time = benchmark_func(topk_softmax_with_capacity,
                              logits,
                              k,
                              None,
                              None,
                              None,
                              False,
                              num_groups,
                              group_topk,
                              scaling_factor,
                              deterministic_mode,
                              score_function,
                              expert_bias,
                              moe_router_fusion,
                              ref_bytes=M * N * 4)

    benchmark_func(group_topk_score,
                   logits,
                   k,
                   expert_bias,
                   num_groups=num_groups,
                   group_topk=group_topk,
                   scaling_factor=scaling_factor,
                   score_function=score_function,
                   ref_bytes=M * N * 4,
                   ref_time=ref_time)


if __name__ == '__main__':
    bench_topk(M=8192, N=256, k=8)
    bench_group_topk_score(M=8192, N=256, k=8)
