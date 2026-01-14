# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.utils.topk import (triton_topk_forward,
                               triton_topk_backward,
                               triton_group_topk_score_forward,
                               triton_group_topk_score_backward)


class TopkFunction(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, x, k, dim):
        values, indices = triton_topk_forward(x, k, dim=dim)
        ctx.dim = dim
        ctx.shape = x.shape
        ctx.save_for_backward(indices)
        return values, indices

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        indices, = ctx.saved_tensors
        grad_input = triton_topk_backward(grad_output, indices, ctx.shape[-1],
                                          dim=ctx.dim)
        return grad_input, None, None


def fused_topk(x, k, dim=-1):
    """
    topk
    Args:
        x: input tensor
        k: topk
        dim: dimension to apply topk, only support -1 currently
    Returns:
        values: topk values
        indices: topk indices
    """
    return TopkFunction.apply(x, k, dim)


class GroupTopkScoreFunction(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, x, topk, expert_bias, num_groups, group_topk,
                scaling_factor, score_function):
        probs, routing_map, counts = triton_group_topk_score_forward(x,
                                                                     topk,
                                                                     expert_bias=expert_bias,
                                                                     num_groups=num_groups,
                                                                     group_topk=group_topk,
                                                                     scaling_factor=scaling_factor,
                                                                     score_function=score_function)
        ctx.save_for_backward(x, routing_map)
        ctx.scaling_factor = scaling_factor
        ctx.score_function = score_function
        return probs, routing_map, counts

    @staticmethod
    def backward(ctx, grad_output, grad_map, grad_counts):
        x, routing_map = ctx.saved_tensors
        grad_input = triton_group_topk_score_backward(grad_output,
                                                      x,
                                                      routing_map,
                                                      scaling_factor=ctx.scaling_factor)
        return grad_input, None, None, None, None, None, None


def group_topk_score(x,
                     topk,
                     expert_bias=None,
                     num_groups=32,
                     group_topk=4,
                     scaling_factor=1.0,
                     score_function='sigmoid'):
    """
    group topk with softmax/sigmoid function
    Args:
        x: input logit tensor
        topk: topk
        expert_bias: expert bias
        num_groups: number of groups
        group_topk: group to apply topk
        scaling_factor: scaling factor
        score_function: scaling function
    Returns:
        probs: topk probs
        routing_map: topk binary map
        counts: token count per expert
    """
    return GroupTopkScoreFunction.apply(x, topk, expert_bias, num_groups,
                                        group_topk, scaling_factor,
                                        score_function)
