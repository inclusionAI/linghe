# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.utils.gate import triton_group_rms_norm_gate_forward, \
    triton_group_rms_norm_gate_backward


class GroupRMSNormGateFunction(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, attn_output, gate, weight, eps=1e-6, group_size=4):
        output = triton_group_rms_norm_gate_forward(
            attn_output,
            gate,
            weight,
            eps=eps,
            group_size=group_size
        )
        ctx.save_for_backward(attn_output, gate, weight)
        ctx.eps = eps
        ctx.group_size = group_size

        return output

    @staticmethod
    def backward(ctx, dy):
        attn_output, gate, weight = ctx.saved_tensors

        dx, dg, dw = triton_group_rms_norm_gate_backward(
            dy,
            attn_output,
            gate,
            weight,
            ctx.eps,
            ctx.group_size
        )

        return dx, dg, dw, None, None


def group_rms_norm_gate(attn_output: torch.Tensor,
                        gate: torch.Tensor,
                        weight: torch.Tensor,
                        eps: float = 1e-6,
                        group_size: int = 4):
    """
    return group_rms_norm(transpose(attn_output, [0,1]), weight) * sigmoid(gate)
    Args:
        attn_output: output of core attn, shape [bs, length, n_heads, head_dim]
        gate: gate tensor for attention output, shape [length, bs, dim]
        weight: weight of RMS norm, shape [dim]
        eps: epsilon for RMS
        group_size: group size of group RMS norm
    Returns:
        output with shape [length, bs, dim]
    """
    return GroupRMSNormGateFunction.apply(attn_output, gate, weight, eps,
                                          group_size)
