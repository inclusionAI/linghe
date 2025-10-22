# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.utils.norm import triton_rms_norm_forward, triton_rms_norm_backward, \
    triton_group_norm_gate_forward, triton_group_norm_gate_backward


class RMSNormFunction(torch.autograd.Function):
    """"""
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        output = triton_rms_norm_forward(
            x,
            weight,
            eps
        )
        # ctx.save_for_backward(x, weight, norm)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps

        return output

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors

        dx, dw = triton_rms_norm_backward(
            dy,
            x,
            weight,
            ctx.eps
        )

        return dx, dw, None


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    """
    rms norm of x with weight
    Args:
        x: activation tensor
        weight: weight tensor
        eps: epsilon for RMS

    Returns:
        rms output
    """
    assert x.contiguous()
    assert weight.contiguous()
    return RMSNormFunction.apply(x, weight, eps)

class GroupRMSNormGateFunction(torch.autograd.Function):
    """"""
    @staticmethod
    def forward(ctx, attn_output, gate, weight, eps=1e-6, group_size=4):
        output = triton_group_norm_gate_forward(
            attn_output,
            gate,
            weight.data,
            eps=eps,
            group_size=group_size
        )
        ctx.save_for_backward(attn_output, gate, weight.data)
        ctx.eps = eps
        ctx.group_size = group_size

        return output

    @staticmethod
    def backward(ctx, dy):
        attn_output, gate, weight = ctx.saved_tensors

        dx, dg, dw = triton_group_norm_gate_backward(
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
    return GroupRMSNormGateFunction.apply(attn_output, gate, weight, eps, group_size)