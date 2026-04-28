# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.utils.gate import (triton_group_rms_norm_gate_forward,
                               triton_group_rms_norm_gate_backward,
                               triton_group_rms_norm_gate_and_mxfp8_quant_forward,
                               triton_group_rms_norm_gate_and_mxfp8_quant_backward)


class GroupRMSNormGateFunction(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, attn_output, gate, weight, eps=1e-6, group_size=4):

        output = triton_group_rms_norm_gate_forward(
            attn_output, gate, weight, eps=eps, group_size=group_size
        )

        ctx.save_for_backward(attn_output, gate, weight)
        ctx.eps = eps
        ctx.group_size = group_size

        return output

    @staticmethod
    def backward(ctx, dy):
        attn_output, gate, weight = ctx.saved_tensors

        dx, dg, dw = triton_group_rms_norm_gate_backward(
            dy, attn_output, gate, weight, eps=ctx.eps, group_size=ctx.group_size
        )

        return dx, dg, dw, None, None


def group_rms_norm_gate(attn_output: torch.Tensor,
                        gate: torch.Tensor,
                        weight: torch.Tensor,
                        eps: float = 1e-6,
                        group_size: int = 4,
                        transpose: bool = True):
    """
    return group_rms_norm(transpose(attn_output, [0,1]), weight) * sigmoid(gate)
    Args:
        attn_output: output of core attn, shape [bs, length, n_heads, head_dim]
        gate: gate tensor for attention output, shape [length, bs, dim]
        weight: weight of RMS norm, shape [dim]
        eps: epsilon for RMS
        group_size: group size of group RMS norm
        transpose: whether gate is transposed
    Returns:
        output with shape [length, bs, dim]
    """
    assert transpose
    return GroupRMSNormGateFunction.apply(attn_output, gate, weight, eps,
                                          group_size)


class Mxfp8GroupRMSNormGateFunction(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, attn_output, gate, weight, quantizer, grad_quantizer, cls, eps=1e-6, group_size=4):

        shape = attn_output.shape 
        assert len(shape) == 3  

        x_q, x_s, xt_q, xt_s = triton_group_rms_norm_gate_and_mxfp8_quant_forward(
            attn_output, gate, weight, eps=eps, group_size=group_size
        )

        output = cls(
            shape=x_q.shape,
            dtype=input.dtype,
            fp8_dtype=quantizer.dtype,
            rowwise_data=x_q.view(shape),
            rowwise_scale_inv=x_s,
            columnwise_data=xt_q.view(shape),
            columnwise_scale_inv=xt_s,
            quantizer=quantizer,
            requires_grad=input.requires_grad,
        )

        ctx.save_for_backward(attn_output, gate, weight)
        ctx.eps = eps
        ctx.group_size = group_size
        ctx.shape = shape
        ctx.grad_quantizer = grad_quantizer
        ctx.cls = cls

        return output

    @staticmethod
    def backward(ctx, dy):
        attn_output, gate, weight = ctx.saved_tensors
        grad_quantizer = ctx.grad_quantizer

        dx, dg_q, dg_s, dgt_q, dgt_s, dw = (
            triton_group_rms_norm_gate_and_mxfp8_quant_backward(
                dy, attn_output, gate, weight, eps=ctx.eps, group_size=ctx.group_size
            )
        )

        dg_out = ctx.cls(
            shape=ctx.shape,
            dtype=dy.dtype,
            fp8_dtype=grad_quantizer.dtype,
            rowwise_data=dg_q.view(ctx.shape) if dg_q is not None else None,
            rowwise_scale_inv=dg_s,
            columnwise_data=dgt_q.view(ctx.shape) if dgt_q is not None else None,
            columnwise_scale_inv=dgt_s,
            quantizer=grad_quantizer,
            requires_grad=attn_output.requires_grad,
        )

        return dx, dg_out, dw, None, None, None, None, None
