# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.utils.norm import (
    triton_rms_norm_forward,
    triton_rms_norm_backward,
    triton_rms_norm_and_block_quant_forward,
)


class RMSNormFunction(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        output, rms = triton_rms_norm_forward(x, weight, eps)
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dx, dw = triton_rms_norm_backward(dy, x, weight, ctx.eps)
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
    assert x.is_contiguous()
    assert weight.is_contiguous()
    return RMSNormFunction.apply(x, weight, eps)


# used in attention rms norm
class BlockRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, rms, eps, quantizer, cls, is_recomputing):
        shape = input.shape
        assert len(shape) == 3

        if is_recomputing is None:
            output_mode = 2
        elif is_recomputing:
            output_mode = 1
        else:
            output_mode = 0

        input_view = input.view(shape[0] * shape[1], shape[2])
        x_q, x_scale, output_rms, xt_q, xt_scale = (
            triton_rms_norm_and_block_quant_forward(
                input_view,
                weight,
                rms=rms,
                eps=eps,
                round_scale=quantizer.force_pow_2_scales,
                output_mode=output_mode,
            )
        )

        transpose_shape = (shape[2], shape[0], shape[1])
        output = cls(
            shape=shape,
            dtype=input.dtype,
            fp8_dtype=quantizer.dtype,
            rowwise_data=x_q.view(shape) if x_q is not None else None,
            rowwise_scale_inv=x_scale,
            columnwise_data=xt_q.view(transpose_shape) if xt_q is not None else None,
            columnwise_scale_inv=xt_scale,
            quantizer=quantizer,
            requires_grad=input.requires_grad,
            is_2D_scaled=False,
        )
        ctx.input_requires_grad = input.requires_grad
        ctx.weight_requires_grad = weight.requires_grad
        ctx.shape = shape
        ctx.eps = eps
        ctx.save_for_backward(input, weight)

        return output, output_rms

    @staticmethod
    def backward(ctx, grad_output, grad_rms):
        shape = grad_output.shape
        grad_output = grad_output.view(shape[0] * shape[1], shape[2])
        input, weight = ctx.saved_tensors
        input = input.view(shape[0] * shape[1], shape[2])
        dx, dw = triton_rms_norm_backward(grad_output, input, weight, eps=ctx.eps)
        dx = dx.view(*shape)

        return dx, dw, None, None, None, None, None


def block_rms_norm(input, weight, rms, quantizer, cls, eps=1e-6, is_recomputing=None):
    output, output_rms = BlockRMSNorm.apply(
        input, weight, rms, eps, quantizer, cls, is_recomputing
    )
    output_rms = output_rms.detach()
    return output, output_rms
