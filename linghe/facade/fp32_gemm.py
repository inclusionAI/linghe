# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.gemm.fp32_gemm import (triton_fp32_gemm,
                                  triton_fp32_gemm_for_backward,
                                  triton_fp32_gemm_for_update)
from linghe.gemm.fp32_gemm import triton_scaled_fp32_gemm, triton_scaled_fp32_gemm_for_update
from linghe.utils.norm import triton_rms_norm_backward, triton_rms_norm_forward

class Fp32GEMM(torch.autograd.Function):
    """"""
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor):
        shape = input.shape
        assert len(shape) == 3
        input = input.view(shape[0] * shape[1], shape[2])

        logits = triton_fp32_gemm(input, weight.data)

        ctx.input_requires_grad = input.requires_grad
        ctx.weight_requires_grad = weight.requires_grad
        ctx.shape = shape
        ctx.save_for_backward(input, weight.data)

        return logits.view(shape[0], shape[1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        shape = grad_output.shape
        grad_output = grad_output.view(shape[0] * shape[1], shape[2])
        input, weight = ctx.saved_tensors

        dx = triton_fp32_gemm_for_backward(grad_output, weight)
        dx = dx.view(*ctx.shape)

        dw = triton_fp32_gemm_for_update(grad_output, input)

        return dx, dw


def fp32_gemm(input: torch.Tensor, weight: torch.Tensor):
    """
    gemm with bf16/fp16 inputs and float32 output,
    currently used in MoE router gemm.
    Args:
        input: bf16/fp16 activation tensor
        weight: bf16/fp16 weight tensor
    Returns:
        output of gemm
    """
    return Fp32GEMM.apply(input, weight)


class RMSNormMoERoute(torch.autograd.Function):
    """"""
    @staticmethod
    def forward(ctx, input, norm_weight, route_weight, eps):
        """
        a naive example to warp rms norm and moe route in an Function
        high-precision input should be used for routing, to make it compatiable with quantization, we rearange the calculation:
        1. rms norm and quantize, the rms norm must output the `rms` value
        2. calculation the route logits with the bf16 input and `rms`, as x/rms@w=x@w/rms
        it can reduce the memory as we do not need to save the fp32 input for backward
        """
        shape = input.shape 
        assert len(shape) == 3
        input = input.view(shape[0]*shape[1], shape[2])

        y, rms = triton_rms_norm_forward(input, 
                                                                            norm_weight, 
                                                                            eps=eps)

        # NOTE: rms is stored as 1/rms
        logits = triton_scaled_fp32_gemm(input, route_weight, rms)

        ctx.input_requires_grad = input.requires_grad
        ctx.shape = shape 
        ctx.eps = eps
        ctx.save_for_backward(input, norm_weight, route_weight, rms)

        return y, logits

    @staticmethod
    def backward(ctx, grad_output, grad_logits):
        """"""
        shape = grad_output.shape 
        grad_output = grad_output.view(shape[0]*shape[1], shape[2])
        input, norm_weight, route_weight, rms = ctx.saved_tensors

        grad_output = triton_fp32_gemm_for_backward(grad_logits, route_weight, grad_output)

        route_dw = triton_scaled_fp32_gemm_for_update(grad_logits, input, rms)
        
        dx, norm_dw = triton_rms_norm_backward(grad_output, input, norm_weight, eps=ctx.eps)
        dx = dx.view(*shape)

        return dx, norm_dw, route_dw, None


def rms_norm_and_route(input, norm_weight, route_weight, eps=1e-6):
    # input: [length,bs,dim]
    output, logits = RMSNormMoERoute.apply(input, norm_weight, route_weight, eps)
    return output, logits