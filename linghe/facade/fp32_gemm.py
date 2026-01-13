# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.gemm.fp32_gemm import (triton_fp32_gemm,
                                   triton_fp32_gemm_for_backward,
                                   triton_fp32_gemm_for_update)


class Fp32GEMM(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor):
        shape = input.shape
        if len(shape) == 3:
            input = input.view(shape[0] * shape[1], shape[2])
        logits = triton_fp32_gemm(input, weight)

        ctx.input_requires_grad = input.requires_grad
        ctx.weight_requires_grad = weight.requires_grad
        ctx.shape = shape
        ctx.save_for_backward(input, weight)
        if len(shape) == 3:
            logits = logits.view(shape[0], shape[1], weight.shape[0])
        return logits

    @staticmethod
    def backward(ctx, grad_output):
        grad_shape = grad_output.shape
        if len(grad_shape) == 3:
            grad_output = grad_output.view(grad_shape[0] * grad_shape[1],
                                           grad_shape[2])

        input, weight = ctx.saved_tensors

        dx = triton_fp32_gemm_for_backward(grad_output, weight)
        if len(grad_shape) == 3:
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
