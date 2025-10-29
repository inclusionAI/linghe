# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.utils.norm import triton_rms_norm_forward, triton_rms_norm_backward

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


