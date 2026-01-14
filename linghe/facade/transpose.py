# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.utils.transpose import triton_transpose


class TransposeFunction(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, x, inner):
        ctx.inner = inner
        return triton_transpose(x, inner=inner)

    @staticmethod
    def backward(ctx, grad_output):
        return triton_transpose(grad_output, inner=ctx.inner)


def transpose(x, inner=True):
    """
    transpose a tensor, x.ndims should not greater than 4
    Args:
        x: input tensor
        inner: 
            if True, transpose the first two dimensions
            if False, transpose the last two dimensions
    Returns:
        a transposed tensor
    """
    return TransposeFunction.apply(x, inner)
