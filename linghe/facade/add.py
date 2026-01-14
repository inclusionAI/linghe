# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.utils.add import triton_inplace_add


class InplaceAddFunction(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor):
        return triton_inplace_add(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


def inplace_add(x: torch.Tensor, y: torch.Tensor):
    """
    inplace add y to x with mix precise
    Args:
        x: to be updated
        y: add to x
    Returns:
        updated x tensor
    """
    return InplaceAddFunction.apply(x, y)
