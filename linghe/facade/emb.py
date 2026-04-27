# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.utils.emb import triton_embedding_forward, triton_embedding_backward


class DeprecatedFusedAccumulationEmbeddingLookup(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, x, w_ptr, g_ptr, dim, dtype, grad_dtype):
        x = x.long()
        ctx.grad_dtype = grad_dtype
        ctx.g_ptr = g_ptr
        ctx.save_for_backward(x)
        return triton_embedding_forward(x, w_ptr, dim, dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        triton_embedding_backward(grad_output, x, ctx.g_ptr, ctx.grad_dtype)
        return None, None, None, None, None, None


def deprecated_fused_accumulation_embedding_lookup(
    x: torch.Tensor, w_ptr, g_ptr, dim, dtype, grad_dtype
):
    """
    embedding lookup
    Args:
        x: input ids
        w_ptr:
        g_ptr:
        dim:
        dtype:
        grad_dtype:
    Returns:
        lookup output
    """
    x = x.double().requires_grad_()
    return DeprecatedFusedAccumulationEmbeddingLookup.apply(
        x, w_ptr, g_ptr, dim, dtype, grad_dtype
    )


class FusedAccumulationEmbeddingLookup(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, x, w, grad_name):
        dim = w.size(1)
        dtype = w.dtype
        ctx.save_for_backward(x, w)
        ctx.grad_name = grad_name
        return triton_embedding_forward(x, w.data_ptr(), dim, dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad = getattr(w, ctx.grad_name)
        triton_embedding_backward(grad_output, x, grad.data_ptr(), grad.dtype)
        return None, None, None


def fused_accumulation_embedding_lookup(
    x: torch.Tensor, w: torch.nn.Parameter, grad_name: str = "grad"
):
    """
    embedding lookup
    Args:
        x: input ids
        w: embedding weight, should contain a `grad_name` tensor
    Returns:
        lookup output
    """
    return FusedAccumulationEmbeddingLookup.apply(x, w, grad_name)


class EmbeddingLookup(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, x, w):
        dim = w.size(1)
        dtype = w.dtype
        ctx.save_for_backward(x, w)
        return triton_embedding_forward(x, w.data_ptr(), dim, dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, w = ctx.saved_tensors
        grad = torch.zeros_like(w)
        triton_embedding_backward(grad_output, x, grad.data_ptr(), grad.dtype)
        return None, grad


def embedding_lookup(x: torch.Tensor, w: torch.nn.Parameter):
    """
    embedding lookup
    Args:
        x: input ids
        w: embedding weight
    Returns:
        lookup output
    """
    return EmbeddingLookup.apply(x, w)
