# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.utils.emb import triton_embedding_forward, triton_embedding_backward


class EmbeddingLookup(torch.autograd.Function):
    """"""
    @staticmethod
    def forward(ctx, x, w_ptr, g_ptr, dim, dtype, grad_dtype, dummy_tensor):
        ctx.grad_dtype = grad_dtype
        ctx.g_ptr = g_ptr
        ctx.save_for_backward(x)
        return triton_embedding_forward(x, w_ptr, dim, dtype)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        triton_embedding_backward(grad_output, x, ctx.g_ptr, ctx.grad_dtype)
        return None, None, None, None, None, None, None, None


def embedding_lookup(x: torch.Tensor, w_ptr, g_ptr, dim, dtype, grad_dtype, dummy_tensor):
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
    return EmbeddingLookup.apply(x, w_ptr, g_ptr, dim, dtype, grad_dtype, dummy_tensor)