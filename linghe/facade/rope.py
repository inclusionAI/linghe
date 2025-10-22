# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.utils.rope import triton_qk_norm_and_half_rope_forward, \
    triton_qk_norm_and_half_rope_backward


class QkNormHalfRopeFunction(torch.autograd.Function):
    """"""
    @staticmethod
    def forward(ctx, qkv, q_norm_weight, k_norm_weight, freqs, H=32, h=4,
                eps=1e-6):
        shape = qkv.shape
        qo, ko, vo = triton_qk_norm_and_half_rope_forward(qkv,
                                                          q_norm_weight.data,
                                                          k_norm_weight.data,
                                                          freqs,
                                                          H=H,
                                                          h=h,
                                                          eps=eps,
                                                          interleaved=True,
                                                          transposed=True)

        ctx.save_for_backward(qkv, q_norm_weight.data, k_norm_weight.data,
                              freqs)
        ctx.H = H
        ctx.h = h
        ctx.eps = eps
        ctx.shape = shape
        return qo, ko, vo

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        qkv, q_norm_weight, k_norm_weight, freqs = ctx.saved_tensors

        dqkv, dqw, dkw = triton_qk_norm_and_half_rope_backward(grad_q,
                                                               grad_k,
                                                               grad_v,
                                                               qkv,
                                                               q_norm_weight,
                                                               k_norm_weight,
                                                               freqs,
                                                               eps=ctx.eps,
                                                               transposed=True,
                                                               interleaved=True)
        return dqkv, dqw, dkw, None, None, None, None


def qk_norm_half_rope(qkv: torch.Tensor,
                      q_norm_weight: torch.Tensor,
                      k_norm_weight: torch.Tensor,
                      freqs: torch.Tensor,
                      H: int = 32,
                      h: int = 4,
                      eps: float = 1e-6):
    """
    split qkv to q/k/v, apply qk norm and half rope to q/k, transpose q/k/v to flash-attention layout
    Args:
        qkv: QKV tensor with size of [S, B, dim], heads are interleaved
        q_norm_weight: rms norm weight for query
        k_norm_weight: rms norm weight for key
        freqs: Freqs tensor based on half dim.
        H: Number of attention heads.
        h: Number of key/value heads.
        eps: epsilon value for L2 normalization.

    Returns:
        - qo: shape [B, S, H, head_dim]
        - ko: shape [B, S, h, head_dim]
        - vo: shape [B, S, h, head_dim]
    """
    return QkNormHalfRopeFunction.apply(qkv,
                                        q_norm_weight,
                                        k_norm_weight,
                                        freqs,
                                        H,
                                        h,
                                        eps)