# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import Optional

import torch

from linghe.attn.mla import (
    triton_mla_forward,
    triton_mla_backward,
    triton_varlen_mla_forward,
    triton_varlen_mla_backward,
)


class MultiLatentAttention(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        padded_cu_seqlens: Optional[torch.Tensor] = None,
        max_q_length: Optional[int] = None,
        causal: bool = True,
        safe: bool = True,
        clip_value: Optional[float] = None,
    ):
        ctx.cu_seqlens = cu_seqlens
        ctx.padded_cu_seqlens = padded_cu_seqlens
        ctx.max_q_length = max_q_length
        ctx.causal = causal
        ctx.safe = safe
        ctx.clip_value = clip_value
        VARLEN = cu_seqlens is not None
        ctx.VARLEN = VARLEN
        if VARLEN:
            output, lse, max_logits = triton_varlen_mla_forward(
                q,
                k,
                v,
                cu_seqlens,
                padded_cu_seqlens=None,
                max_q_length=max_q_length,
                causal=causal,
                safe=safe,
                clip_value=clip_value,
            )
        else:
            output, lse, max_logits = triton_mla_forward(
                q, k, v, causal=causal, safe=safe, clip_value=clip_value
            )
        ctx.save_for_backward(q, k, v, output, lse, max_logits)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, output, lse, max_logits = ctx.saved_tensors
        if ctx.VARLEN:
            dq, dk, dv = triton_varlen_mla_backward(
                grad_output,
                output,
                q,
                k,
                v,
                lse,
                max_logits,
                ctx.cu_seqlens,
                ctx.max_q_length,
                padded_cu_seqlens=ctx.padded_cu_seqlens,
                causal=ctx.causal,
                safe=ctx.safe,
                clip_value=ctx.clip_value,
            )
        else:
            dq, dk, dv = triton_mla_backward(
                grad_output,
                output,
                q,
                k,
                v,
                lse,
                max_logits,
                causal=ctx.causal,
                safe=ctx.safe,
                clip_value=ctx.clip_value,
            )
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def multi_latend_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    padded_cu_seqlens: Optional[torch.Tensor] = None,
    max_q_length: Optional[int] = None,
    causal: bool = True,
    safe: bool = True,
    clip_value: float = 0.0,
):
    """
    inplace add y to x with mix precise
    Args:
        x: to be updated
        y: add to x
    Returns:
        updated x tensor
    """
    return MultiLatentAttention.apply(
        q, k, v, cu_seqlens, padded_cu_seqlens, max_q_length, causal, safe, clip_value
    )
