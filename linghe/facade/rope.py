# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import Optional
import torch

from linghe.utils.rope import triton_mla_rope_backward, triton_mla_rope_forward, triton_qk_norm_and_half_rope_forward, \
    triton_qk_norm_and_half_rope_backward, \
    triton_mla_rope_forward, \
    triton_mla_rope_backward


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


class MLARopeFunction(torch.autograd.Function):
    """"""
    @staticmethod
    def forward(ctx, q, kv, k_pos_emb, freqs, mscale, cu_seqlens_q, cu_seqlens_kv, cp_size, cp_rank):
        qo, ko, vo = triton_mla_rope_forward(q,
                                             kv,
                                            k_pos_emb,
                                            freqs,
                                            mscale=mscale,
                                            cu_seqlens_q=cu_seqlens_q,
                                            cu_seqlens_kv=cu_seqlens_kv,
                                            cp_size=cp_size,
                                            cp_rank=cp_rank)

        ctx.save_for_backward(freqs)
        ctx.mscale = mscale
        ctx.cp_size = cp_size 
        ctx.cp_rank = cp_rank
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_kv = cu_seqlens_kv
        return qo, ko, vo

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        freqs, = ctx.saved_tensors
        dq, dkv, dp = triton_mla_rope_backward(grad_q,
                                                  grad_k,
                                                  grad_v,
                                                  freqs,
                                                  mscale=ctx.mscale,
                                                  cu_seqlens_q=ctx.cu_seqlens_q,
                                                  cu_seqlens_kv=ctx.cu_seqlens_kv,
                                                  cp_size=ctx.cp_size,
                                                  cp_rank=ctx.cp_rank)
        return dq, dkv, dp, None, None, None, None, None, None


def mla_rope(q: torch.Tensor,
                      kv: torch.Tensor,
                      k_pos_emb: torch.Tensor,
                      freqs: torch.Tensor,
                      cu_seqlens_q: Optional[torch.Tensor] = None,
                      cu_seqlens_kv: Optional[torch.Tensor] = None,
                      mscale: float = 1.0,
                      cp_size: int = 1,
                      cp_rank: int = 0):
    """
    inplace apply rope to tail 64 dims, split kv and apply rope to k_pos_emb and copy to k
    Args:
        q: query tensor with size of [S, B, H, 128] (cu_seqlens is None) 
                or [N, H, 128] (cu_seqlens is not None)
        kv: kv tensor with size of [S, B, H, 256] (cu_seqlens is None) or 
            [N, H, 256] (cu_seqlens is not None)
        k_pos_emb: k pos emb with size of [S, B, 1, 64] (cu_seqlens is None) or 
            [N, 1, 64] (cu_seqlens is not None)
        freqs: Freqs tensor with size of [S, 64]
        cu_seqlens_q: cumulative query lengths tensor with size of [B+1]
        cu_seqlens_kv: cumulative kv lengths tensor with size of [B+1]
        mscale: mscale of rope
        cp_size: context-parallel size
        cp_rank: context-parallel rank
    Returns:
        - qo: shape [S, B, H, 192] or [N, H, 192]
        - ko: shape [S, B, H, 192] or [N, H, 192]
        - vo: shape [S, B, H, 128] or [N, H, 128]
    """
    return MLARopeFunction.apply(q,
                                kv,
                                k_pos_emb,
                                freqs,
                                mscale,
                                cu_seqlens_q,
                                cu_seqlens_kv,
                                cp_size,
                                cp_rank)