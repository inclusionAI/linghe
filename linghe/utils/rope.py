# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def half_rope_forward_kernel(q_ptr, k_ptr, freqs_ptr, qo_ptr, ko_ptr, B,
                             q_stride,
                             k_stride,
                             H: tl.constexpr,
                             h: tl.constexpr,
                             D: tl.constexpr,
                             d: tl.constexpr,
                             TRANSPOSED: tl.constexpr
                             ):
    pid = tl.program_id(0)
    L = tl.num_programs(0)

    freqs = tl.load(freqs_ptr + pid * D + tl.arange(0, D))
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)
    signs = tl.arange(0, 2).to(tl.float32) * 2 - 1

    
    for i in range(B):
        if TRANSPOSED:
            # [len, bs, q_head, head_dim]
            q = tl.load(
                q_ptr + pid * B * q_stride + i * q_stride + 2 * D * tl.arange(0, H)[
                                                                    :,
                                                                    None] + tl.arange(
                    0, D)[None, :])
        else:
            # [bs, len, q_head, head_dim]
            q = tl.load(
                q_ptr + pid * q_stride + i * L * q_stride + 2 * D * tl.arange(0, H)[
                                                                    :,
                                                                    None] + tl.arange(
                    0, D)[None, :])
        qr = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(q, (H, 2, d)), (0, 2, 1)),
                    dim=2) * signs, (0, 2, 1)), (H, D))
        q = q * cos + qr * sin
        if TRANSPOSED:
            tl.store(
                qo_ptr + pid * B * H * D * 2 + i * H * D * 2 + 2 * D * tl.arange(0,
                                                                                H)[
                                                                    :,
                                                                    None] + tl.arange(
                    0, D)[None, :], q)

            q = tl.load(
                q_ptr + pid * B * q_stride + i * q_stride + D + 2 * D * tl.arange(0,
                                                                                H)[
                                                                        :,
                                                                        None] + tl.arange(
                    0, D)[None, :])
            tl.store(
                qo_ptr + pid * B * H * D * 2 + i * H * D * 2 + D + 2 * D * tl.arange(
                    0, H)[:, None] + tl.arange(0, D)[None, :], q)
        else:
            tl.store(
                qo_ptr + pid * H * D * 2 + i * L * H * D * 2 + 2 * D * tl.arange(0,
                                                                                H)[
                                                                    :,
                                                                    None] + tl.arange(
                    0, D)[None, :], q)

            q = tl.load(
                q_ptr + pid * q_stride + i * L * q_stride + D + 2 * D * tl.arange(0,
                                                                                H)[
                                                                        :,
                                                                        None] + tl.arange(
                    0, D)[None, :])
            tl.store(
                qo_ptr + pid * H * D * 2 + i * L * H * D * 2 + D + 2 * D * tl.arange(
                    0, H)[:, None] + tl.arange(0, D)[None, :], q) 


    for i in range(B):
        if TRANSPOSED:
            k = tl.load(
                k_ptr + pid * B * k_stride + i * k_stride + 2 * D * tl.arange(0, h)[
                                                                    :,
                                                                    None] + tl.arange(
                    0, D)[None, :])
        else:
            k = tl.load(
                k_ptr + pid * k_stride + i * L * k_stride + 2 * D * tl.arange(0, h)[
                                                                    :,
                                                                    None] + tl.arange(
                    0, D)[None, :])
        kr = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(k, (h, 2, d)), (0, 2, 1)),
                    dim=2) * signs, (0, 2, 1)), (h, D))
        k = k * cos + kr * sin
        if TRANSPOSED:
            tl.store(
                ko_ptr + pid * B * h * D * 2 + i * h * D * 2 + 2 * D * tl.arange(0,
                                                                                h)[
                                                                    :,
                                                                    None] + tl.arange(
                    0, D)[None, :], k)

            k = tl.load(
                k_ptr + pid * B * k_stride + i * k_stride + D + 2 * D * tl.arange(0,
                                                                                h)[
                                                                        :,
                                                                        None] + tl.arange(
                    0, D)[None, :])
            tl.store(
                ko_ptr + pid * B * h * D * 2 + i * h * D * 2 + D + 2 * D * tl.arange(
                    0, h)[:, None] + tl.arange(0, D)[None, :], k)
        else:
            tl.store(
                ko_ptr + pid * h * D * 2 + i * L * h * D * 2 + 2 * D * tl.arange(0,
                                                                                h)[
                                                                    :,
                                                                    None] + tl.arange(
                    0, D)[None, :], k)

            k = tl.load(
                k_ptr + pid * k_stride + i * L * k_stride + D + 2 * D * tl.arange(0,
                                                                                h)[
                                                                        :,
                                                                        None] + tl.arange(
                    0, D)[None, :])
            tl.store(
                ko_ptr + pid * h * D * 2 + i * L * h * D * 2 + D + 2 * D * tl.arange(
                    0, h)[:, None] + tl.arange(0, D)[None, :], k)


def triton_half_rope_forward(q, k, freqs, transposed=True):
    """
    apply norm to qk, then apply half rope to qk
    Args:
        q: query tensor, [len, bs, q_head, head_dim]
        k: key tensor, [len, bs, kv_head, head_dim]
        freqs: rope freqs

    Returns:
        - qo: query output
        - ko: key output
    """
    if transposed:
        L, B, H, D = q.shape
    else:
        B, L, H, D = q.shape
    h = k.shape[2]
    assert freqs.shape[1] == D // 2
    num_stages = 3
    num_warps = 2

    q_stride = q.stride(1)
    k_stride = k.stride(1)
    if transposed:
        qo = torch.empty((L, B, H, D), dtype=q.dtype, device=q.device)
        ko = torch.empty((L, B, h, D), dtype=q.dtype, device=q.device)
    else:
        qo = torch.empty((B, L, H, D), dtype=q.dtype, device=q.device)
        ko = torch.empty((B, L, h, D), dtype=q.dtype, device=q.device)
    grid = (L,)
    half_rope_forward_kernel[grid](
        q, k,
        freqs,
        qo, ko,
        B,
        q_stride,
        k_stride,
        H,
        h,
        D // 2,
        D // 4,
        transposed,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return qo, ko


@triton.jit
def half_rope_backward_kernel(q_ptr, k_ptr, freqs_ptr,
                              B,
                              H: tl.constexpr,
                              h: tl.constexpr,
                              D: tl.constexpr,
                              d: tl.constexpr,
                              TRANSPOSED: tl.constexpr
                              ):
    pid = tl.program_id(0)
    L = tl.num_programs(0)

    freqs = tl.load(freqs_ptr + pid * D + tl.arange(0, D))
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)
    signs = -tl.arange(0, 2).to(tl.float32) * 2 + 1

    # [len, bs, q_head, head_dim]
    for i in range(B):
        if TRANSPOSED:
            q = tl.load(
                q_ptr + pid * B * H * D * 2 + i * H * D * 2 + 2 * D * tl.arange(0,
                                                                                H)[
                                                                    :,
                                                                    None] + tl.arange(
                    0, D)[None, :])
        else:
            q = tl.load(
                q_ptr + pid * H * D * 2 + i * L * H * D * 2 + 2 * D * tl.arange(0,
                                                                                H)[
                                                                    :,
                                                                    None] + tl.arange(
                    0, D)[None, :])
        qr = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(q, (H, 2, d)), (0, 2, 1)),
                    dim=2) * signs, (0, 2, 1)), (H, D))
        q = q * cos + qr * sin
        if TRANSPOSED:
            tl.store(
                q_ptr + pid * B * H * D * 2 + i * H * D * 2 + 2 * D * tl.arange(0,
                                                                                H)[
                                                                    :,
                                                                    None] + tl.arange(
                    0, D)[None, :], q)
        else:
            tl.store(
                q_ptr + pid * H * D * 2 + i * L * H * D * 2 + 2 * D * tl.arange(0,
                                                                                H)[
                                                                    :,
                                                                    None] + tl.arange(
                    0, D)[None, :], q)  

    for i in range(B):
        if TRANSPOSED:
            k = tl.load(
                k_ptr + pid * B * h * D * 2 + i * h * D * 2 + 2 * D * tl.arange(0,
                                                                                h)[
                                                                    :,
                                                                    None] + tl.arange(
                    0, D)[None, :])
        else:
            k = tl.load(
                k_ptr + pid * h * D * 2 + i * L * h * D * 2 + 2 * D * tl.arange(0,
                                                                                h)[
                                                                    :,
                                                                    None] + tl.arange(
                    0, D)[None, :])
        kr = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(k, (h, 2, d)), (0, 2, 1)),
                    dim=2) * signs, (0, 2, 1)), (h, D))
        k = k * cos + kr * sin
        if TRANSPOSED:
            tl.store(
                k_ptr + pid * B * h * D * 2 + i * h * D * 2 + 2 * D * tl.arange(0,
                                                                                h)[
                                                                    :,
                                                                    None] + tl.arange(
                    0, D)[None, :], k)
        else:
            tl.store(
                k_ptr + pid * h * D * 2 + i * L * h * D * 2 + 2 * D * tl.arange(0,
                                                                                h)[
                                                                    :,
                                                                    None] + tl.arange(
                    0, D)[None, :], k)


def triton_half_rope_backward(q_grad, k_grad, freqs, inplace=False, transposed=True):
    assert inplace
    if transposed:
        L, B, H, D = q_grad.shape
    else:
        B, L, H, D = q_grad.shape
    h = k_grad.shape[2]
    assert freqs.shape[1] == D // 2
    num_stages = 3
    num_warps = 2

    grid = (L,)
    half_rope_backward_kernel[grid](
        q_grad, k_grad,
        freqs,
        B,
        H,
        h,
        D // 2,
        D // 4,
        transposed,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return q_grad, k_grad


@triton.jit
def qk_norm_and_half_rope_forward_kernel(qkv_ptr,
                                         q_norm_weight_ptr, k_norm_weight_ptr,
                                         freqs_ptr,
                                         qo_ptr, ko_ptr, vo_ptr,
                                         B,
                                         stride,
                                         eps,
                                         H: tl.constexpr,
                                         h: tl.constexpr,
                                         D: tl.constexpr,
                                         d: tl.constexpr,
                                         INTERLEAVED: tl.constexpr,
                                         TRANSPOSED: tl.constexpr):
    pid = tl.program_id(0)
    L = tl.num_programs(0)
    DD = D * 2

    freqs = tl.load(freqs_ptr + pid * D + tl.arange(0, D))
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)
    signs = tl.arange(0, 2).to(tl.float32) * 2 - 1

    q_weight_0 = tl.load(q_norm_weight_ptr + tl.arange(0, D))
    q_weight_1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D))
    q_ptr = qkv_ptr
    w = H // h

    # [len, bs, q_head, head_dim] -> [bs, len, q_head, head_dim]
    if INTERLEAVED:
        row_offs = tl.arange(0, H) + tl.arange(0, H) // w * 2
    else:
        row_offs = tl.arange(0, H)

    for i in range(B):
        if TRANSPOSED:
            q0 = tl.load(q_ptr + pid * B * stride + i * stride + DD * row_offs[:,
                                                                    None] + tl.arange(
                0, D)[None, :])
            q1 = tl.load(q_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,
                                                                None] + tl.arange(
                    0, D)[None, :])
        else:
            q0 = tl.load(q_ptr + i * L * stride + pid * stride + DD * row_offs[:,
                                                                    None] + tl.arange(
                0, D)[None, :])
            q1 = tl.load(q_ptr + i * L * stride + pid * stride + D + DD * row_offs[:,
                                                                None] + tl.arange(
                    0, D)[None, :])

        rms = 1 / tl.sqrt((tl.sum(q0 * q0, 1) + tl.sum(q1 * q1, 1)) / DD + eps)
        q1 *= rms[:, None]
        q1 *= q_weight_1
        tl.store(
            qo_ptr + pid * H * DD + i * L * H * DD + D + DD * tl.arange(0, H)[:,
                                                              None] + tl.arange(
                0, D)[None, :], q1)

        q0 *= rms[:, None]
        q0 *= q_weight_0
        qr = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(q0, (H, 2, d)), (0, 2, 1)),
                    dim=2) * signs, (0, 2, 1)), (H, D))
        q0 = q0 * cos + qr * sin
        tl.store(
            qo_ptr + pid * H * DD + i * L * H * DD + DD * tl.arange(0, H)[:,
                                                          None] + tl.arange(0,
                                                                            D)[
                                                                  None, :], q0)

    k_weight_0 = tl.load(k_norm_weight_ptr + tl.arange(0, D))
    k_weight_1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D))
    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        k_ptr = qkv_ptr + DD * w
    else:
        row_offs = tl.arange(0, h)
        k_ptr = qkv_ptr + DD * H
    for i in range(B):
        if TRANSPOSED:
            k0 = tl.load(k_ptr + pid * B * stride + i * stride + DD * row_offs[:,
                                                                    None] + tl.arange(
                0, D)[None, :])
            k1 = tl.load(
                k_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,
                                                                None] + tl.arange(
                    0, D)[None, :])
        else:
            k0 = tl.load(k_ptr + i * L * stride + pid * stride + DD * row_offs[:,
                                                                    None] + tl.arange(
                0, D)[None, :])
            k1 = tl.load(
                k_ptr + i * L * stride + pid * stride + D + DD * row_offs[:,
                                                                None] + tl.arange(
                    0, D)[None, :])
        rms = 1 / tl.sqrt((tl.sum(k0 * k0, 1) + tl.sum(k1 * k1, 1)) / DD + eps)
        k1 *= rms[:, None]
        k1 *= k_weight_1
        tl.store(
            ko_ptr + pid * h * DD + i * L * h * DD + D + DD * tl.arange(0, h)[:,
                                                              None] + tl.arange(
                0, D)[None, :], k1)

        k0 *= rms[:, None]
        k0 *= k_weight_0
        kr = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(k0, (h, 2, d)), (0, 2, 1)),
                    dim=2) * signs, (0, 2, 1)), (h, D))
        k0 = k0 * cos + kr * sin
        tl.store(
            ko_ptr + pid * h * DD + i * L * h * DD + DD * tl.arange(0, h)[:,
                                                          None] + tl.arange(0,
                                                                            D)[
                                                                  None, :], k0)

    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        v_ptr = qkv_ptr + DD * w + DD
    else:
        row_offs = tl.arange(0, h)
        v_ptr = qkv_ptr + DD * H + DD * h
    for i in range(B):
        if TRANSPOSED:
            v0 = tl.load(v_ptr + pid * B * stride + i * stride + DD * row_offs[:,
                                                                    None] + tl.arange(
                0, D)[None, :])
        else:
            v0 = tl.load(v_ptr + i * L * stride + pid * stride + DD * row_offs[:,
                                                                    None] + tl.arange(
                0, D)[None, :])
        tl.store(
            vo_ptr + pid * h * DD + i * L * h * DD + DD * tl.arange(0, h)[:,
                                                        None] + tl.arange(0,
                                                                            D)[
                                                                None, :], v0)
            
        if TRANSPOSED:
            v1 = tl.load(
                v_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,
                                                                None] + tl.arange(
                    0, D)[None, :])
        else:
            v1 = tl.load(
                v_ptr + i * L * stride + pid * stride + D + DD * row_offs[:,
                                                                None] + tl.arange(
                    0, D)[None, :])
        tl.store(
            vo_ptr + pid * h * DD + i * L * h * DD + D + DD * tl.arange(0, h)[:,
                                                              None] + tl.arange(
                0, D)[None, :], v1)


def triton_qk_norm_and_half_rope_forward(qkv, q_norm_weight, k_norm_weight,
                                         freqs, H=32, h=4, eps=1e-6,
                                         interleaved=True, transposed=True):

    """
    split qkv to q/k/v, apply qk norm and half rope to q/k,
        transpose q/k/v to flash-attention layout
    Args:
        qkv: QKV tensor with size of [S, B, dim], heads are interleaved
        q_norm_weight: rms norm weight for query
        k_norm_weight: rms norm weight for key
        freqs: Freqs tensor based on half dim.
        H: Number of attention heads.
        h: Number of key/value heads.
        eps: epsilon value for L2 normalization.
        interleaved: whether head of qkv is interleaved,
            interleaved: [q...qkvq...qkv]
            non-interleaved: [q...qk...kv...v]
        transposed: whether qkv is tranposed
            transposed: [S, B, dim]
            non-transposed: [B, S, dim]
    Returns:
        - qo: shape [B, S, H, head_dim]
        - ko: shape [B, S, h, head_dim]
        - vo: shape [B, S, h, head_dim]
    """
    if transposed:
        L, B, Dim = qkv.shape
    else:
        B, L, Dim = qkv.shape
    stride = qkv.stride(1)  # qkv may be a slice of a tensor
    D = Dim // (H + 2 * h)
    dtype = qkv.dtype
    device = qkv.device
    qo = torch.empty((B, L, H, D), dtype=dtype, device=device)
    ko = torch.empty((B, L, h, D), dtype=dtype, device=device)
    vo = torch.empty((B, L, h, D), dtype=dtype, device=device)

    num_stages = 5
    num_warps = 2
    grid = (L,)
    qk_norm_and_half_rope_forward_kernel[grid](
        qkv,
        q_norm_weight, k_norm_weight,
        freqs,
        qo, ko, vo,
        B,
        stride,
        eps,
        H,
        h,
        D // 2,
        D // 4,
        interleaved,
        transposed,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return qo, ko, vo


@triton.jit
def qk_norm_and_half_rope_backward_kernel(gq_ptr, gk_ptr, gv_ptr,
                                          qkv_ptr,
                                          q_norm_weight_ptr, k_norm_weight_ptr,
                                          freqs_ptr,
                                          dqkv_ptr,
                                          dqw_ptr, dkw_ptr,
                                          B,
                                          stride,
                                          grad_stride,
                                          eps,
                                          H: tl.constexpr,
                                          h: tl.constexpr,
                                          D: tl.constexpr,
                                          d: tl.constexpr,
                                          INTERLEAVED: tl.constexpr,
                                          TRANSPOSED: tl.constexpr
                                          ):
    pid = tl.program_id(0)
    L = tl.num_programs(0)
    DD = 2 * D
    w = H // h

    freqs = tl.load(freqs_ptr + pid * D + tl.arange(0, D))
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)
    signs = -tl.arange(0, 2).to(tl.float32) * 2 + 1

    q_w0 = tl.load(q_norm_weight_ptr + tl.arange(0, D))
    q_w1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D))

    dqw_0 = tl.zeros((D,), dtype=tl.float32)
    dqw_1 = tl.zeros((D,), dtype=tl.float32)
    q_ptr = qkv_ptr
    dq_ptr = dqkv_ptr
    # [bs, len, q_head, head_dim] -> [len, bs, q_head, head_dim]
    if INTERLEAVED:
        row_offs = tl.arange(0, H) + tl.arange(0, H) // w * 2
    else:
        row_offs = tl.arange(0, H)

    for i in range(B):
        gq_0 = tl.load(
            gq_ptr + i * L * H * DD + pid * H * DD + DD * tl.arange(0, H)[:,
                                                          None] + tl.arange(0,
                                                                            D)[
                                                                  None, :])
        gq_1 = tl.load(
            gq_ptr + i * L * H * DD + pid * H * DD + D + DD * tl.arange(0, H)[:,
                                                              None] + tl.arange(
                0, D)[None, :])

        gq_r = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(gq_0, (H, 2, d)), (0, 2, 1)),
                    dim=2) * signs, (0, 2, 1)), (H, D))
        gq_0 = gq_0 * cos + gq_r * sin

        if TRANSPOSED:
            q0 = tl.load(q_ptr + pid * B * stride + i * stride + DD * row_offs[:,
                                                                    None] + tl.arange(
                0, D)[None, :])
            q1 = tl.load(
                q_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,
                                                                None] + tl.arange(
                    0, D)[None, :])
        else:
            q0 = tl.load(q_ptr + pid * stride + i * L * stride + DD * row_offs[:,
                                                                    None] + tl.arange(
                0, D)[None, :])
            q1 = tl.load(
                q_ptr + pid * stride + i * L * stride + D + DD * row_offs[:,
                                                                None] + tl.arange(
                    0, D)[None, :])

        rms = tl.sqrt((tl.sum(q0 * q0, 1) + tl.sum(q1 * q1, 1)) / DD + eps)
        r = (1 / rms)[:, None]

        dqw_0 += tl.sum(q0 * gq_0 * r, 0)
        dqw_1 += tl.sum(q1 * gq_1 * r, 0)

        s = tl.sum(q0 * gq_0 * q_w0, 1) + tl.sum(q1 * gq_1 * q_w1, 1)

        dq_0 = r * gq_0 * q_w0 - r * r * r / DD * q0 * s[:, None]
        dq_1 = r * gq_1 * q_w1 - r * r * r / DD * q1 * s[:, None]

        if TRANSPOSED:
            tl.store(dq_ptr + pid * B * grad_stride + i * grad_stride + DD * row_offs[:,
                                                                None] + tl.arange(
                0, D)[None, :], dq_0)
            tl.store(dq_ptr + pid * B * grad_stride + i * grad_stride + D + DD * row_offs[:,
                                                                    None] + tl.arange(
                0, D)[None, :], dq_1)
        else:
            tl.store(dq_ptr + pid * grad_stride + i * L * grad_stride + DD * row_offs[:,
                                                                None] + tl.arange(
                0, D)[None, :], dq_0)
            tl.store(dq_ptr + pid * grad_stride + i * L * grad_stride + D + DD * row_offs[:,
                                                                    None] + tl.arange(
                0, D)[None, :], dq_1)

    tl.store(dqw_ptr + pid * D * 2 + tl.arange(0, D), dqw_0)
    tl.store(dqw_ptr + pid * D * 2 + D + tl.arange(0, D), dqw_1)

    k_w0 = tl.load(k_norm_weight_ptr + tl.arange(0, D))
    k_w1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D))

    dkw_0 = tl.zeros((D,), dtype=tl.float32)
    dkw_1 = tl.zeros((D,), dtype=tl.float32)
    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        k_ptr = qkv_ptr + DD * w
        dk_ptr = dqkv_ptr + DD * w
    else:
        row_offs = tl.arange(0, h)
        k_ptr = qkv_ptr + DD * H
        dk_ptr = dqkv_ptr + DD * H
    # [bs, len, k_head, head_dim] -> [len, bs, k_head, head_dim]
    for i in range(B):
        gk_0 = tl.load(
            gk_ptr + i * L * h * DD + pid * h * DD + DD * tl.arange(0, h)[:,
                                                          None] + tl.arange(0,
                                                                            D)[
                                                                  None, :])
        gk_1 = tl.load(
            gk_ptr + i * L * h * DD + pid * h * DD + D + DD * tl.arange(0, h)[:,
                                                              None] + tl.arange(
                0, D)[None, :])

        gk_r = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(gk_0, (h, 2, d)), (0, 2, 1)),
                    dim=2) * signs, (0, 2, 1)), (h, D))
        gk_0 = gk_0 * cos + gk_r * sin

        if TRANSPOSED:
            k0 = tl.load(k_ptr + pid * B * stride + i * stride + DD * row_offs[:,
                                                                    None] + tl.arange(
                0, D)[None, :])
            k1 = tl.load(
                k_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,
                                                                None] + tl.arange(
                    0, D)[None, :])
        else:
            k0 = tl.load(k_ptr + pid * stride + i * L * stride + DD * row_offs[:,
                                                                    None] + tl.arange(
                0, D)[None, :])
            k1 = tl.load(
                k_ptr + pid * stride + i * L * stride + D + DD * row_offs[:,
                                                                None] + tl.arange(
                    0, D)[None, :])

        rms = tl.sqrt((tl.sum(k0 * k0, 1) + tl.sum(k1 * k1, 1)) / DD + eps)
        r = (1 / rms)[:, None]

        dkw_0 += tl.sum(k0 * gk_0 * r, 0)
        dkw_1 += tl.sum(k1 * gk_1 * r, 0)

        s = tl.sum(k0 * gk_0 * k_w0, 1) + tl.sum(k1 * gk_1 * k_w1, 1)

        dk_0 = r * gk_0 * k_w0 - r * r * r / DD * k0 * s[:, None]
        dk_1 = r * gk_1 * k_w1 - r * r * r / DD * k1 * s[:, None]

        if TRANSPOSED:
            tl.store(dk_ptr + pid * B * grad_stride + i * grad_stride + DD * row_offs[:,
                                                                None] + tl.arange(
                0, D)[None, :], dk_0)
            tl.store(dk_ptr + pid * B * grad_stride + i * grad_stride + D + DD * row_offs[:,
                                                                    None] + tl.arange(
                0, D)[None, :], dk_1)
        else:
            tl.store(dk_ptr + pid * grad_stride + i * L * grad_stride + DD * row_offs[:,
                                                                None] + tl.arange(
                0, D)[None, :], dk_0)
            tl.store(dk_ptr + pid * grad_stride + i * L * grad_stride + D + DD * row_offs[:,
                                                                    None] + tl.arange(
                0, D)[None, :], dk_1)
    tl.store(dkw_ptr + pid * D * 2 + tl.arange(0, D), dkw_0)
    tl.store(dkw_ptr + pid * D * 2 + D + tl.arange(0, D), dkw_1)

    # [bs, len, k_head, head_dim] -> [len, bs, k_head + 2 * kv_head, head_dim]
    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        dv_ptr = dqkv_ptr + DD * w + DD
    else:
        row_offs = tl.arange(0, h)
        dv_ptr = dqkv_ptr + DD * H + DD * h
    for i in range(B):
        v0 = tl.load(
            gv_ptr + i * L * h * DD + pid * h * DD + DD * tl.arange(0, h)[:,
                                                        None] + tl.arange(0,
                                                                            D)[
                                                                None, :])

        if TRANSPOSED:                                        
            tl.store(dv_ptr + pid * B * grad_stride + i * grad_stride + DD * row_offs[:,
                                                                None] + tl.arange(
                0, D)[None, :], v0)
        else:
            tl.store(dv_ptr + pid * grad_stride + i * L * grad_stride + DD * row_offs[:,
                                                                None] + tl.arange(
                0, D)[None, :], v0)   


        v1 = tl.load(
            gv_ptr + i * L * h * DD + pid * h * DD + D + DD * tl.arange(0, h)[:,
                                                              None] + tl.arange(
                0, D)[None, :])

        if TRANSPOSED:
            tl.store(dv_ptr + pid * B * grad_stride + i * grad_stride + D + DD * row_offs[:,
                                                                    None] + tl.arange(
                0, D)[None, :], v1)
        else:
            tl.store(dv_ptr + pid * grad_stride + i * L * grad_stride + D + DD * row_offs[:,
                                                                    None] + tl.arange(
                0, D)[None, :], v1)


def triton_qk_norm_and_half_rope_backward(gq, gk, gv, qkv, q_norm_weight,
                                          k_norm_weight, freqs, eps=1e-6,
                                          interleaved=True, transposed=True):
    """
    backward kernel of triton_qk_norm_and_half_rope_forward
    Args:
        gq: gradient of qo, [len, bs, q_head, head_dim]
        gk: gradient of ko, [len, bs, q_head, head_dim]
        gv: gradient of vo, [len, bs, q_head, head_dim]
        qkv: input qkv
        q_norm_weight: rms norm weight for query
        k_norm_weight: rms norm weight for key
        freqs: Freqs tensor based on half dim.
        eps: epsilon value for L2 normalization.
        interleaved: whether head of qkv is interleaved,
            interleaved: [q...qkvq...qkv]
            non-interleaved: [q...qk...kv...v]
        transposed: whether qkv is tranposed
            transposed: [S, B, dim]
            non-transposed: [B, S, dim]

    Returns:
        - dqkv: gradient of qkv
        - dqw: gradient of q_norm_weight
        - dkw: gradient of k_norm_weight
    """
    B, L, H, D = gq.shape
    stride = qkv.stride(1)
    h = gk.shape[2]
    num_stages = 5
    num_warps = 1

    dtype = gq.dtype
    device = gq.device
    if transposed:
        dqkv = torch.empty((L, B, (H + 2 * h) * D), dtype=dtype, device=device)
    else:
        dqkv = torch.empty((B, L, (H + 2 * h) * D), dtype=dtype, device=device)
    grad_stride = dqkv.stride(1)  # for potential fused kernel

    tmp_dqw = torch.empty((L, D), dtype=torch.float32, device=device)
    tmp_dkw = torch.empty((L, D), dtype=torch.float32, device=device)

    grid = (L,)
    qk_norm_and_half_rope_backward_kernel[grid](
        gq, gk, gv,
        qkv,
        q_norm_weight, k_norm_weight,
        freqs,
        dqkv,
        tmp_dqw, tmp_dkw,
        B,
        stride,
        grad_stride,
        eps,
        H,
        h,
        D // 2,
        D // 4,
        interleaved,
        transposed,
        num_stages=num_stages,
        num_warps=num_warps
    )
    dqw = tmp_dqw.sum(0).to(dtype)
    dkw = tmp_dkw.sum(0).to(dtype)
    return dqkv, dqw, dkw
