# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def half_rope_forward_kernel(
    q_ptr,
    k_ptr,
    freqs_ptr,
    qo_ptr,
    ko_ptr,
    B,
    q_stride,
    k_stride,
    H: tl.constexpr,
    h: tl.constexpr,
    D: tl.constexpr,
    d: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    pid = tl.program_id(0)
    L = tl.num_programs(0)

    freqs = tl.load(freqs_ptr + pid * D + tl.arange(0, D)).to(tl.float32)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)
    signs = tl.arange(0, 2).to(tl.float32) * 2 - 1

    for i in range(B):
        if TRANSPOSED:
            # [len, bs, q_head, head_dim]
            q = tl.load(
                q_ptr
                + pid * B * q_stride
                + i * q_stride
                + 2 * D * tl.arange(0, H)[:, None]
                + tl.arange(0, D)[None, :]
            )
        else:
            # [bs, len, q_head, head_dim]
            q = tl.load(
                q_ptr
                + pid * q_stride
                + i * L * q_stride
                + 2 * D * tl.arange(0, H)[:, None]
                + tl.arange(0, D)[None, :]
            )
        qr = tl.reshape(
            tl.permute(
                tl.flip(tl.permute(tl.reshape(q, (H, 2, d)), (0, 2, 1)), dim=2) * signs,
                (0, 2, 1),
            ),
            (H, D),
        )
        q = q * cos + qr * sin
        if TRANSPOSED:
            tl.store(
                qo_ptr
                + pid * B * H * D * 2
                + i * H * D * 2
                + 2 * D * tl.arange(0, H)[:, None]
                + tl.arange(0, D)[None, :],
                q,
            )

            q = tl.load(
                q_ptr
                + pid * B * q_stride
                + i * q_stride
                + D
                + 2 * D * tl.arange(0, H)[:, None]
                + tl.arange(0, D)[None, :]
            )
            tl.store(
                qo_ptr
                + pid * B * H * D * 2
                + i * H * D * 2
                + D
                + 2 * D * tl.arange(0, H)[:, None]
                + tl.arange(0, D)[None, :],
                q,
            )
        else:
            tl.store(
                qo_ptr
                + pid * H * D * 2
                + i * L * H * D * 2
                + 2 * D * tl.arange(0, H)[:, None]
                + tl.arange(0, D)[None, :],
                q,
            )

            q = tl.load(
                q_ptr
                + pid * q_stride
                + i * L * q_stride
                + D
                + 2 * D * tl.arange(0, H)[:, None]
                + tl.arange(0, D)[None, :]
            )
            tl.store(
                qo_ptr
                + pid * H * D * 2
                + i * L * H * D * 2
                + D
                + 2 * D * tl.arange(0, H)[:, None]
                + tl.arange(0, D)[None, :],
                q,
            )

    for i in range(B):
        if TRANSPOSED:
            k = tl.load(
                k_ptr
                + pid * B * k_stride
                + i * k_stride
                + 2 * D * tl.arange(0, h)[:, None]
                + tl.arange(0, D)[None, :]
            )
        else:
            k = tl.load(
                k_ptr
                + pid * k_stride
                + i * L * k_stride
                + 2 * D * tl.arange(0, h)[:, None]
                + tl.arange(0, D)[None, :]
            )
        kr = tl.reshape(
            tl.permute(
                tl.flip(tl.permute(tl.reshape(k, (h, 2, d)), (0, 2, 1)), dim=2) * signs,
                (0, 2, 1),
            ),
            (h, D),
        )
        k = k * cos + kr * sin
        if TRANSPOSED:
            tl.store(
                ko_ptr
                + pid * B * h * D * 2
                + i * h * D * 2
                + 2 * D * tl.arange(0, h)[:, None]
                + tl.arange(0, D)[None, :],
                k,
            )

            k = tl.load(
                k_ptr
                + pid * B * k_stride
                + i * k_stride
                + D
                + 2 * D * tl.arange(0, h)[:, None]
                + tl.arange(0, D)[None, :]
            )
            tl.store(
                ko_ptr
                + pid * B * h * D * 2
                + i * h * D * 2
                + D
                + 2 * D * tl.arange(0, h)[:, None]
                + tl.arange(0, D)[None, :],
                k,
            )
        else:
            tl.store(
                ko_ptr
                + pid * h * D * 2
                + i * L * h * D * 2
                + 2 * D * tl.arange(0, h)[:, None]
                + tl.arange(0, D)[None, :],
                k,
            )

            k = tl.load(
                k_ptr
                + pid * k_stride
                + i * L * k_stride
                + D
                + 2 * D * tl.arange(0, h)[:, None]
                + tl.arange(0, D)[None, :]
            )
            tl.store(
                ko_ptr
                + pid * h * D * 2
                + i * L * h * D * 2
                + D
                + 2 * D * tl.arange(0, h)[:, None]
                + tl.arange(0, D)[None, :],
                k,
            )


def triton_half_rope_forward(q, k, freqs, transposed=True):
    """
    apply half rope to qk
    Args:
        q: query tensor, [len, bs, q_head, head_dim]
        k: key tensor, [len, bs, kv_head, head_dim]
        freqs: rope freqs

    Returns:
        - qo: query output
        - ko: key output
    """
    assert q.is_contiguous() and k.is_contiguous() and freqs.is_contiguous()
    if transposed:
        L, B, H, D = q.shape
    else:
        B, L, H, D = q.shape
    h = k.shape[2]
    assert freqs.shape[1] == D // 2
    assert triton.next_power_of_2(H) == H
    assert triton.next_power_of_2(D) == D
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
        q,
        k,
        freqs,
        qo,
        ko,
        B,
        q_stride,
        k_stride,
        H,
        h,
        D // 2,
        D // 4,
        transposed,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return qo, ko


@triton.jit
def half_rope_backward_kernel(
    q_ptr,
    k_ptr,
    freqs_ptr,
    B,
    H: tl.constexpr,
    h: tl.constexpr,
    D: tl.constexpr,
    d: tl.constexpr,
    TRANSPOSED: tl.constexpr,
):
    pid = tl.program_id(0)
    L = tl.num_programs(0)

    freqs = tl.load(freqs_ptr + pid * D + tl.arange(0, D)).to(tl.float32)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)
    signs = -tl.arange(0, 2).to(tl.float32) * 2 + 1

    # [len, bs, q_head, head_dim]
    for i in range(B):
        if TRANSPOSED:
            q = tl.load(
                q_ptr
                + pid * B * H * D * 2
                + i * H * D * 2
                + 2 * D * tl.arange(0, H)[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
        else:
            q = tl.load(
                q_ptr
                + pid * H * D * 2
                + i * L * H * D * 2
                + 2 * D * tl.arange(0, H)[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
        qr = tl.reshape(
            tl.permute(
                tl.flip(tl.permute(tl.reshape(q, (H, 2, d)), (0, 2, 1)), dim=2) * signs,
                (0, 2, 1),
            ),
            (H, D),
        )
        qo = (q * cos + qr * sin).to(q_ptr.dtype.element_ty)
        if TRANSPOSED:
            tl.store(
                q_ptr
                + pid * B * H * D * 2
                + i * H * D * 2
                + 2 * D * tl.arange(0, H)[:, None]
                + tl.arange(0, D)[None, :],
                qo,
            )
        else:
            tl.store(
                q_ptr
                + pid * H * D * 2
                + i * L * H * D * 2
                + 2 * D * tl.arange(0, H)[:, None]
                + tl.arange(0, D)[None, :],
                qo,
            )

    for i in range(B):
        if TRANSPOSED:
            k = tl.load(
                k_ptr
                + pid * B * h * D * 2
                + i * h * D * 2
                + 2 * D * tl.arange(0, h)[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
        else:
            k = tl.load(
                k_ptr
                + pid * h * D * 2
                + i * L * h * D * 2
                + 2 * D * tl.arange(0, h)[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
        kr = tl.reshape(
            tl.permute(
                tl.flip(tl.permute(tl.reshape(k, (h, 2, d)), (0, 2, 1)), dim=2) * signs,
                (0, 2, 1),
            ),
            (h, D),
        )
        ko = (k * cos + kr * sin).to(k_ptr.dtype.element_ty)
        if TRANSPOSED:
            tl.store(
                k_ptr
                + pid * B * h * D * 2
                + i * h * D * 2
                + 2 * D * tl.arange(0, h)[:, None]
                + tl.arange(0, D)[None, :],
                ko,
            )
        else:
            tl.store(
                k_ptr
                + pid * h * D * 2
                + i * L * h * D * 2
                + 2 * D * tl.arange(0, h)[:, None]
                + tl.arange(0, D)[None, :],
                ko,
            )


def triton_half_rope_backward(q_grad, k_grad, freqs, inplace=False, transposed=True):
    assert q_grad.is_contiguous() and k_grad.is_contiguous()
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
        q_grad,
        k_grad,
        freqs,
        B,
        H,
        h,
        D // 2,
        D // 4,
        transposed,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    return q_grad, k_grad


@triton.jit
def qk_norm_and_half_rope_forward_kernel(
    qkv_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    freqs_ptr,
    qo_ptr,
    ko_ptr,
    vo_ptr,
    B,
    stride,
    eps,
    H: tl.constexpr,
    h: tl.constexpr,
    D: tl.constexpr,
    d: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    SILU: tl.constexpr,
):
    pid = tl.program_id(0)
    L = tl.num_programs(0)
    DD = D * 2

    freqs = tl.load(freqs_ptr + pid * D + tl.arange(0, D)).to(tl.float32)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)
    signs = tl.arange(0, 2).to(tl.float32) * 2 - 1

    q_weight_0 = tl.load(q_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    q_weight_1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)
    q_ptr = qkv_ptr
    w = H // h

    # [len, bs, q_head, head_dim] -> [bs, len, q_head, head_dim]
    if INTERLEAVED:
        row_offs = tl.arange(0, H) + tl.arange(0, H) // w * 2
    else:
        row_offs = tl.arange(0, H)

    for i in range(B):
        if TRANSPOSED:
            q0 = tl.load(
                q_ptr
                + pid * B * stride
                + i * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
            q1 = tl.load(
                q_ptr
                + pid * B * stride
                + i * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
        else:
            q0 = tl.load(
                q_ptr
                + i * L * stride
                + pid * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
            q1 = tl.load(
                q_ptr
                + i * L * stride
                + pid * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
        if SILU:
            q0 = q0 * tl.sigmoid(q0)
            q1 = q1 * tl.sigmoid(q1)
        rms = tl.rsqrt((tl.sum(q0 * q0, 1) + tl.sum(q1 * q1, 1)) / DD + eps)
        q1 *= rms[:, None]
        q1 *= q_weight_1
        tl.store(
            qo_ptr
            + pid * H * DD
            + i * L * H * DD
            + D
            + DD * tl.arange(0, H)[:, None]
            + tl.arange(0, D)[None, :],
            q1,
        )

        q0 *= rms[:, None]
        q0 *= q_weight_0
        qr = tl.reshape(
            tl.permute(
                tl.flip(tl.permute(tl.reshape(q0, (H, 2, d)), (0, 2, 1)), dim=2)
                * signs,
                (0, 2, 1),
            ),
            (H, D),
        )
        q0 = q0 * cos + qr * sin
        tl.store(
            qo_ptr
            + pid * H * DD
            + i * L * H * DD
            + DD * tl.arange(0, H)[:, None]
            + tl.arange(0, D)[None, :],
            q0,
        )

    k_weight_0 = tl.load(k_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    k_weight_1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)
    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        k_ptr = qkv_ptr + DD * w
    else:
        row_offs = tl.arange(0, h)
        k_ptr = qkv_ptr + DD * H
    for i in range(B):
        if TRANSPOSED:
            k0 = tl.load(
                k_ptr
                + pid * B * stride
                + i * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
            k1 = tl.load(
                k_ptr
                + pid * B * stride
                + i * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
        else:
            k0 = tl.load(
                k_ptr
                + i * L * stride
                + pid * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
            k1 = tl.load(
                k_ptr
                + i * L * stride
                + pid * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
        if SILU:
            k0 = k0 * tl.sigmoid(k0)
            k1 = k1 * tl.sigmoid(k1)
        rms = tl.rsqrt((tl.sum(k0 * k0, 1) + tl.sum(k1 * k1, 1)) / DD + eps)
        k1 *= rms[:, None]
        k1 *= k_weight_1
        tl.store(
            ko_ptr
            + pid * h * DD
            + i * L * h * DD
            + D
            + DD * tl.arange(0, h)[:, None]
            + tl.arange(0, D)[None, :],
            k1,
        )

        k0 *= rms[:, None]
        k0 *= k_weight_0
        kr = tl.reshape(
            tl.permute(
                tl.flip(tl.permute(tl.reshape(k0, (h, 2, d)), (0, 2, 1)), dim=2)
                * signs,
                (0, 2, 1),
            ),
            (h, D),
        )
        k0 = k0 * cos + kr * sin
        tl.store(
            ko_ptr
            + pid * h * DD
            + i * L * h * DD
            + DD * tl.arange(0, h)[:, None]
            + tl.arange(0, D)[None, :],
            k0,
        )

    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        v_ptr = qkv_ptr + DD * w + DD
    else:
        row_offs = tl.arange(0, h)
        v_ptr = qkv_ptr + DD * H + DD * h
    for i in range(B):
        if TRANSPOSED:
            v0 = tl.load(
                v_ptr
                + pid * B * stride
                + i * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
            v1 = tl.load(
                v_ptr
                + pid * B * stride
                + i * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
        else:
            v0 = tl.load(
                v_ptr
                + i * L * stride
                + pid * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
            v1 = tl.load(
                v_ptr
                + i * L * stride
                + pid * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
        if SILU:
            v0 = v0 * tl.sigmoid(v0)
            v1 = v1 * tl.sigmoid(v1)

        tl.store(
            vo_ptr
            + pid * h * DD
            + i * L * h * DD
            + DD * tl.arange(0, h)[:, None]
            + tl.arange(0, D)[None, :],
            v0,
        )
        tl.store(
            vo_ptr
            + pid * h * DD
            + i * L * h * DD
            + D
            + DD * tl.arange(0, h)[:, None]
            + tl.arange(0, D)[None, :],
            v1,
        )


@triton.jit
def compatible_qk_norm_and_half_rop_forward_kernel(
    qkv_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    freqs_ptr,
    qo_ptr,
    ko_ptr,
    vo_ptr,
    B,
    stride,
    eps,
    H: tl.constexpr,
    h: tl.constexpr,
    H_p: tl.constexpr,
    h_p: tl.constexpr,
    D: tl.constexpr,
    d: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    SILU: tl.constexpr,
):
    pid = tl.program_id(0)
    L = tl.num_programs(0)
    DD = D * 2

    freqs = tl.load(freqs_ptr + pid * D + tl.arange(0, D)).to(tl.float32)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)
    signs = tl.arange(0, 2).to(tl.float32) * 2 - 1

    q_weight_0 = tl.load(q_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    q_weight_1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)
    q_ptr = qkv_ptr
    w = H // h  # H =8 h = 2 w=4

    # [len, bs, q_head, head_dim] -> [bs, len, q_head, head_dim]
    if INTERLEAVED:
        # row_offs = tl.arange(0, H) + tl.arange(0, H) // w * 2
        row_offs = tl.arange(0, H_p) + tl.arange(0, H_p) // w * 2
        row_mask = row_offs[:, None] < (H + 2 * h)
    else:
        # row_offs = tl.arange(0, H)
        row_offs = tl.arange(0, H_p)
        row_mask = row_offs[:, None] < H

    for i in range(B):
        if TRANSPOSED:
            q0 = tl.load(
                q_ptr
                + pid * B * stride
                + i * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)
            q1 = tl.load(
                q_ptr
                + pid * B * stride
                + i * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)
        else:
            q0 = tl.load(
                q_ptr
                + i * L * stride
                + pid * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)
            q1 = tl.load(
                q_ptr
                + i * L * stride
                + pid * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)
        if SILU:
            q0 = q0 * tl.sigmoid(q0.to(tl.float32))
            q1 = q1 * tl.sigmoid(q1.to(tl.float32))
        rms = tl.rsqrt((tl.sum(q0 * q0, 1) + tl.sum(q1 * q1, 1)) / DD + eps)
        q1 *= rms[:, None]
        q1 *= q_weight_1
        q_mask = tl.arange(0, H_p)[:, None] < H
        tl.store(
            qo_ptr
            + pid * H * DD
            + i * L * H * DD
            + D
            + DD * tl.arange(0, H_p)[:, None]
            + tl.arange(0, D)[None, :],
            q1,
            mask=q_mask,
        )

        q0 *= rms[:, None]
        q0 *= q_weight_0
        qr = tl.reshape(
            tl.permute(
                tl.flip(tl.permute(tl.reshape(q0, (H_p, 2, d)), (0, 2, 1)), dim=2)
                * signs,
                (0, 2, 1),
            ),
            (H_p, D),
        )
        q0 = q0 * cos + qr * sin
        tl.store(
            qo_ptr
            + pid * H * DD
            + i * L * H * DD
            + DD * tl.arange(0, H_p)[:, None]
            + tl.arange(0, D)[None, :],
            q0,
            mask=q_mask,
        )

    k_weight_0 = tl.load(k_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    k_weight_1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)
    if INTERLEAVED:
        # row_offs = tl.arange(0, h) * (w + 2)
        row_offs = tl.arange(0, h_p) * (w + 2)
        row_mask = row_offs[:, None] < (h * (w + 2))
        k_ptr = qkv_ptr + DD * w
    else:
        # row_offs = tl.arange(0, h)
        row_offs = tl.arange(0, h_p)
        row_mask = tl.arange(0, h_p)[:, None] < h
        k_ptr = qkv_ptr + DD * H

    for i in range(B):
        if TRANSPOSED:
            k0 = tl.load(
                k_ptr
                + pid * B * stride
                + i * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)
            k1 = tl.load(
                k_ptr
                + pid * B * stride
                + i * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)
        else:
            k0 = tl.load(
                k_ptr
                + i * L * stride
                + pid * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)
            k1 = tl.load(
                k_ptr
                + i * L * stride
                + pid * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)

        if SILU:
            k0 = k0 * tl.sigmoid(k0)
            k1 = k1 * tl.sigmoid(k1)
        rms = tl.rsqrt((tl.sum(k0 * k0, 1) + tl.sum(k1 * k1, 1)) / DD + eps)
        k1 *= rms[:, None]
        k1 *= k_weight_1
        k_mask = tl.arange(0, h_p)[:, None] < h
        tl.store(
            ko_ptr
            + pid * h * DD
            + i * L * h * DD
            + D
            + DD * tl.arange(0, h_p)[:, None]
            + tl.arange(0, D)[None, :],
            k1,
            mask=k_mask,
        )

        k0 *= rms[:, None]
        k0 *= k_weight_0
        kr = tl.reshape(
            tl.permute(
                tl.flip(tl.permute(tl.reshape(k0, (h_p, 2, d)), (0, 2, 1)), dim=2)
                * signs,
                (0, 2, 1),
            ),
            (h_p, D),
        )
        k0 = k0 * cos + kr * sin
        tl.store(
            ko_ptr
            + pid * h * DD
            + i * L * h * DD
            + DD * tl.arange(0, h_p)[:, None]
            + tl.arange(0, D)[None, :],
            k0,
            mask=k_mask,
        )

    if INTERLEAVED:
        # row_offs = tl.arange(0, h) * (w + 2)
        row_offs = tl.arange(0, h_p) * (w + 2)
        row_mask = row_offs[:, None] < (h * (w + 2))
        v_ptr = qkv_ptr + DD * w + DD
    else:
        # row_offs = tl.arange(0, h)
        row_offs = tl.arange(0, h_p)
        row_mask = tl.arange(0, h_p)[:, None] < h
        v_ptr = qkv_ptr + DD * H + DD * h

    for i in range(B):
        if TRANSPOSED:
            v0 = tl.load(
                v_ptr
                + pid * B * stride
                + i * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)
            v1 = tl.load(
                v_ptr
                + pid * B * stride
                + i * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)
        else:
            v0 = tl.load(
                v_ptr
                + i * L * stride
                + pid * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)
            v1 = tl.load(
                v_ptr
                + i * L * stride
                + pid * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)
        if SILU:
            v0 = v0 * tl.sigmoid(v0)
            v1 = v1 * tl.sigmoid(v1)

        v_mask = tl.arange(0, h_p)[:, None] < h
        tl.store(
            vo_ptr
            + pid * h * DD
            + i * L * h * DD
            + DD * tl.arange(0, h_p)[:, None]
            + tl.arange(0, D)[None, :],
            v0,
            mask=v_mask,
        )
        tl.store(
            vo_ptr
            + pid * h * DD
            + i * L * h * DD
            + D
            + DD * tl.arange(0, h_p)[:, None]
            + tl.arange(0, D)[None, :],
            v1,
            mask=v_mask,
        )


def triton_qk_norm_and_half_rope_forward(
    qkv,
    q_norm_weight,
    k_norm_weight,
    freqs,
    H=32,
    h=4,
    eps=1e-6,
    interleaved=True,
    transposed=True,
    silu=False,
):
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
        silu: apply silu on qkv before qk norm and rope
    Returns:
        - qo: shape [B, S, H, head_dim]
        - ko: shape [B, S, h, head_dim]
        - vo: shape [B, S, h, head_dim]
    """
    assert qkv.is_contiguous() and freqs.is_contiguous()
    assert k_norm_weight.is_contiguous() and q_norm_weight.is_contiguous()
    if transposed:
        L, B, Dim = qkv.shape
    else:
        B, L, Dim = qkv.shape
    stride = qkv.stride(1)  # qkv may be a slice of a tensor
    D = k_norm_weight.size(0)
    tp = (H + 2 * h) * D // Dim
    if tp > 1:
        H = H // tp
        h = h // tp
    # D = Dim // (H + 2 * h)  # error with tp
    assert freqs.size(0) == L and freqs.size(-1) == D // 2, f"{freqs.shape=} {L=} {D=}"
    dtype = qkv.dtype
    device = qkv.device
    qo = torch.empty((B, L, H, D), dtype=dtype, device=device)
    ko = torch.empty((B, L, h, D), dtype=dtype, device=device)
    vo = torch.empty((B, L, h, D), dtype=dtype, device=device)

    num_stages = 5
    num_warps = 2
    grid = (L,)

    H_p = triton.next_power_of_2(H)
    h_p = triton.next_power_of_2(h)

    if H_p == H and h_p == h:
        qk_norm_and_half_rope_forward_kernel[grid](
            qkv,
            q_norm_weight,
            k_norm_weight,
            freqs,
            qo,
            ko,
            vo,
            B,
            stride,
            eps,
            H,
            h,
            D // 2,
            D // 4,
            interleaved,
            transposed,
            silu,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    else:
        compatible_qk_norm_and_half_rop_forward_kernel[grid](
            qkv,
            q_norm_weight,
            k_norm_weight,
            freqs,
            qo,
            ko,
            vo,
            B,
            stride,
            eps,
            H,
            h,
            H_p,
            h_p,
            D // 2,
            D // 4,
            interleaved,
            transposed,
            silu,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    return qo, ko, vo


@triton.jit
def qk_norm_and_half_rope_backward_kernel(
    gq_ptr,
    gk_ptr,
    gv_ptr,
    qkv_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    freqs_ptr,
    dqkv_ptr,
    dqw_ptr,
    dkw_ptr,
    B,
    stride,
    grad_stride,
    eps,
    H: tl.constexpr,
    h: tl.constexpr,
    D: tl.constexpr,
    d: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    SILU: tl.constexpr,
):
    pid = tl.program_id(0)
    L = tl.num_programs(0)
    DD = 2 * D
    w = H // h

    freqs = tl.load(freqs_ptr + pid * D + tl.arange(0, D)).to(tl.float32)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)
    signs = -tl.arange(0, 2).to(tl.float32) * 2 + 1

    q_w0 = tl.load(q_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    q_w1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)

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
            gq_ptr
            + i * L * H * DD
            + pid * H * DD
            + DD * tl.arange(0, H)[:, None]
            + tl.arange(0, D)[None, :]
        ).to(tl.float32)
        gq_1 = tl.load(
            gq_ptr
            + i * L * H * DD
            + pid * H * DD
            + D
            + DD * tl.arange(0, H)[:, None]
            + tl.arange(0, D)[None, :]
        ).to(tl.float32)

        gq_r = tl.reshape(
            tl.permute(
                tl.flip(tl.permute(tl.reshape(gq_0, (H, 2, d)), (0, 2, 1)), dim=2)
                * signs,
                (0, 2, 1),
            ),
            (H, D),
        )
        gq_0 = gq_0 * cos + gq_r * sin

        if TRANSPOSED:
            q0 = tl.load(
                q_ptr
                + pid * B * stride
                + i * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
            q1 = tl.load(
                q_ptr
                + pid * B * stride
                + i * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
        else:
            q0 = tl.load(
                q_ptr
                + pid * stride
                + i * L * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
            q1 = tl.load(
                q_ptr
                + pid * stride
                + i * L * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)

        if SILU:
            s0 = tl.sigmoid(q0)
            s1 = tl.sigmoid(q1)
            q_0 = q0 * s0
            q_1 = q1 * s1

            r = tl.rsqrt((tl.sum(q_0 * q_0, 1) + tl.sum(q_1 * q_1, 1)) / DD + eps)[
                :, None
            ]

            dqw_0 += tl.sum(q_0 * gq_0 * r, 0)
            dqw_1 += tl.sum(q_1 * gq_1 * r, 0)

            s = tl.sum(q_0 * gq_0 * q_w0, 1) + tl.sum(q_1 * gq_1 * q_w1, 1)

            dq_0 = r * gq_0 * q_w0 - r * r * r / DD * q_0 * s[:, None]
            dq_1 = r * gq_1 * q_w1 - r * r * r / DD * q_1 * s[:, None]

            dq_0 = dq_0 * s0 * (1 + q0 * (1 - s0))
            dq_1 = dq_1 * s1 * (1 + q1 * (1 - s1))

        else:
            r = tl.rsqrt((tl.sum(q0 * q0, 1) + tl.sum(q1 * q1, 1)) / DD + eps)[:, None]

            dqw_0 += tl.sum(q0 * gq_0 * r, 0)
            dqw_1 += tl.sum(q1 * gq_1 * r, 0)

            s = tl.sum(q0 * gq_0 * q_w0, 1) + tl.sum(q1 * gq_1 * q_w1, 1)

            dq_0 = r * gq_0 * q_w0 - r * r * r / DD * q0 * s[:, None]
            dq_1 = r * gq_1 * q_w1 - r * r * r / DD * q1 * s[:, None]

        if TRANSPOSED:
            tl.store(
                dq_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dq_0,
            )
            tl.store(
                dq_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dq_1,
            )
        else:
            tl.store(
                dq_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dq_0,
            )
            tl.store(
                dq_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dq_1,
            )

    tl.store(dqw_ptr + pid * D * 2 + tl.arange(0, D), dqw_0)
    tl.store(dqw_ptr + pid * D * 2 + D + tl.arange(0, D), dqw_1)

    k_w0 = tl.load(k_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    k_w1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)

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
            gk_ptr
            + i * L * h * DD
            + pid * h * DD
            + DD * tl.arange(0, h)[:, None]
            + tl.arange(0, D)[None, :]
        ).to(tl.float32)
        gk_1 = tl.load(
            gk_ptr
            + i * L * h * DD
            + pid * h * DD
            + D
            + DD * tl.arange(0, h)[:, None]
            + tl.arange(0, D)[None, :]
        ).to(tl.float32)

        gk_r = tl.reshape(
            tl.permute(
                tl.flip(tl.permute(tl.reshape(gk_0, (h, 2, d)), (0, 2, 1)), dim=2)
                * signs,
                (0, 2, 1),
            ),
            (h, D),
        )
        gk_0 = gk_0 * cos + gk_r * sin

        if TRANSPOSED:
            k0 = tl.load(
                k_ptr
                + pid * B * stride
                + i * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
            k1 = tl.load(
                k_ptr
                + pid * B * stride
                + i * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
        else:
            k0 = tl.load(
                k_ptr
                + pid * stride
                + i * L * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)
            k1 = tl.load(
                k_ptr
                + pid * stride
                + i * L * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :]
            ).to(tl.float32)

        if SILU:

            s0 = tl.sigmoid(k0)
            s1 = tl.sigmoid(k1)
            k_0 = k0 * s0
            k_1 = k1 * s1

            r = tl.rsqrt((tl.sum(k_0 * k_0, 1) + tl.sum(k_1 * k_1, 1)) / DD + eps)[
                :, None
            ]

            dkw_0 += tl.sum(k_0 * gk_0 * r, 0)
            dkw_1 += tl.sum(k_1 * gk_1 * r, 0)

            s = tl.sum(k_0 * gk_0 * k_w0, 1) + tl.sum(k_1 * gk_1 * k_w1, 1)

            dk_0 = r * gk_0 * k_w0 - r * r * r / DD * k_0 * s[:, None]
            dk_1 = r * gk_1 * k_w1 - r * r * r / DD * k_1 * s[:, None]

            dk_0 = dk_0 * s0 * (1 + k0 * (1 - s0))
            dk_1 = dk_1 * s1 * (1 + k1 * (1 - s1))

        else:
            r = tl.rsqrt((tl.sum(k0 * k0, 1) + tl.sum(k1 * k1, 1)) / DD + eps)[:, None]

            dkw_0 += tl.sum(k0 * gk_0 * r, 0)
            dkw_1 += tl.sum(k1 * gk_1 * r, 0)

            s = tl.sum(k0 * gk_0 * k_w0, 1) + tl.sum(k1 * gk_1 * k_w1, 1)

            dk_0 = r * gk_0 * k_w0 - r * r * r / DD * k0 * s[:, None]
            dk_1 = r * gk_1 * k_w1 - r * r * r / DD * k1 * s[:, None]

        if TRANSPOSED:
            tl.store(
                dk_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dk_0,
            )
            tl.store(
                dk_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dk_1,
            )
        else:
            tl.store(
                dk_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dk_0,
            )
            tl.store(
                dk_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dk_1,
            )
    tl.store(dkw_ptr + pid * D * 2 + tl.arange(0, D), dkw_0)
    tl.store(dkw_ptr + pid * D * 2 + D + tl.arange(0, D), dkw_1)

    # [bs, len, k_head, head_dim] -> [len, bs, k_head + 2 * kv_head, head_dim]
    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        v_ptr = qkv_ptr + DD * w + DD
        dv_ptr = dqkv_ptr + DD * w + DD
    else:
        row_offs = tl.arange(0, h)
        v_ptr = qkv_ptr + DD * H + DD * h
        dv_ptr = dqkv_ptr + DD * H + DD * h
    for i in range(B):

        gv_0 = tl.load(
            gv_ptr
            + i * L * h * DD
            + pid * h * DD
            + DD * tl.arange(0, h)[:, None]
            + tl.arange(0, D)[None, :]
        ).to(tl.float32)
        gv_1 = tl.load(
            gv_ptr
            + i * L * h * DD
            + pid * h * DD
            + D
            + DD * tl.arange(0, h)[:, None]
            + tl.arange(0, D)[None, :]
        ).to(tl.float32)

        if SILU:
            if TRANSPOSED:
                v0 = tl.load(
                    v_ptr
                    + pid * B * stride
                    + i * stride
                    + DD * row_offs[:, None]
                    + tl.arange(0, D)[None, :]
                ).to(tl.float32)
                v1 = tl.load(
                    v_ptr
                    + pid * B * stride
                    + i * stride
                    + D
                    + DD * row_offs[:, None]
                    + tl.arange(0, D)[None, :]
                ).to(tl.float32)
            else:
                v0 = tl.load(
                    v_ptr
                    + i * L * stride
                    + pid * stride
                    + DD * row_offs[:, None]
                    + tl.arange(0, D)[None, :]
                ).to(tl.float32)
                v1 = tl.load(
                    v_ptr
                    + i * L * stride
                    + pid * stride
                    + D
                    + DD * row_offs[:, None]
                    + tl.arange(0, D)[None, :]
                ).to(tl.float32)

            s0 = tl.sigmoid(v0)
            s1 = tl.sigmoid(v1)
            dv_0 = gv_0 * s0 * (1 + v0 * (1 - s0))
            dv_1 = gv_1 * s1 * (1 + v1 * (1 - s1))
        else:
            dv_0 = gv_0
            dv_1 = gv_1

        if TRANSPOSED:
            tl.store(
                dv_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dv_0,
            )
            tl.store(
                dv_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dv_1,
            )
        else:
            tl.store(
                dv_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dv_0,
            )
            tl.store(
                dv_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dv_1,
            )


@triton.jit
def compatible_qk_norm_and_half_rope_backward_kernel(
    gq_ptr,
    gk_ptr,
    gv_ptr,
    qkv_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    freqs_ptr,
    dqkv_ptr,
    dqw_ptr,
    dkw_ptr,
    B,
    stride,
    grad_stride,
    eps,
    H: tl.constexpr,
    h: tl.constexpr,
    H_p: tl.constexpr,
    h_p: tl.constexpr,
    D: tl.constexpr,
    d: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    SILU: tl.constexpr,
):
    pid = tl.program_id(0)
    L = tl.num_programs(0)
    DD = 2 * D
    w = H // h

    freqs = tl.load(freqs_ptr + pid * D + tl.arange(0, D)).to(tl.float32)
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
        # row_offs = tl.arange(0, H) + tl.arange(0, H) // w * 2
        row_offs = tl.arange(0, H_p) + tl.arange(0, H_p) // w * 2
        row_mask = row_offs[:, None] < (H + 2 * h)
    else:
        # row_offs = tl.arange(0, H)
        row_offs = tl.arange(0, H_p)
        row_mask = row_offs[:, None] < H

    for i in range(B):
        gq_0 = tl.load(
            gq_ptr
            + i * L * H * DD
            + pid * H * DD
            + DD * tl.arange(0, H_p)[:, None]
            + tl.arange(0, D)[None, :],
            mask=tl.arange(0, H_p)[:, None] < H,
        ).to(tl.float32)
        gq_1 = tl.load(
            gq_ptr
            + i * L * H * DD
            + pid * H * DD
            + D
            + DD * tl.arange(0, H_p)[:, None]
            + tl.arange(0, D)[None, :],
            mask=tl.arange(0, H_p)[:, None] < H,
        ).to(tl.float32)

        gq_r = tl.reshape(
            tl.permute(
                tl.flip(tl.permute(tl.reshape(gq_0, (H_p, 2, d)), (0, 2, 1)), dim=2)
                * signs,
                (0, 2, 1),
            ),
            (H_p, D),
        )
        gq_0 = gq_0 * cos + gq_r * sin

        if TRANSPOSED:
            # q0 = tl.load(q_ptr + pid * B * stride + i * stride + DD * row_offs[:,None] + tl.arange(0, D)[None, :])
            # q1 = tl.load(q_ptr + pid * B * stride + i * stride + D + DD * row_offs[:,None] + tl.arange(0, D)[None, :])
            q0 = tl.load(
                q_ptr
                + pid * B * stride
                + i * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)
            q1 = tl.load(
                q_ptr
                + pid * B * stride
                + i * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)

        else:
            # q0 = tl.load(q_ptr + pid * stride + i * L * stride + DD * row_offs[:,None] + tl.arange(0, D)[None, :])
            # q1 = tl.load(q_ptr + pid * stride + i * L * stride + D + DD * row_offs[:,None] + tl.arange(0, D)[None, :])
            q0 = tl.load(
                q_ptr
                + pid * stride
                + i * L * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)
            q1 = tl.load(
                q_ptr
                + pid * stride
                + i * L * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)

        if SILU:
            s0 = tl.sigmoid(q0)
            s1 = tl.sigmoid(q1)
            q_0 = q0 * s0
            q_1 = q1 * s1

            r = tl.rsqrt((tl.sum(q_0 * q_0, 1) + tl.sum(q_1 * q_1, 1)) / DD + eps)[
                :, None
            ]

            dqw_0 += tl.sum(q_0 * gq_0 * r, 0)
            dqw_1 += tl.sum(q_1 * gq_1 * r, 0)

            s = tl.sum(q_0 * gq_0 * q_w0, 1) + tl.sum(q_1 * gq_1 * q_w1, 1)

            dq_0 = r * gq_0 * q_w0 - r * r * r / DD * q_0 * s[:, None]
            dq_1 = r * gq_1 * q_w1 - r * r * r / DD * q_1 * s[:, None]

            dq_0 = dq_0 * s0 * (1 + q0 * (1 - s0))
            dq_1 = dq_1 * s1 * (1 + q1 * (1 - s1))

        else:
            r = tl.rsqrt((tl.sum(q0 * q0, 1) + tl.sum(q1 * q1, 1)) / DD + eps)[:, None]

            dqw_0 += tl.sum(q0 * gq_0 * r, 0)
            dqw_1 += tl.sum(q1 * gq_1 * r, 0)

            s = tl.sum(q0 * gq_0 * q_w0, 1) + tl.sum(q1 * gq_1 * q_w1, 1)

            dq_0 = r * gq_0 * q_w0 - r * r * r / DD * q0 * s[:, None]
            dq_1 = r * gq_1 * q_w1 - r * r * r / DD * q1 * s[:, None]

        if TRANSPOSED:
            # tl.store(dq_ptr + pid * B * grad_stride + i * grad_stride + DD * row_offs[:,None] + tl.arange(0, D)[None, :], dq_0)
            # tl.store(dq_ptr + pid * B * grad_stride + i * grad_stride + D + DD * row_offs[:,None] + tl.arange(0, D)[None, :], dq_1)
            tl.store(
                dq_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dq_0,
                mask=row_mask,
            )
            tl.store(
                dq_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dq_1,
                mask=row_mask,
            )

        else:
            # tl.store(dq_ptr + pid * grad_stride + i * L * grad_stride + DD * row_offs[:,None] + tl.arange(0, D)[None, :], dq_0)
            # tl.store(dq_ptr + pid * grad_stride + i * L * grad_stride + D + DD * row_offs[:,None] + tl.arange(0, D)[None, :], dq_1)
            tl.store(
                dq_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dq_0,
                mask=row_mask,
            )
            tl.store(
                dq_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dq_1,
                mask=row_mask,
            )

    tl.store(dqw_ptr + pid * D * 2 + tl.arange(0, D), dqw_0)
    tl.store(dqw_ptr + pid * D * 2 + D + tl.arange(0, D), dqw_1)

    k_w0 = tl.load(k_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    k_w1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)

    dkw_0 = tl.zeros((D,), dtype=tl.float32)
    dkw_1 = tl.zeros((D,), dtype=tl.float32)
    if INTERLEAVED:
        # row_offs = tl.arange(0, h) * (w + 2)
        row_offs = tl.arange(0, h_p) * (w + 2)
        row_mask = row_offs[:, None] < (h * (w + 2))
        k_ptr = qkv_ptr + DD * w
        dk_ptr = dqkv_ptr + DD * w
    else:
        # row_offs = tl.arange(0, h)
        row_offs = tl.arange(0, h_p)
        row_mask = row_offs[:, None] < h
        k_ptr = qkv_ptr + DD * H
        dk_ptr = dqkv_ptr + DD * H
    # [bs, len, k_head, head_dim] -> [len, bs, k_head, head_dim]
    for i in range(B):
        gk_0 = tl.load(
            gk_ptr
            + i * L * h * DD
            + pid * h * DD
            + DD * tl.arange(0, h_p)[:, None]
            + tl.arange(0, D)[None, :],
            mask=tl.arange(0, h_p)[:, None] < h,
        ).to(tl.float32)
        gk_1 = tl.load(
            gk_ptr
            + i * L * h * DD
            + pid * h * DD
            + D
            + DD * tl.arange(0, h_p)[:, None]
            + tl.arange(0, D)[None, :],
            mask=tl.arange(0, h_p)[:, None] < h,
        ).to(tl.float32)

        gk_r = tl.reshape(
            tl.permute(
                tl.flip(tl.permute(tl.reshape(gk_0, (h_p, 2, d)), (0, 2, 1)), dim=2)
                * signs,
                (0, 2, 1),
            ),
            (h_p, D),
        )
        gk_0 = gk_0 * cos + gk_r * sin

        if TRANSPOSED:
            k0 = tl.load(
                k_ptr
                + pid * B * stride
                + i * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)
            k1 = tl.load(
                k_ptr
                + pid * B * stride
                + i * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)
        else:
            k0 = tl.load(
                k_ptr
                + pid * stride
                + i * L * stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)
            k1 = tl.load(
                k_ptr
                + pid * stride
                + i * L * stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                mask=row_mask,
            ).to(tl.float32)

        if SILU:

            s0 = tl.sigmoid(k0)
            s1 = tl.sigmoid(k1)
            k_0 = k0 * s0
            k_1 = k1 * s1

            r = tl.rsqrt((tl.sum(k_0 * k_0, 1) + tl.sum(k_1 * k_1, 1)) / DD + eps)[
                :, None
            ]

            dkw_0 += tl.sum(k_0 * gk_0 * r, 0)
            dkw_1 += tl.sum(k_1 * gk_1 * r, 0)

            s = tl.sum(k_0 * gk_0 * k_w0, 1) + tl.sum(k_1 * gk_1 * k_w1, 1)

            dk_0 = r * gk_0 * k_w0 - r * r * r / DD * k_0 * s[:, None]
            dk_1 = r * gk_1 * k_w1 - r * r * r / DD * k_1 * s[:, None]

            dk_0 = dk_0 * s0 * (1 + k0 * (1 - s0))
            dk_1 = dk_1 * s1 * (1 + k1 * (1 - s1))

        else:
            r = tl.rsqrt((tl.sum(k0 * k0, 1) + tl.sum(k1 * k1, 1)) / DD + eps)[:, None]

            dkw_0 += tl.sum(k0 * gk_0 * r, 0)
            dkw_1 += tl.sum(k1 * gk_1 * r, 0)

            s = tl.sum(k0 * gk_0 * k_w0, 1) + tl.sum(k1 * gk_1 * k_w1, 1)

            dk_0 = r * gk_0 * k_w0 - r * r * r / DD * k0 * s[:, None]
            dk_1 = r * gk_1 * k_w1 - r * r * r / DD * k1 * s[:, None]

        if TRANSPOSED:
            tl.store(
                dk_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dk_0,
                mask=row_mask,
            )
            tl.store(
                dk_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dk_1,
                mask=row_mask,
            )
        else:
            tl.store(
                dk_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dk_0,
                mask=row_mask,
            )
            tl.store(
                dk_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dk_1,
                mask=row_mask,
            )

    tl.store(dkw_ptr + pid * D * 2 + tl.arange(0, D), dkw_0)
    tl.store(dkw_ptr + pid * D * 2 + D + tl.arange(0, D), dkw_1)

    # [bs, len, k_head, head_dim] -> [len, bs, k_head + 2 * kv_head, head_dim]
    if INTERLEAVED:
        # row_offs = tl.arange(0, h) * (w + 2)
        row_offs = tl.arange(0, h_p) * (w + 2)
        row_mask = row_offs[:, None] < (h * (w + 2))
        v_ptr = qkv_ptr + DD * w + DD
        dv_ptr = dqkv_ptr + DD * w + DD
    else:
        # row_offs = tl.arange(0, h)
        row_offs = tl.arange(0, h_p)
        row_mask = row_offs[:, None] < h
        v_ptr = qkv_ptr + DD * H + DD * h
        dv_ptr = dqkv_ptr + DD * H + DD * h
    for i in range(B):

        gv_0 = tl.load(
            gv_ptr
            + i * L * h * DD
            + pid * h * DD
            + DD * tl.arange(0, h_p)[:, None]
            + tl.arange(0, D)[None, :],
            mask=tl.arange(0, h_p)[:, None] < h,
        ).to(tl.float32)
        gv_1 = tl.load(
            gv_ptr
            + i * L * h * DD
            + pid * h * DD
            + D
            + DD * tl.arange(0, h_p)[:, None]
            + tl.arange(0, D)[None, :],
            mask=tl.arange(0, h_p)[:, None] < h,
        ).to(tl.float32)

        if SILU:
            if TRANSPOSED:
                v0 = tl.load(
                    v_ptr
                    + pid * B * stride
                    + i * stride
                    + DD * row_offs[:, None]
                    + tl.arange(0, D)[None, :],
                    mask=row_mask,
                ).to(tl.float32)
                v1 = tl.load(
                    v_ptr
                    + pid * B * stride
                    + i * stride
                    + D
                    + DD * row_offs[:, None]
                    + tl.arange(0, D)[None, :],
                    mask=row_mask,
                ).to(tl.float32)
            else:
                v0 = tl.load(
                    v_ptr
                    + i * L * stride
                    + pid * stride
                    + DD * row_offs[:, None]
                    + tl.arange(0, D)[None, :],
                    mask=row_mask,
                ).to(tl.float32)
                v1 = tl.load(
                    v_ptr
                    + i * L * stride
                    + pid * stride
                    + D
                    + DD * row_offs[:, None]
                    + tl.arange(0, D)[None, :],
                    mask=row_mask,
                ).to(tl.float32)

            s0 = tl.sigmoid(v0)
            s1 = tl.sigmoid(v1)
            dv_0 = gv_0 * s0 * (1 + v0 * (1 - s0))
            dv_1 = gv_1 * s1 * (1 + v1 * (1 - s1))
        else:
            dv_0 = gv_0
            dv_1 = gv_1

        if TRANSPOSED:
            tl.store(
                dv_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dv_0,
                mask=row_mask,
            )
            tl.store(
                dv_ptr
                + pid * B * grad_stride
                + i * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dv_1,
                mask=row_mask,
            )
        else:
            tl.store(
                dv_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dv_0,
                mask=row_mask,
            )
            tl.store(
                dv_ptr
                + pid * grad_stride
                + i * L * grad_stride
                + D
                + DD * row_offs[:, None]
                + tl.arange(0, D)[None, :],
                dv_1,
                mask=row_mask,
            )


def triton_qk_norm_and_half_rope_backward(
    gq,
    gk,
    gv,
    qkv,
    q_norm_weight,
    k_norm_weight,
    freqs,
    eps=1e-6,
    interleaved=True,
    transposed=True,
    silu=False,
):
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
        silu: whether silu is applied to qkv

    Returns:
        - dqkv: gradient of qkv
        - dqw: gradient of q_norm_weight
        - dkw: gradient of k_norm_weight
    """
    assert gq.is_contiguous() and gk.is_contiguous() and gv.is_contiguous()
    B, L, H, D = gq.shape
    h = gk.shape[2]
    stride = qkv.stride(1)

    dtype = gq.dtype
    device = gq.device
    if transposed:
        dqkv = torch.empty((L, B, (H + 2 * h) * D), dtype=dtype, device=device)
    else:
        dqkv = torch.empty((B, L, (H + 2 * h) * D), dtype=dtype, device=device)
    grad_stride = dqkv.stride(1)  # for potential fused kernel

    tmp_dqw = torch.empty((L, D), dtype=torch.float32, device=device)
    tmp_dkw = torch.empty((L, D), dtype=torch.float32, device=device)

    H_p = triton.next_power_of_2(H)
    h_p = triton.next_power_of_2(h)

    num_stages = 5
    num_warps = 1
    grid = (L,)
    if H == H_p and h == h_p:
        qk_norm_and_half_rope_backward_kernel[grid](
            gq,
            gk,
            gv,
            qkv,
            q_norm_weight,
            k_norm_weight,
            freqs,
            dqkv,
            tmp_dqw,
            tmp_dkw,
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
            silu,
            num_stages=num_stages,
            num_warps=num_warps,
        )

    else:
        compatible_qk_norm_and_half_rope_backward_kernel[grid](
            gq,
            gk,
            gv,
            qkv,
            q_norm_weight,
            k_norm_weight,
            freqs,
            dqkv,
            tmp_dqw,
            tmp_dkw,
            B,
            stride,
            grad_stride,
            eps,
            H,
            h,
            H_p,
            h_p,
            D // 2,
            D // 4,
            interleaved,
            transposed,
            silu,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    dqw = tmp_dqw.sum(0)
    dkw = tmp_dkw.sum(0)
    return dqkv, dqw, dkw


@triton.jit
def _get_varlen_token_idx(cu_seqlens, pid_m, seq_num, padded_seq_num, cp_rank, cp_size):
    cus = (
        tl.load(
            cu_seqlens + tl.arange(0, padded_seq_num),
            mask=tl.arange(0, padded_seq_num) <= seq_num,
        )
        // cp_size
    )
    cu = tl.max(tl.where(cus > pid_m, 0, cus), 0)
    cun = tl.min(tl.where(cus <= cu, 2**24, cus), 0)
    length = cun - cu
    token_idx = pid_m - cu

    if cp_size > 1:
        if token_idx < length // 2:
            token_idx = token_idx + cp_rank * length // 2
        else:
            token_idx = (token_idx - length // 2) + (
                2 * cp_size - cp_rank - 1
            ) * length // 2
    return token_idx


# not used
@triton.jit
def _get_fixlen_token_idx(num_tokens, pid_m, seq_num, cp_rank, cp_size, transpose):
    L = num_tokens // seq_num
    if transpose:
        token_idx = pid_m % L
    else:
        token_idx = pid_m // seq_num
    if cp_size > 1:
        if token_idx < L // 2:
            token_idx = token_idx + cp_rank * L // 2
        else:
            token_idx = (token_idx - L // 2) + (2 * cp_size - cp_rank - 1) * L // 2
    return token_idx


@triton.jit
def varlen_qk_norm_and_half_rope_forward_kernel(
    qkv_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    freqs_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_kv_ptr,
    qo_ptr,
    ko_ptr,
    vo_ptr,
    stride,
    eps,
    mscale,
    cp_rank,
    B,
    PB: tl.constexpr,
    H: tl.constexpr,
    h: tl.constexpr,
    D: tl.constexpr,
    d: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    SILU: tl.constexpr,
    CP_SIZE: tl.constexpr,
    REUSE: tl.constexpr,
):
    pid = tl.program_id(0)

    pos = _get_varlen_token_idx(cu_seqlens_q_ptr, pid, B, PB, cp_rank, CP_SIZE)

    DD = D * 2

    freqs = tl.load(freqs_ptr + pos * D + tl.arange(0, D)).to(tl.float32)
    cos = tl.cos(freqs) * mscale
    sin = tl.sin(freqs) * mscale
    signs = tl.arange(0, 2).to(tl.float32) * 2 - 1

    q_weight_0 = tl.load(q_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    q_weight_1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)
    q_ptr = qkv_ptr
    w = H // h

    # [len, bs, q_head, head_dim] -> [bs, len, q_head, head_dim]
    if INTERLEAVED:
        row_offs = tl.arange(0, H) + tl.arange(0, H) // w * 2
    else:
        row_offs = tl.arange(0, H)

    q0 = tl.load(
        q_ptr + pid * stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :]
    ).to(tl.float32)
    q1 = tl.load(
        q_ptr + pid * stride + D + DD * row_offs[:, None] + tl.arange(0, D)[None, :]
    ).to(tl.float32)

    if SILU:
        q0 = q0 * tl.sigmoid(q0)
        q1 = q1 * tl.sigmoid(q1)
    rms = tl.rsqrt((tl.sum(q0 * q0, 1) + tl.sum(q1 * q1, 1)) / DD + eps)
    q1 *= rms[:, None]
    q1 *= q_weight_1
    tl.store(
        qo_ptr
        + pid * H * DD
        + D
        + DD * tl.arange(0, H)[:, None]
        + tl.arange(0, D)[None, :],
        q1,
    )

    q0 *= rms[:, None]
    q0 *= q_weight_0
    qr = tl.reshape(
        tl.permute(
            tl.flip(tl.permute(tl.reshape(q0, (H, 2, d)), (0, 2, 1)), dim=2) * signs,
            (0, 2, 1),
        ),
        (H, D),
    )
    q0 = q0 * cos + qr * sin
    tl.store(
        qo_ptr
        + pid * H * DD
        + DD * tl.arange(0, H)[:, None]
        + tl.arange(0, D)[None, :],
        q0,
    )

    k_weight_0 = tl.load(k_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    k_weight_1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)

    if not REUSE:
        pos = _get_varlen_token_idx(cu_seqlens_kv_ptr, pid, B, PB, cp_rank, CP_SIZE)
        freqs = tl.load(freqs_ptr + pos * D + tl.arange(0, D)).to(tl.float32)
        cos = tl.cos(freqs) * mscale
        sin = tl.sin(freqs) * mscale

    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        k_ptr = qkv_ptr + DD * w
    else:
        row_offs = tl.arange(0, h)
        k_ptr = qkv_ptr + DD * H

    k0 = tl.load(
        k_ptr + pid * stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :]
    ).to(tl.float32)
    k1 = tl.load(
        k_ptr + pid * stride + D + DD * row_offs[:, None] + tl.arange(0, D)[None, :]
    ).to(tl.float32)

    if SILU:
        k0 = k0 * tl.sigmoid(k0)
        k1 = k1 * tl.sigmoid(k1)
    rms = tl.rsqrt((tl.sum(k0 * k0, 1) + tl.sum(k1 * k1, 1)) / DD + eps)
    k1 *= rms[:, None]
    k1 *= k_weight_1
    tl.store(
        ko_ptr
        + pid * h * DD
        + D
        + DD * tl.arange(0, h)[:, None]
        + tl.arange(0, D)[None, :],
        k1,
    )

    k0 *= rms[:, None]
    k0 *= k_weight_0
    kr = tl.reshape(
        tl.permute(
            tl.flip(tl.permute(tl.reshape(k0, (h, 2, d)), (0, 2, 1)), dim=2) * signs,
            (0, 2, 1),
        ),
        (h, D),
    )
    k0 = k0 * cos + kr * sin
    tl.store(
        ko_ptr
        + pid * h * DD
        + DD * tl.arange(0, h)[:, None]
        + tl.arange(0, D)[None, :],
        k0,
    )

    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        v_ptr = qkv_ptr + DD * w + DD
    else:
        row_offs = tl.arange(0, h)
        v_ptr = qkv_ptr + DD * H + DD * h

    v0 = tl.load(
        v_ptr + pid * stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :]
    ).to(tl.float32)
    v1 = tl.load(
        v_ptr + pid * stride + D + DD * row_offs[:, None] + tl.arange(0, D)[None, :]
    ).to(tl.float32)

    if SILU:
        v0 = v0 * tl.sigmoid(v0)
        v1 = v1 * tl.sigmoid(v1)

    tl.store(
        vo_ptr
        + pid * h * DD
        + DD * tl.arange(0, h)[:, None]
        + tl.arange(0, D)[None, :],
        v0,
    )
    tl.store(
        vo_ptr
        + pid * h * DD
        + D
        + DD * tl.arange(0, h)[:, None]
        + tl.arange(0, D)[None, :],
        v1,
    )


@triton.jit
def compatible_varlen_qk_norm_and_half_rope_forward_kernel(
    qkv_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    freqs_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_kv_ptr,
    qo_ptr,
    ko_ptr,
    vo_ptr,
    stride,
    eps,
    mscale,
    cp_rank,
    B,
    PB: tl.constexpr,
    H: tl.constexpr,
    h: tl.constexpr,
    PH: tl.constexpr,
    ph: tl.constexpr,
    D: tl.constexpr,
    d: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    SILU: tl.constexpr,
    CP_SIZE: tl.constexpr,
    REUSE: tl.constexpr,
):
    pid = tl.program_id(0)

    pos = _get_varlen_token_idx(cu_seqlens_q_ptr, pid, B, PB, cp_rank, CP_SIZE)

    DD = D * 2

    freqs = tl.load(freqs_ptr + pos * D + tl.arange(0, D)).to(tl.float32)
    cos = tl.cos(freqs) * mscale
    sin = tl.sin(freqs) * mscale
    signs = tl.arange(0, 2).to(tl.float32) * 2 - 1

    q_weight_0 = tl.load(q_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    q_weight_1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)
    q_ptr = qkv_ptr
    w = H // h

    # [len, bs, q_head, head_dim] -> [bs, len, q_head, head_dim]
    if INTERLEAVED:
        row_offs = tl.arange(0, PH) + tl.arange(0, PH) // w * 2
        row_mask = row_offs[:, None] < (H + 2 * h)
    else:
        row_offs = tl.arange(0, H)
        row_mask = row_offs[:, None] < H
    q_mask = tl.arange(0, PH)[:, None] < H

    q0 = tl.load(
        q_ptr + pid * stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :],
        mask=row_mask,
    ).to(tl.float32)
    q1 = tl.load(
        q_ptr + pid * stride + D + DD * row_offs[:, None] + tl.arange(0, D)[None, :],
        mask=row_mask,
    ).to(tl.float32)

    if SILU:
        q0 = q0 * tl.sigmoid(q0)
        q1 = q1 * tl.sigmoid(q1)
    rms = tl.rsqrt((tl.sum(q0 * q0, 1) + tl.sum(q1 * q1, 1)) / DD + eps)
    q1 *= rms[:, None]
    q1 *= q_weight_1
    tl.store(
        qo_ptr
        + pid * H * DD
        + D
        + DD * tl.arange(0, PH)[:, None]
        + tl.arange(0, D)[None, :],
        q1,
        mask=q_mask,
    )

    q0 *= rms[:, None]
    q0 *= q_weight_0
    qr = tl.reshape(
        tl.permute(
            tl.flip(tl.permute(tl.reshape(q0, (PH, 2, d)), (0, 2, 1)), dim=2) * signs,
            (0, 2, 1),
        ),
        (PH, D),
    )
    q0 = q0 * cos + qr * sin
    tl.store(
        qo_ptr
        + pid * H * DD
        + DD * tl.arange(0, PH)[:, None]
        + tl.arange(0, D)[None, :],
        q0,
        mask=q_mask,
    )

    k_weight_0 = tl.load(k_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    k_weight_1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)

    if not REUSE:
        pos = _get_varlen_token_idx(cu_seqlens_kv_ptr, pid, B, PB, cp_rank, CP_SIZE)
        freqs = tl.load(freqs_ptr + pos * D + tl.arange(0, D)).to(tl.float32)
        cos = tl.cos(freqs) * mscale
        sin = tl.sin(freqs) * mscale

    if INTERLEAVED:
        row_offs = tl.arange(0, ph) * (w + 2)
        k_ptr = qkv_ptr + DD * w
        row_mask = row_offs[:, None] < (h * (w + 2))
    else:
        row_offs = tl.arange(0, ph)
        k_ptr = qkv_ptr + DD * H
        row_mask = tl.arange(0, ph)[:, None] < h

    k0 = tl.load(
        k_ptr + pid * stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :],
        mask=row_mask,
    ).to(tl.float32)
    k1 = tl.load(
        k_ptr + pid * stride + D + DD * row_offs[:, None] + tl.arange(0, D)[None, :],
        mask=row_mask,
    ).to(tl.float32)

    if SILU:
        k0 = k0 * tl.sigmoid(k0)
        k1 = k1 * tl.sigmoid(k1)
    rms = tl.rsqrt((tl.sum(k0 * k0, 1) + tl.sum(k1 * k1, 1)) / DD + eps)
    k1 *= rms[:, None]
    k1 *= k_weight_1
    k_mask = tl.arange(0, ph)[:, None] < h
    tl.store(
        ko_ptr
        + pid * h * DD
        + D
        + DD * tl.arange(0, ph)[:, None]
        + tl.arange(0, D)[None, :],
        k1,
        mask=k_mask,
    )

    k0 *= rms[:, None]
    k0 *= k_weight_0
    kr = tl.reshape(
        tl.permute(
            tl.flip(tl.permute(tl.reshape(k0, (ph, 2, d)), (0, 2, 1)), dim=2) * signs,
            (0, 2, 1),
        ),
        (ph, D),
    )
    k0 = k0 * cos + kr * sin
    tl.store(
        ko_ptr
        + pid * h * DD
        + DD * tl.arange(0, ph)[:, None]
        + tl.arange(0, D)[None, :],
        k0,
        mask=k_mask,
    )

    if INTERLEAVED:
        row_offs = tl.arange(0, ph) * (w + 2)
        row_mask = row_offs[:, None] < (h * (w + 2))
        v_ptr = qkv_ptr + DD * w + DD
    else:
        row_offs = tl.arange(0, ph)
        row_mask = tl.arange(0, ph)[:, None] < h
        v_ptr = qkv_ptr + DD * H + DD * h

    v0 = tl.load(
        v_ptr + pid * stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :],
        mask=row_mask,
    ).to(tl.float32)
    v1 = tl.load(
        v_ptr + pid * stride + D + DD * row_offs[:, None] + tl.arange(0, D)[None, :],
        mask=row_mask,
    ).to(tl.float32)

    if SILU:
        v0 = v0 * tl.sigmoid(v0)
        v1 = v1 * tl.sigmoid(v1)

    v_mask = tl.arange(0, ph)[:, None] < h
    tl.store(
        vo_ptr
        + pid * h * DD
        + DD * tl.arange(0, ph)[:, None]
        + tl.arange(0, D)[None, :],
        v0,
        mask=v_mask,
    )
    tl.store(
        vo_ptr
        + pid * h * DD
        + D
        + DD * tl.arange(0, ph)[:, None]
        + tl.arange(0, D)[None, :],
        v1,
        mask=v_mask,
    )


def triton_varlen_qk_norm_and_half_rope_forward(
    qkv,
    q_norm_weight,
    k_norm_weight,
    freqs,
    cu_seqlens_q,
    cu_seqlens_kv,
    H=32,
    h=4,
    eps=1e-6,
    interleaved=True,
    silu=False,
    cp_rank=0,
    cp_size=1,
    mscale=1.0,
    reuse=False,
):
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
        silu: apply silu on qkv before qk norm and rope
    Returns:
        - qo: shape [B, S, H, head_dim]
        - ko: shape [B, S, h, head_dim]
        - vo: shape [B, S, h, head_dim]
    """
    assert qkv.is_contiguous() and q_norm_weight.is_contiguous()
    assert k_norm_weight.is_contiguous() and freqs.is_contiguous()
    T, Dim = qkv.shape
    stride = qkv.stride(0)  # qkv may be a slice of a tensor
    D = Dim // (H + 2 * h)
    B = cu_seqlens_q.size(0) - 1
    PB = max(triton.next_power_of_2(B), 128)  # reduce jit
    dtype = qkv.dtype
    device = qkv.device
    qo = torch.empty((T, H, D), dtype=dtype, device=device)
    ko = torch.empty((T, h, D), dtype=dtype, device=device)
    vo = torch.empty((T, h, D), dtype=dtype, device=device)

    num_stages = 5
    num_warps = 2
    grid = (T,)

    PH = triton.next_power_of_2(H)
    ph = triton.next_power_of_2(h)

    if PH == H and ph == h:
        varlen_qk_norm_and_half_rope_forward_kernel[grid](
            qkv,
            q_norm_weight,
            k_norm_weight,
            freqs,
            cu_seqlens_q,
            cu_seqlens_kv,
            qo,
            ko,
            vo,
            stride,
            eps,
            mscale,
            cp_rank,
            B,
            PB,
            H,
            h,
            D // 2,
            D // 4,
            interleaved,
            silu,
            cp_size,
            reuse,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    else:
        compatible_varlen_qk_norm_and_half_rope_forward_kernel[grid](
            qkv,
            q_norm_weight,
            k_norm_weight,
            freqs,
            cu_seqlens_q,
            cu_seqlens_kv,
            qo,
            ko,
            vo,
            stride,
            eps,
            mscale,
            cp_rank,
            B,
            PB,
            H,
            h,
            PH,
            ph,
            D // 2,
            D // 4,
            interleaved,
            silu,
            cp_size,
            reuse,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    return qo, ko, vo


@triton.jit
def varlen_qk_norm_and_half_rope_backward_kernel(
    gq_ptr,
    gk_ptr,
    gv_ptr,
    qkv_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    freqs_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_kv_ptr,
    dqkv_ptr,
    dqw_ptr,
    dkw_ptr,
    B,
    stride,
    grad_stride,
    eps,
    mscale,
    cp_rank,
    PB: tl.constexpr,
    H: tl.constexpr,
    h: tl.constexpr,
    D: tl.constexpr,
    d: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    SILU: tl.constexpr,
    CP_SIZE: tl.constexpr,
    REUSE: tl.constexpr,
):
    pid = tl.program_id(0)
    DD = 2 * D
    w = H // h

    pos = _get_varlen_token_idx(cu_seqlens_q_ptr, pid, B, PB, cp_rank, CP_SIZE)

    freqs = tl.load(freqs_ptr + pos * D + tl.arange(0, D)).to(tl.float32)
    cos = tl.cos(freqs) * mscale
    sin = tl.sin(freqs) * mscale
    signs = -tl.arange(0, 2).to(tl.float32) * 2 + 1

    q_w0 = tl.load(q_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    q_w1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)

    dqw_0 = tl.zeros((D,), dtype=tl.float32)
    dqw_1 = tl.zeros((D,), dtype=tl.float32)
    q_ptr = qkv_ptr
    dq_ptr = dqkv_ptr
    # [bs, len, q_head, head_dim] -> [len, bs, q_head, head_dim]
    if INTERLEAVED:
        row_offs = tl.arange(0, H) + tl.arange(0, H) // w * 2
    else:
        row_offs = tl.arange(0, H)

    gq_0 = tl.load(
        gq_ptr + pid * H * DD + DD * tl.arange(0, H)[:, None] + tl.arange(0, D)[None, :]
    ).to(tl.float32)
    gq_1 = tl.load(
        gq_ptr
        + pid * H * DD
        + D
        + DD * tl.arange(0, H)[:, None]
        + tl.arange(0, D)[None, :]
    ).to(tl.float32)

    gq_r = tl.reshape(
        tl.permute(
            tl.flip(tl.permute(tl.reshape(gq_0, (H, 2, d)), (0, 2, 1)), dim=2) * signs,
            (0, 2, 1),
        ),
        (H, D),
    )
    gq_0 = gq_0 * cos + gq_r * sin

    q0 = tl.load(
        q_ptr + pid * stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :]
    ).to(tl.float32)
    q1 = tl.load(
        q_ptr + pid * stride + D + DD * row_offs[:, None] + tl.arange(0, D)[None, :]
    ).to(tl.float32)

    if SILU:
        s0 = tl.sigmoid(q0)
        s1 = tl.sigmoid(q1)
        q_0 = q0 * s0
        q_1 = q1 * s1

        r = tl.rsqrt((tl.sum(q_0 * q_0, 1) + tl.sum(q_1 * q_1, 1)) / DD + eps)[:, None]

        dqw_0 += tl.sum(q_0 * gq_0 * r, 0)
        dqw_1 += tl.sum(q_1 * gq_1 * r, 0)

        s = tl.sum(q_0 * gq_0 * q_w0, 1) + tl.sum(q_1 * gq_1 * q_w1, 1)

        dq_0 = r * gq_0 * q_w0 - r * r * r / DD * q_0 * s[:, None]
        dq_1 = r * gq_1 * q_w1 - r * r * r / DD * q_1 * s[:, None]

        dq_0 = dq_0 * s0 * (1 + q0 * (1 - s0))
        dq_1 = dq_1 * s1 * (1 + q1 * (1 - s1))

    else:
        r = tl.rsqrt((tl.sum(q0 * q0, 1) + tl.sum(q1 * q1, 1)) / DD + eps)[:, None]

        dqw_0 += tl.sum(q0 * gq_0 * r, 0)
        dqw_1 += tl.sum(q1 * gq_1 * r, 0)

        s = tl.sum(q0 * gq_0 * q_w0, 1) + tl.sum(q1 * gq_1 * q_w1, 1)

        dq_0 = r * gq_0 * q_w0 - r * r * r / DD * q0 * s[:, None]
        dq_1 = r * gq_1 * q_w1 - r * r * r / DD * q1 * s[:, None]

    tl.store(
        dq_ptr + pid * grad_stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :],
        dq_0,
    )
    tl.store(
        dq_ptr
        + pid * grad_stride
        + D
        + DD * row_offs[:, None]
        + tl.arange(0, D)[None, :],
        dq_1,
    )

    tl.store(dqw_ptr + pid * D * 2 + tl.arange(0, D), dqw_0)
    tl.store(dqw_ptr + pid * D * 2 + D + tl.arange(0, D), dqw_1)

    if not REUSE:
        pos = _get_varlen_token_idx(cu_seqlens_kv_ptr, pid, B, PB, cp_rank, CP_SIZE)
        freqs = tl.load(freqs_ptr + pos * D + tl.arange(0, D)).to(tl.float32)
        cos = tl.cos(freqs) * mscale
        sin = tl.sin(freqs) * mscale

    k_w0 = tl.load(k_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    k_w1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)

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

    gk_0 = tl.load(
        gk_ptr + pid * h * DD + DD * tl.arange(0, h)[:, None] + tl.arange(0, D)[None, :]
    ).to(tl.float32)
    gk_1 = tl.load(
        gk_ptr
        + pid * h * DD
        + D
        + DD * tl.arange(0, h)[:, None]
        + tl.arange(0, D)[None, :]
    ).to(tl.float32)

    gk_r = tl.reshape(
        tl.permute(
            tl.flip(tl.permute(tl.reshape(gk_0, (h, 2, d)), (0, 2, 1)), dim=2) * signs,
            (0, 2, 1),
        ),
        (h, D),
    )
    gk_0 = gk_0 * cos + gk_r * sin

    k0 = tl.load(
        k_ptr + pid * stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :]
    ).to(tl.float32)
    k1 = tl.load(
        k_ptr + pid * stride + D + DD * row_offs[:, None] + tl.arange(0, D)[None, :]
    ).to(tl.float32)

    if SILU:

        s0 = tl.sigmoid(k0)
        s1 = tl.sigmoid(k1)
        k_0 = k0 * s0
        k_1 = k1 * s1

        r = tl.rsqrt((tl.sum(k_0 * k_0, 1) + tl.sum(k_1 * k_1, 1)) / DD + eps)[:, None]

        dkw_0 += tl.sum(k_0 * gk_0 * r, 0)
        dkw_1 += tl.sum(k_1 * gk_1 * r, 0)

        s = tl.sum(k_0 * gk_0 * k_w0, 1) + tl.sum(k_1 * gk_1 * k_w1, 1)

        dk_0 = r * gk_0 * k_w0 - r * r * r / DD * k_0 * s[:, None]
        dk_1 = r * gk_1 * k_w1 - r * r * r / DD * k_1 * s[:, None]

        dk_0 = dk_0 * s0 * (1 + k0 * (1 - s0))
        dk_1 = dk_1 * s1 * (1 + k1 * (1 - s1))

    else:
        r = tl.rsqrt((tl.sum(k0 * k0, 1) + tl.sum(k1 * k1, 1)) / DD + eps)[:, None]

        dkw_0 += tl.sum(k0 * gk_0 * r, 0)
        dkw_1 += tl.sum(k1 * gk_1 * r, 0)

        s = tl.sum(k0 * gk_0 * k_w0, 1) + tl.sum(k1 * gk_1 * k_w1, 1)

        dk_0 = r * gk_0 * k_w0 - r * r * r / DD * k0 * s[:, None]
        dk_1 = r * gk_1 * k_w1 - r * r * r / DD * k1 * s[:, None]

    tl.store(
        dk_ptr + pid * grad_stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :],
        dk_0,
    )
    tl.store(
        dk_ptr
        + pid * grad_stride
        + D
        + DD * row_offs[:, None]
        + tl.arange(0, D)[None, :],
        dk_1,
    )

    tl.store(dkw_ptr + pid * D * 2 + tl.arange(0, D), dkw_0)
    tl.store(dkw_ptr + pid * D * 2 + D + tl.arange(0, D), dkw_1)

    # [t, k_head, head_dim] -> [t, k_head + 2 * kv_head, head_dim]
    if INTERLEAVED:
        row_offs = tl.arange(0, h) * (w + 2)
        v_ptr = qkv_ptr + DD * w + DD
        dv_ptr = dqkv_ptr + DD * w + DD
    else:
        row_offs = tl.arange(0, h)
        v_ptr = qkv_ptr + DD * H + DD * h
        dv_ptr = dqkv_ptr + DD * H + DD * h

    gv_0 = tl.load(
        gv_ptr + pid * h * DD + DD * tl.arange(0, h)[:, None] + tl.arange(0, D)[None, :]
    ).to(tl.float32)
    gv_1 = tl.load(
        gv_ptr
        + pid * h * DD
        + D
        + DD * tl.arange(0, h)[:, None]
        + tl.arange(0, D)[None, :]
    ).to(tl.float32)

    if SILU:
        v0 = tl.load(
            v_ptr + pid * stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :]
        ).to(tl.float32)
        v1 = tl.load(
            v_ptr + pid * stride + D + DD * row_offs[:, None] + tl.arange(0, D)[None, :]
        ).to(tl.float32)

        s0 = tl.sigmoid(v0)
        s1 = tl.sigmoid(v1)
        dv_0 = gv_0 * s0 * (1 + v0 * (1 - s0))
        dv_1 = gv_1 * s1 * (1 + v1 * (1 - s1))
    else:
        dv_0 = gv_0
        dv_1 = gv_1

    tl.store(
        dv_ptr + pid * grad_stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :],
        dv_0,
    )
    tl.store(
        dv_ptr
        + pid * grad_stride
        + D
        + DD * row_offs[:, None]
        + tl.arange(0, D)[None, :],
        dv_1,
    )


@triton.jit
def compatible_varlen_qk_norm_and_half_rope_backward_kernel(
    gq_ptr,
    gk_ptr,
    gv_ptr,
    qkv_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    freqs_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_kv_ptr,
    dqkv_ptr,
    dqw_ptr,
    dkw_ptr,
    B,
    stride,
    grad_stride,
    eps,
    mscale,
    cp_rank,
    PB: tl.constexpr,
    H: tl.constexpr,
    h: tl.constexpr,
    PH: tl.constexpr,
    ph: tl.constexpr,
    D: tl.constexpr,
    d: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    SILU: tl.constexpr,
    CP_SIZE: tl.constexpr,
    REUSE: tl.constexpr,
):
    pid = tl.program_id(0)
    DD = 2 * D
    w = H // h

    pos = _get_varlen_token_idx(cu_seqlens_q_ptr, pid, B, PB, cp_rank, CP_SIZE)

    freqs = tl.load(freqs_ptr + pos * D + tl.arange(0, D)).to(tl.float32)
    cos = tl.cos(freqs) * mscale
    sin = tl.sin(freqs) * mscale
    signs = -tl.arange(0, 2).to(tl.float32) * 2 + 1

    q_w0 = tl.load(q_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    q_w1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)

    dqw_0 = tl.zeros((D,), dtype=tl.float32)
    dqw_1 = tl.zeros((D,), dtype=tl.float32)
    q_ptr = qkv_ptr
    dq_ptr = dqkv_ptr
    # [bs, len, q_head, head_dim] -> [len, bs, q_head, head_dim]
    if INTERLEAVED:
        row_offs = tl.arange(0, PH) + tl.arange(0, PH) // w * 2
        row_mask = row_offs[:, None] < (H + 2 * h)
    else:
        row_offs = tl.arange(0, PH)
        row_mask = row_offs[:, None] < H

    gq_0 = tl.load(
        gq_ptr
        + pid * H * DD
        + DD * tl.arange(0, PH)[:, None]
        + tl.arange(0, D)[None, :],
        mask=tl.arange(0, PH)[:, None] < H,
    ).to(tl.float32)
    gq_1 = tl.load(
        gq_ptr
        + pid * H * DD
        + D
        + DD * tl.arange(0, PH)[:, None]
        + tl.arange(0, D)[None, :],
        mask=tl.arange(0, PH)[:, None] < H,
    ).to(tl.float32)

    gq_r = tl.reshape(
        tl.permute(
            tl.flip(tl.permute(tl.reshape(gq_0, (PH, 2, d)), (0, 2, 1)), dim=2) * signs,
            (0, 2, 1),
        ),
        (PH, D),
    )
    gq_0 = gq_0 * cos + gq_r * sin

    q0 = tl.load(
        q_ptr + pid * stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :],
        mask=row_mask,
    ).to(tl.float32)
    q1 = tl.load(
        q_ptr + pid * stride + D + DD * row_offs[:, None] + tl.arange(0, D)[None, :],
        mask=row_mask,
    ).to(tl.float32)

    if SILU:
        s0 = tl.sigmoid(q0)
        s1 = tl.sigmoid(q1)
        q_0 = q0 * s0
        q_1 = q1 * s1

        r = tl.rsqrt((tl.sum(q_0 * q_0, 1) + tl.sum(q_1 * q_1, 1)) / DD + eps)[:, None]

        dqw_0 += tl.sum(q_0 * gq_0 * r, 0)
        dqw_1 += tl.sum(q_1 * gq_1 * r, 0)

        s = tl.sum(q_0 * gq_0 * q_w0, 1) + tl.sum(q_1 * gq_1 * q_w1, 1)

        dq_0 = r * gq_0 * q_w0 - r * r * r / DD * q_0 * s[:, None]
        dq_1 = r * gq_1 * q_w1 - r * r * r / DD * q_1 * s[:, None]

        dq_0 = dq_0 * s0 * (1 + q0 * (1 - s0))
        dq_1 = dq_1 * s1 * (1 + q1 * (1 - s1))

    else:
        r = tl.rsqrt((tl.sum(q0 * q0, 1) + tl.sum(q1 * q1, 1)) / DD + eps)[:, None]

        dqw_0 += tl.sum(q0 * gq_0 * r, 0)
        dqw_1 += tl.sum(q1 * gq_1 * r, 0)

        s = tl.sum(q0 * gq_0 * q_w0, 1) + tl.sum(q1 * gq_1 * q_w1, 1)

        dq_0 = r * gq_0 * q_w0 - r * r * r / DD * q0 * s[:, None]
        dq_1 = r * gq_1 * q_w1 - r * r * r / DD * q1 * s[:, None]

    tl.store(
        dq_ptr + pid * grad_stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :],
        dq_0,
        mask=row_mask,
    )
    tl.store(
        dq_ptr
        + pid * grad_stride
        + D
        + DD * row_offs[:, None]
        + tl.arange(0, D)[None, :],
        dq_1,
        mask=row_mask,
    )

    tl.store(dqw_ptr + pid * D * 2 + tl.arange(0, D), dqw_0)
    tl.store(dqw_ptr + pid * D * 2 + D + tl.arange(0, D), dqw_1)

    if not REUSE:
        pos = _get_varlen_token_idx(cu_seqlens_kv_ptr, pid, B, PB, cp_rank, CP_SIZE)
        freqs = tl.load(freqs_ptr + pos * D + tl.arange(0, D)).to(tl.float32)
        cos = tl.cos(freqs) * mscale
        sin = tl.sin(freqs) * mscale

    k_w0 = tl.load(k_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    k_w1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)

    dkw_0 = tl.zeros((D,), dtype=tl.float32)
    dkw_1 = tl.zeros((D,), dtype=tl.float32)
    if INTERLEAVED:
        row_offs = tl.arange(0, ph) * (w + 2)
        row_mask = row_offs[:, None] < (h * (w + 2))
        k_ptr = qkv_ptr + DD * w
        dk_ptr = dqkv_ptr + DD * w
    else:
        row_offs = tl.arange(0, ph)
        row_mask = row_offs[:, None] < h
        k_ptr = qkv_ptr + DD * H
        dk_ptr = dqkv_ptr + DD * H

    gk_0 = tl.load(
        gk_ptr
        + pid * h * DD
        + DD * tl.arange(0, ph)[:, None]
        + tl.arange(0, D)[None, :],
        mask=tl.arange(0, ph)[:, None] < h,
    ).to(tl.float32)
    gk_1 = tl.load(
        gk_ptr
        + pid * h * DD
        + D
        + DD * tl.arange(0, ph)[:, None]
        + tl.arange(0, D)[None, :],
        mask=tl.arange(0, ph)[:, None] < h,
    ).to(tl.float32)

    gk_r = tl.reshape(
        tl.permute(
            tl.flip(tl.permute(tl.reshape(gk_0, (ph, 2, d)), (0, 2, 1)), dim=2) * signs,
            (0, 2, 1),
        ),
        (ph, D),
    )
    gk_0 = gk_0 * cos + gk_r * sin

    k0 = tl.load(
        k_ptr + pid * stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :],
        mask=row_mask,
    ).to(tl.float32)
    k1 = tl.load(
        k_ptr + pid * stride + D + DD * row_offs[:, None] + tl.arange(0, D)[None, :],
        mask=row_mask,
    ).to(tl.float32)

    if SILU:

        s0 = tl.sigmoid(k0)
        s1 = tl.sigmoid(k1)
        k_0 = k0 * s0
        k_1 = k1 * s1

        r = tl.rsqrt((tl.sum(k_0 * k_0, 1) + tl.sum(k_1 * k_1, 1)) / DD + eps)[:, None]

        dkw_0 += tl.sum(k_0 * gk_0 * r, 0)
        dkw_1 += tl.sum(k_1 * gk_1 * r, 0)

        s = tl.sum(k_0 * gk_0 * k_w0, 1) + tl.sum(k_1 * gk_1 * k_w1, 1)

        dk_0 = r * gk_0 * k_w0 - r * r * r / DD * k_0 * s[:, None]
        dk_1 = r * gk_1 * k_w1 - r * r * r / DD * k_1 * s[:, None]

        dk_0 = dk_0 * s0 * (1 + k0 * (1 - s0))
        dk_1 = dk_1 * s1 * (1 + k1 * (1 - s1))

    else:
        r = tl.rsqrt((tl.sum(k0 * k0, 1) + tl.sum(k1 * k1, 1)) / DD + eps)[:, None]

        dkw_0 += tl.sum(k0 * gk_0 * r, 0)
        dkw_1 += tl.sum(k1 * gk_1 * r, 0)

        s = tl.sum(k0 * gk_0 * k_w0, 1) + tl.sum(k1 * gk_1 * k_w1, 1)

        dk_0 = r * gk_0 * k_w0 - r * r * r / DD * k0 * s[:, None]
        dk_1 = r * gk_1 * k_w1 - r * r * r / DD * k1 * s[:, None]

    tl.store(
        dk_ptr + pid * grad_stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :],
        dk_0,
        mask=row_mask,
    )
    tl.store(
        dk_ptr
        + pid * grad_stride
        + D
        + DD * row_offs[:, None]
        + tl.arange(0, D)[None, :],
        dk_1,
        mask=row_mask,
    )

    tl.store(dkw_ptr + pid * D * 2 + tl.arange(0, D), dkw_0)
    tl.store(dkw_ptr + pid * D * 2 + D + tl.arange(0, D), dkw_1)

    # [t, k_head, head_dim] -> [t, k_head + 2 * kv_head, head_dim]
    if INTERLEAVED:
        row_offs = tl.arange(0, ph) * (w + 2)
        row_mask = row_offs[:, None] < (h * (w + 2))
        v_ptr = qkv_ptr + DD * w + DD
        dv_ptr = dqkv_ptr + DD * w + DD
    else:
        row_offs = tl.arange(0, ph)
        row_mask = row_offs[:, None] < h
        v_ptr = qkv_ptr + DD * H + DD * h
        dv_ptr = dqkv_ptr + DD * H + DD * h

    gv_0 = tl.load(
        gv_ptr
        + pid * h * DD
        + DD * tl.arange(0, ph)[:, None]
        + tl.arange(0, D)[None, :],
        mask=tl.arange(0, ph)[:, None] < h,
    ).to(tl.float32)
    gv_1 = tl.load(
        gv_ptr
        + pid * h * DD
        + D
        + DD * tl.arange(0, ph)[:, None]
        + tl.arange(0, D)[None, :],
        mask=tl.arange(0, ph)[:, None] < h,
    ).to(tl.float32)

    if SILU:
        v0 = tl.load(
            v_ptr + pid * stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :],
            mask=row_mask,
        ).to(tl.float32)
        v1 = tl.load(
            v_ptr
            + pid * stride
            + D
            + DD * row_offs[:, None]
            + tl.arange(0, D)[None, :],
            mask=row_mask,
        ).to(tl.float32)

        s0 = tl.sigmoid(v0)
        s1 = tl.sigmoid(v1)
        dv_0 = gv_0 * s0 * (1 + v0 * (1 - s0))
        dv_1 = gv_1 * s1 * (1 + v1 * (1 - s1))
    else:
        dv_0 = gv_0
        dv_1 = gv_1

    tl.store(
        dv_ptr + pid * grad_stride + DD * row_offs[:, None] + tl.arange(0, D)[None, :],
        dv_0,
        mask=row_mask,
    )
    tl.store(
        dv_ptr
        + pid * grad_stride
        + D
        + DD * row_offs[:, None]
        + tl.arange(0, D)[None, :],
        dv_1,
        mask=row_mask,
    )


def triton_varlen_qk_norm_and_half_rope_backward(
    gq,
    gk,
    gv,
    qkv,
    q_norm_weight,
    k_norm_weight,
    freqs,
    cu_seqlens_q,
    cu_seqlens_kv,
    eps=1e-6,
    interleaved=True,
    silu=False,
    cp_rank=0,
    cp_size=1,
    mscale=1.0,
    reuse=False,
):
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
        silu: whether silu is applied to qkv

    Returns:
        - dqkv: gradient of qkv
        - dqw: gradient of q_norm_weight
        - dkw: gradient of k_norm_weight
    """
    assert gq.is_contiguous() and gk.is_contiguous() and gv.is_contiguous()
    T, H, D = gq.shape
    stride = qkv.stride(0)
    h = gk.shape[1]
    B = cu_seqlens_q.size(0) - 1
    PB = max(triton.next_power_of_2(B), 128)
    num_stages = 5
    num_warps = 1

    dtype = gq.dtype
    device = gq.device
    dqkv = torch.empty((T, (H + 2 * h) * D), dtype=dtype, device=device)
    grad_stride = dqkv.stride(0)  # for potential fused kernel

    tmp_dqw = torch.empty((T, D), dtype=torch.float32, device=device)
    tmp_dkw = torch.empty((T, D), dtype=torch.float32, device=device)

    grid = (T,)

    PH = triton.next_power_of_2(H)
    ph = triton.next_power_of_2(h)

    if PH == H and ph == h:
        varlen_qk_norm_and_half_rope_backward_kernel[grid](
            gq,
            gk,
            gv,
            qkv,
            q_norm_weight,
            k_norm_weight,
            freqs,
            cu_seqlens_q,
            cu_seqlens_kv,
            dqkv,
            tmp_dqw,
            tmp_dkw,
            B,
            stride,
            grad_stride,
            eps,
            mscale,
            cp_rank,
            PB,
            H,
            h,
            D // 2,
            D // 4,
            interleaved,
            silu,
            cp_size,
            reuse,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    else:
        compatible_varlen_qk_norm_and_half_rope_backward_kernel[grid](
            gq,
            gk,
            gv,
            qkv,
            q_norm_weight,
            k_norm_weight,
            freqs,
            cu_seqlens_q,
            cu_seqlens_kv,
            dqkv,
            tmp_dqw,
            tmp_dkw,
            B,
            stride,
            grad_stride,
            eps,
            mscale,
            cp_rank,
            PB,
            H,
            h,
            PH,
            ph,
            D // 2,
            D // 4,
            interleaved,
            silu,
            cp_size,
            reuse,
            num_stages=num_stages,
            num_warps=num_warps,
        )
    dqw = tmp_dqw.sum(0)
    dkw = tmp_dkw.sum(0)
    return dqkv, dqw, dkw


@triton.jit
def mla_rope_forward_kernel(
    q_ptr,
    kv_ptr,
    k_pos_emb_ptr,
    freqs_ptr,
    qo_ptr,
    ko_ptr,
    vo_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_kv_ptr,
    mscale,
    kpe_stride,
    B,
    cp_rank,
    PB: tl.constexpr,
    cp_size: tl.constexpr,
    H: tl.constexpr,
    VARLEN: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    REUSE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tokens = tl.num_programs(0)

    if VARLEN:
        pos = _get_varlen_token_idx(cu_seqlens_q_ptr, pid, B, PB, cp_rank, cp_size)
        L = num_tokens
        bid = 0
    else:
        pos = pid // B
        L = num_tokens // B
        bid = pid % B

    freqs = tl.load(freqs_ptr + pos * 64 + tl.arange(0, 64)).to(tl.float32)

    cos = tl.cos(freqs) * mscale
    sin = tl.sin(freqs) * mscale
    signs = tl.arange(0, 2).to(tl.float32) * 2 - 1

    q = tl.load(
        q_ptr
        + pid * H * 192
        + 128
        + 192 * tl.arange(0, H)[:, None]
        + tl.arange(0, 64)[None, :]
    ).to(tl.float32)
    qt = tl.permute(tl.reshape(q, (H, 32, 2)), (0, 2, 1))
    q = tl.reshape(qt, (H, 64))
    qr = tl.reshape(
        tl.permute(tl.flip(tl.permute(qt, (0, 2, 1)), dim=2) * signs, (0, 2, 1)),
        (H, 64),
    )
    q = q * cos + qr * sin

    # q0, q1 = tl.split(tl.reshape(q, (H, 32, 2)))
    # qo0 = q0 * cos0 - q1 * sin0
    # qo1 = q1 * cos1 + q0 * sin1
    if TRANSPOSE:
        # [L, B, H, D] -> [B, L, H, D]
        qn = tl.load(
            q_ptr
            + pid * H * 192
            + 192 * tl.arange(0, H)[:, None]
            + tl.arange(0, 128)[None, :]
        ).to(tl.float32)
        tl.store(
            qo_ptr
            + (bid * L + pos) * H * 192
            + 128
            + 192 * tl.arange(0, H)[:, None]
            + tl.arange(0, 64)[None, :],
            q,
        )
        tl.store(
            qo_ptr
            + (bid * L + pos) * H * 192
            + 192 * tl.arange(0, H)[:, None]
            + tl.arange(0, 128)[None, :],
            qn,
        )
    else:
        tl.store(
            q_ptr
            + pid * H * 192
            + 128
            + 192 * tl.arange(0, H)[:, None]
            + tl.arange(0, 64)[None, :],
            q,
        )

    k = tl.load(k_pos_emb_ptr + pid * kpe_stride + tl.arange(0, 64)).to(tl.float32)

    if VARLEN and not REUSE:
        pos = _get_varlen_token_idx(cu_seqlens_kv_ptr, pid, B, PB, cp_rank, cp_size)
        freqs = tl.load(freqs_ptr + pos * 64 + tl.arange(0, 64))
        cos = tl.cos(freqs) * mscale
        sin = tl.sin(freqs) * mscale

    kt = tl.permute(tl.reshape(k, (32, 2)), (1, 0))
    k = tl.reshape(kt, (64,))
    kr = tl.reshape(
        tl.permute(tl.flip(tl.permute(kt, (1, 0)), dim=1) * signs, (1, 0)), (64,)
    )
    k = k * cos + kr * sin
    if TRANSPOSE:
        tl.store(
            ko_ptr
            + (bid * L + pos) * H * 192
            + 128
            + 192 * tl.arange(0, H)[:, None]
            + tl.arange(0, 64)[None, :],
            k[None, :],
        )
    else:
        tl.store(
            ko_ptr
            + pid * H * 192
            + 128
            + 192 * tl.arange(0, H)[:, None]
            + tl.arange(0, 64)[None, :],
            k[None, :],
        )

    k = tl.load(
        kv_ptr
        + pid * H * 256
        + 256 * tl.arange(0, H)[:, None]
        + tl.arange(0, 128)[None, :]
    ).to(tl.float32)
    if TRANSPOSE:
        tl.store(
            ko_ptr
            + (bid * L + pos) * H * 192
            + 192 * tl.arange(0, H)[:, None]
            + tl.arange(0, 128)[None, :],
            k,
        )
    else:
        tl.store(
            ko_ptr
            + pid * H * 192
            + 192 * tl.arange(0, H)[:, None]
            + tl.arange(0, 128)[None, :],
            k,
        )

    v = tl.load(
        kv_ptr
        + pid * H * 256
        + 128
        + 256 * tl.arange(0, H)[:, None]
        + tl.arange(0, 128)[None, :]
    ).to(tl.float32)
    if TRANSPOSE:
        tl.store(
            vo_ptr
            + (bid * L + pos) * H * 128
            + 128 * tl.arange(0, H)[:, None]
            + tl.arange(0, 128)[None, :],
            v,
        )
    else:
        tl.store(
            vo_ptr
            + pid * H * 128
            + 128 * tl.arange(0, H)[:, None]
            + tl.arange(0, 128)[None, :],
            v,
        )


def triton_mla_rope_forward(
    q,
    kv,
    k_pos_emb,
    freqs,
    mscale=1.0,
    transpose=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    cp_rank=0,
    cp_size=1,
    reuse=False,
):
    """
    apply MLA-type rope to qkv
    Args:
        q: query tensor, [len, bs, n_heads, 192]
        kv: key-value tensor, [len, bs, n_heads, 256]
        k_pos_emb: k pos emb, [len, bs, 1, 64]
        freqs: rope freqs, [len, 64]
        mscale: mscale for rope
        transpose: whether transpose the output to [bs, len, n_heads, dim] layout
        cu_seqlens_q: accummulated query length
        cu_seqlens_kv: accummulated kv length
        cp_rank: rank of context parallel
        cp_size: size of context parallel

    Returns:
        - qo: inplace updated query, [len, bs, n_heads, 192] if not transpose
              else  [bs, len, n_heads, 192]
        - ko: key output, [len, bs, n_heads, 192] if not transpose
              else [bs, len, n_heads, 192]
        - vo: value output, [len, bs, n_heads, 128] if not transpose
              else [bs, len, n_heads, 128]
    """

    assert q.is_contiguous() and freqs.is_contiguous()
    VARLEN = cu_seqlens_q is not None

    dtype = q.dtype
    device = q.device
    if VARLEN:
        assert cu_seqlens_kv is not None
        N, H, D = q.shape
        B = cu_seqlens_q.shape[0] - 1
        PB = max(triton.next_power_of_2(B), 128)
        qo = None
        ko = torch.empty((N, H, 192), dtype=dtype, device=device)
        vo = torch.empty((N, H, 128), dtype=dtype, device=device)
        kpe_stride = k_pos_emb.stride(0)
    else:
        L, B, H, D = q.shape
        PB = 1
        if transpose:
            qo = torch.empty((B, L, H, 192), dtype=dtype, device=device)
            ko = torch.empty((B, L, H, 192), dtype=dtype, device=device)
            vo = torch.empty((B, L, H, 128), dtype=dtype, device=device)
        else:
            qo = None
            ko = torch.empty((L, B, H, 192), dtype=dtype, device=device)
            vo = torch.empty((L, B, H, 128), dtype=dtype, device=device)
        N = L * B
        kpe_stride = k_pos_emb.stride(0) if B == 1 else k_pos_emb.stride(1)
    assert D == 192 and kv.shape[-1] == 256 and k_pos_emb.shape[-1] == 64
    assert (
        kv.stride(-2) == 256 and k_pos_emb.stride(-2) == 64
    ), f"{kv.stride()=} {k_pos_emb.stride()=}"
    num_stages = 2
    num_warps = 2

    grid = (N,)
    mla_rope_forward_kernel[grid](
        q,
        kv,
        k_pos_emb,
        freqs,
        qo,
        ko,
        vo,
        cu_seqlens_q,
        cu_seqlens_kv,
        mscale,
        kpe_stride,
        B,
        cp_rank,
        PB,
        cp_size,
        H,
        VARLEN,
        False if VARLEN else transpose,
        reuse,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    if VARLEN or not transpose:
        qo = q
    return qo, ko, vo


@triton.jit
def mla_rope_backward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    freqs_ptr,
    dq_ptr,
    dkv_ptr,
    dp_ptr,
    cu_seqlens_q_ptr,
    cu_seqlens_kv_ptr,
    mscale,
    B,
    cp_rank,
    PB: tl.constexpr,
    cp_size: tl.constexpr,
    H: tl.constexpr,
    VARLEN: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    REUSE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tokens = tl.num_programs(0)

    if VARLEN:
        pos = _get_varlen_token_idx(cu_seqlens_q_ptr, pid, B, PB, cp_rank, cp_size)
        L = num_tokens
        bid = 0
    else:
        L = num_tokens // B
        if TRANSPOSED:
            # [B,L,H,D]
            pos = pid % L
            bid = pid // L
        else:
            # [L,B,H,D]
            pos = pid // B
            bid = pid % B

    freqs0 = tl.load(freqs_ptr + pos * 64 + tl.arange(0, 32)).to(tl.float32)
    freqs1 = tl.load(freqs_ptr + pos * 64 + 32 + tl.arange(0, 32)).to(tl.float32)

    q0 = tl.load(
        q_ptr
        + pid * H * 192
        + 128
        + 192 * tl.arange(0, H)[:, None]
        + tl.arange(0, 32)[None, :]
    ).to(tl.float32)
    q1 = tl.load(
        q_ptr
        + pid * H * 192
        + 160
        + 192 * tl.arange(0, H)[:, None]
        + tl.arange(0, 32)[None, :]
    ).to(tl.float32)

    cos0 = tl.cos(freqs0) * mscale
    sin0 = tl.sin(freqs0) * mscale

    cos1 = tl.cos(freqs1) * mscale
    sin1 = tl.sin(freqs1) * mscale

    dq0 = q0 * cos0 + q1 * sin1
    dq1 = q1 * cos1 - q0 * sin0
    dq = tl.reshape(tl.join(dq0, dq1), (H, 64))

    if TRANSPOSED:
        # [B,L,H,D] -> [L,B,H,D]
        dqn = tl.load(
            q_ptr
            + pid * H * 192
            + 192 * tl.arange(0, H)[:, None]
            + tl.arange(0, 128)[None, :]
        )
        tl.store(
            dq_ptr
            + (pos * B + bid) * H * 192
            + 128
            + 192 * tl.arange(0, H)[:, None]
            + tl.arange(0, 64)[None, :],
            dq,
        )
        tl.store(
            dq_ptr
            + (pos * B + bid) * H * 192
            + 192 * tl.arange(0, H)[:, None]
            + tl.arange(0, 128)[None, :],
            dqn,
        )
    else:
        tl.store(
            q_ptr
            + pid * H * 192
            + 128
            + 192 * tl.arange(0, H)[:, None]
            + tl.arange(0, 64)[None, :],
            dq,
        )

    # qr = tl.reshape(tl.permute(
    #     tl.flip(tl.permute(tl.reshape(q, (H, 2, 32)), (0, 2, 1)),
    #             dim=2) * signs, (0, 2, 1)), (H, 64))
    # q = q * cos + qr * sin
    # q = tl.reshape(tl.permute(tl.reshape(q, (H, 2, 32)), (0, 2, 1)), (H, 64))

    if VARLEN and not REUSE:
        pos = _get_varlen_token_idx(cu_seqlens_kv_ptr, pid, B, PB, cp_rank, cp_size)

        freqs0 = tl.load(freqs_ptr + pos * 64 + tl.arange(0, 32)).to(tl.float32)
        freqs1 = tl.load(freqs_ptr + pos * 64 + 32 + tl.arange(0, 32)).to(tl.float32)

        cos0 = tl.cos(freqs0) * mscale
        sin0 = tl.sin(freqs0) * mscale

        cos1 = tl.cos(freqs1) * mscale
        sin1 = tl.sin(freqs1) * mscale

    kp0 = tl.load(
        k_ptr
        + pid * H * 192
        + 128
        + 192 * tl.arange(0, H)[:, None]
        + tl.arange(0, 32)[None, :]
    ).to(tl.float32)
    kp1 = tl.load(
        k_ptr
        + pid * H * 192
        + 160
        + 192 * tl.arange(0, H)[:, None]
        + tl.arange(0, 32)[None, :]
    ).to(tl.float32)
    dkp0 = tl.sum(kp0 * cos0 + kp1 * sin1, 0)
    dkp1 = tl.sum(kp1 * cos1 - kp0 * sin0, 0)
    dkp = tl.reshape(tl.join(dkp0, dkp1), (64,))

    if TRANSPOSED:
        tl.store(dp_ptr + (pos * B + bid) * 64 + tl.arange(0, 64), dkp)
    else:
        tl.store(dp_ptr + pid * 64 + tl.arange(0, 64), dkp)

    k = tl.load(
        k_ptr
        + pid * H * 192
        + 192 * tl.arange(0, H)[:, None]
        + tl.arange(0, 128)[None, :]
    )
    if TRANSPOSED:
        tl.store(
            dkv_ptr
            + (pos * B + bid) * H * 256
            + 256 * tl.arange(0, H)[:, None]
            + tl.arange(0, 128)[None, :],
            k,
        )
    else:
        tl.store(
            dkv_ptr
            + pid * H * 256
            + 256 * tl.arange(0, H)[:, None]
            + tl.arange(0, 128)[None, :],
            k,
        )

    v = tl.load(
        v_ptr
        + pid * H * 128
        + 128 * tl.arange(0, H)[:, None]
        + tl.arange(0, 128)[None, :]
    )

    if TRANSPOSED:
        tl.store(
            dkv_ptr
            + (pos * B + bid) * H * 256
            + 128
            + 256 * tl.arange(0, H)[:, None]
            + tl.arange(0, 128)[None, :],
            v,
        )
    else:
        tl.store(
            dkv_ptr
            + pid * H * 256
            + 128
            + 256 * tl.arange(0, H)[:, None]
            + tl.arange(0, 128)[None, :],
            v,
        )


def triton_mla_rope_backward(
    q_grad,
    k_grad,
    v_grad,
    freqs,
    mscale=1.0,
    transposed=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    cp_rank=0,
    cp_size=1,
    reuse=False,
):
    assert q_grad.is_contiguous() and k_grad.is_contiguous() and v_grad.is_contiguous()
    VARLEN = cu_seqlens_q is not None

    dtype = q_grad.dtype
    device = q_grad.device
    if VARLEN:
        assert cu_seqlens_kv is not None
        N, H, D = q_grad.shape
        B = cu_seqlens_q.shape[0] - 1
        PB = max(triton.next_power_of_2(B), 128)
        assert B <= 128
        dq = None
        dkv = torch.empty((N, H, 256), dtype=dtype, device=device)
        dp = torch.empty((N, 1, 64), dtype=dtype, device=device)
    else:
        if transposed:
            B, L, H, D = q_grad.shape
            N = L * B
            dq = torch.empty((L, B, H, 192), dtype=dtype, device=device)
            dkv = torch.empty((L, B, H, 256), dtype=dtype, device=device)
            dp = torch.empty((L, B, 1, 64), dtype=dtype, device=device)
        else:
            L, B, H, D = q_grad.shape
            N = L * B
            dq = None
            dkv = torch.empty((L, B, H, 256), dtype=dtype, device=device)
            dp = torch.empty((L, B, 1, 64), dtype=dtype, device=device)
        PB = 1

    num_stages = 2
    num_warps = 4
    grid = (N,)
    mla_rope_backward_kernel[grid](
        q_grad,
        k_grad,
        v_grad,
        freqs,
        dq,
        dkv,
        dp,
        cu_seqlens_q,
        cu_seqlens_kv,
        mscale,
        B,
        cp_rank,
        PB,
        cp_size,
        H,
        VARLEN,
        False if VARLEN else transposed,
        reuse,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    if VARLEN or not transposed:
        dq = q_grad
    return dq, dkv, dp
