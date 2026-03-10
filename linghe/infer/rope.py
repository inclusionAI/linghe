# -*- coding: utf-8 -*-
import torch
import triton
import triton.language as tl



@triton.jit
def varlen_qk_norm_and_half_rope_kernel(qkv_ptr,
                                        q_norm_weight_ptr,
                                        k_norm_weight_ptr,
                                        freqs_ptr,
                                        position_ids,
                                        qo_ptr, ko_ptr, vo_ptr,
                                        stride,
                                        eps,
                                        linear_scale_value,
                                        H: tl.constexpr,
                                        h: tl.constexpr,
                                        PH: tl.constexpr,
                                        ph: tl.constexpr,
                                        D: tl.constexpr,
                                        d: tl.constexpr,
                                        INTERLEAVED: tl.constexpr,
                                        SILU: tl.constexpr,
                                        SCALE: tl.constexpr):
    pid = tl.program_id(0)

    pos = tl.load(position_ids + pid)

    DD = D * 2

    cos = tl.load(freqs_ptr + pos * D + tl.arange(0, D) % d).to(tl.float32)

    sin = tl.load(freqs_ptr + pos * D + d + tl.arange(0, D) % d).to(tl.float32)

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
        row_offs = tl.arange(0, PH)
        row_mask = row_offs[:, None] < H
    q_mask = tl.arange(0, PH)[:, None] < H

    q0 = tl.load(q_ptr
                 + pid * stride
                 + DD * row_offs[:, None]
                 + tl.arange(0, D)[None, :],
                 mask=row_mask).to(tl.float32)
    q1 = tl.load(q_ptr
                 + pid * stride
                 + D
                 + DD * row_offs[:, None]
                 + tl.arange(0, D)[None, :],
                 mask=row_mask).to(tl.float32)

    if SILU:
        q0 = q0 * tl.sigmoid(q0)
        q1 = q1 * tl.sigmoid(q1)
    rms = tl.rsqrt((tl.sum(q0 * q0, 1) + tl.sum(q1 * q1, 1)) / DD + eps)
    q1 *= rms[:, None]
    q1 *= q_weight_1

    if SCALE:
        q1 *= linear_scale_value

    tl.store(
        qo_ptr
        + pid * H * DD
        + D
        + DD * tl.arange(0, PH)[:, None]
        + tl.arange(0, D)[None, :],
        q1,
        mask=q_mask)

    q0 *= rms[:, None]
    q0 *= q_weight_0

    if SCALE:
        q0 *= linear_scale_value

    qr = tl.reshape(tl.permute(
        tl.flip(tl.permute(tl.reshape(q0,
                                      (PH, 2, d)),
                           (0, 2, 1)),
                dim=2) * signs,
        (0, 2, 1)),
        (PH, D))
    q0 = q0 * cos + qr * sin
    tl.store(
        qo_ptr
        + pid * H * DD
        + DD * tl.arange(0, PH)[:, None]
        + tl.arange(0, D)[None, :],
        q0,
        mask=q_mask)

    k_weight_0 = tl.load(k_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    k_weight_1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)

    if INTERLEAVED:
        row_offs = tl.arange(0, ph) * (w + 2)
        k_ptr = qkv_ptr + DD * w
        row_mask = row_offs[:, None] < (h * (w + 2))
    else:
        row_offs = tl.arange(0, ph)
        k_ptr = qkv_ptr + DD * H
        row_mask = tl.arange(0, ph)[:, None] < h

    k0 = tl.load(k_ptr
                 + pid * stride
                 + DD * row_offs[:, None]
                 + tl.arange(0, D)[None, :],
                 mask=row_mask).to(tl.float32)
    k1 = tl.load(
        k_ptr
        + pid * stride
        + D + DD * row_offs[:, None]
        + tl.arange(0, D)[None, :],
        mask=row_mask).to(tl.float32)

    if SILU:
        k0 = k0 * tl.sigmoid(k0)
        k1 = k1 * tl.sigmoid(k1)
    rms = tl.rsqrt((tl.sum(k0 * k0, 1) + tl.sum(k1 * k1, 1)) / DD + eps)
    k1 *= rms[:, None]
    k1 *= k_weight_1
    k_mask = tl.arange(0, ph)[:, None] < h
    tl.store(ko_ptr
             + pid * h * DD
             + D
             + DD * tl.arange(0, ph)[:, None]
             + tl.arange(0, D)[None, :],
             k1,
             mask=k_mask)

    k0 *= rms[:, None]
    k0 *= k_weight_0
    kr = tl.reshape(tl.permute(
        tl.flip(tl.permute(tl.reshape(k0,
                                      (ph, 2, d)),
                           (0, 2, 1)),
                dim=2) * signs,
        (0, 2, 1)),
        (ph, D))
    k0 = k0 * cos + kr * sin
    tl.store(
        ko_ptr
        + pid * h * DD
        + DD * tl.arange(0, ph)[:, None]
        + tl.arange(0, D)[None, :],
        k0,
        mask=k_mask)

    if INTERLEAVED:
        row_offs = tl.arange(0, ph) * (w + 2)
        row_mask = row_offs[:, None] < (h * (w + 2))
        v_ptr = qkv_ptr + DD * w + DD
    else:
        row_offs = tl.arange(0, ph)
        row_mask = tl.arange(0, ph)[:, None] < h
        v_ptr = qkv_ptr + DD * H + DD * h

    v0 = tl.load(v_ptr
                 + pid * stride
                 + DD * row_offs[:, None]
                 + tl.arange(0, D)[None, :],
                 mask=row_mask).to(tl.float32)
    v1 = tl.load(v_ptr
                 + pid * stride
                 + D
                 + DD * row_offs[:, None]
                 + tl.arange(0, D)[None, :],
                 mask=row_mask).to(tl.float32)

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
        mask=v_mask)
    tl.store(
        vo_ptr
        + pid * h * DD
        + D
        + DD * tl.arange(0, ph)[:, None]
        + tl.arange(0, D)[None, :],
        v1,
        mask=v_mask)


def triton_varlen_qk_norm_and_half_rope(qkv,
                                        q_norm_weight,
                                        k_norm_weight,
                                        freqs,
                                        position_ids,
                                        H=32,
                                        h=4,
                                        eps=1e-6,
                                        scaling=1.0,
                                        interleaved=False,
                                        silu=False,
                                        linear_scale=False,
                                        output_dtype=None):
    """
    split qkv to q/k/v, apply qk norm and half rope to q/k
    TOOD: support arbitrary rotary percent rather than 0.5
    Args:
        qkv: QKV tensor with size of [S, dim]
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
        output_dtype: dtype of output tensors
    Returns:
        - qo: shape [S, H, head_dim]
        - ko: shape [S, h, head_dim]
        - vo: shape [S, h, head_dim]
    """
    assert qkv.is_contiguous() and freqs.is_contiguous()
    assert q_norm_weight.is_contiguous() and k_norm_weight.is_contiguous()
    T, Dim = qkv.shape
    stride = qkv.stride(0)  # qkv may be a slice of a tensor
    D = Dim // (H + 2 * h)
    dtype = qkv.dtype if output_dtype is None else output_dtype
    device = qkv.device
    qo = torch.empty((T, H, D), dtype=dtype, device=device)
    ko = torch.empty((T, h, D), dtype=dtype, device=device)
    vo = torch.empty((T, h, D), dtype=dtype, device=device)

    num_stages = 5
    num_warps = 2
    grid = (T,)

    PH = triton.next_power_of_2(H)
    ph = triton.next_power_of_2(h)

    varlen_qk_norm_and_half_rope_kernel[grid](
        qkv,
        q_norm_weight,
        k_norm_weight,
        freqs,
        position_ids,
        qo,
        ko,
        vo,
        stride,
        eps,
        scaling,
        H,
        h,
        PH,
        ph,
        D // 2,
        D // 4,
        interleaved,
        silu,
        linear_scale,
        num_stages=num_stages,
        num_warps=num_warps)
    return qo, ko, vo