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
                                        B: tl.constexpr,
                                        D: tl.constexpr,
                                        d: tl.constexpr,
                                        INTERLEAVED: tl.constexpr,
                                        SILU: tl.constexpr,
                                        SCALE: tl.constexpr):
    pid = tl.program_id(0)
    hid = tl.program_id(1)

    pos = tl.load(position_ids + pid)

    DD = D * 2

    cos = tl.load(freqs_ptr + pos * D + tl.arange(0, D) % d).to(tl.float32)

    sin = tl.load(freqs_ptr + pos * D + d + tl.arange(0, D) % d).to(tl.float32)

    signs = tl.arange(0, 2).to(tl.float32) * 2 - 1

    q_weight_0 = tl.load(q_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    q_weight_1 = tl.load(q_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)
    q_ptr = qkv_ptr
    G: tl.constexpr = H // h

    if INTERLEAVED:
        if B == 1:
            # query per token
            row_offs = tl.arange(0, PH) + tl.arange(0, PH) // G * 2
            row_mask = tl.arange(0, PH)[:, None] < H
        else:
            # query per kv head
            row_offs = hid * (G + 2) + tl.arange(0, G)
            row_mask = None
    else:
        if B == 1:
            # query per token
            row_offs = tl.arange(0, PH)
            row_mask = row_offs[:, None] < H
        else:
            # query per kv head
            row_offs = hid * G + tl.arange(0, G)
            row_mask = None


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

    if B == 1:
        q_mask = tl.arange(0, PH)[:, None] < H
        tl.store(
            qo_ptr
            + pid * H * DD
            + D
            + DD * tl.arange(0, PH)[:, None]
            + tl.arange(0, D)[None, :],
            q1,
            mask=q_mask)
    else:
        tl.store(
            qo_ptr
            + pid * H * DD
            + D
            + hid * G * DD 
            + DD * tl.arange(0, G)[:, None]
            + tl.arange(0, D)[None, :],
            q1,
            mask=None)

    q0 *= rms[:, None]
    q0 *= q_weight_0

    if SCALE:
        q0 *= linear_scale_value


    if B == 1:
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
    else:
        qr = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(q0,
                                        (G, 2, d)),
                            (0, 2, 1)),
                    dim=2) * signs,
            (0, 2, 1)),
            (G, D))
        q0 = q0 * cos + qr * sin
        tl.store(
            qo_ptr
            + pid * H * DD
            + hid * G * DD
            + DD * tl.arange(0, G)[:, None]
            + tl.arange(0, D)[None, :],
            q0,
            mask=None)

    k_weight_0 = tl.load(k_norm_weight_ptr + tl.arange(0, D)).to(tl.float32)
    k_weight_1 = tl.load(k_norm_weight_ptr + D + tl.arange(0, D)).to(tl.float32)

    if INTERLEAVED:
        if B == 1:
            k_ptr = qkv_ptr + DD * G
            row_offs = tl.arange(0, ph) * (G + 2)
            row_mask = row_offs[:, None] < (h * (G + 2))
        else:
            k_ptr = qkv_ptr + DD * G
            row_offs = hid * (G + 2) + tl.arange(0, 1) * (G + 2)
            row_mask = None
    else:
        if B == 1:
            row_offs = tl.arange(0, ph)
            k_ptr = qkv_ptr + DD * H
            row_mask = tl.arange(0, ph)[:, None] < h
        else:
            row_offs = hid + tl.arange(0, 1)
            k_ptr = qkv_ptr + DD * H
            row_mask = None

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
    
    if B == 1:
        k_mask = tl.arange(0, ph)[:, None] < h
        tl.store(ko_ptr
                + pid * h * DD
                + D
                + DD * tl.arange(0, ph)[:, None]
                + tl.arange(0, D)[None, :],
                k1,
                mask=k_mask)
    else:
        tl.store(ko_ptr
                + pid * h * DD
                + D
                + hid * DD
                + DD * tl.arange(0, 1)[:, None]
                + tl.arange(0, D)[None, :],
                k1,
                mask=None)

    k0 *= rms[:, None]
    k0 *= k_weight_0
    if B == 1:
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
    else:
        kr = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(k0,
                                        (1, 2, d)),
                            (0, 2, 1)),
                    dim=2) * signs,
            (0, 2, 1)),
            (1, D))
        k0 = k0 * cos + kr * sin
        tl.store(
            ko_ptr
            + pid * h * DD
            + hid * DD
            + DD * tl.arange(0, 1)[:, None]
            + tl.arange(0, D)[None, :],
            k0,
            mask=None)

    if INTERLEAVED:
        if B == 1:
            v_ptr = qkv_ptr + DD * G + DD
            row_offs = hid * h * (G + 2) + tl.arange(0, ph) * (G + 2)
            row_mask = row_offs[:, None] < (h * (G + 2))
        else:
            v_ptr = qkv_ptr + DD * G + DD
            row_offs = hid * (G + 2) + tl.arange(0, 1) * (G + 2)
            row_mask = None
    else:
        if B == 1:
            v_ptr = qkv_ptr + DD * H + DD * h
            row_offs = hid * h + tl.arange(0, ph)
            row_mask = tl.arange(0, ph)[:, None] < h
        else:
            v_ptr = qkv_ptr + DD * H + DD * h
            row_offs = hid + tl.arange(0, 1)
            row_mask = None

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
    if B == 1:
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
    else:
        tl.store(
            vo_ptr
            + pid * h * DD
            + hid * DD
            + DD * tl.arange(0, 1)[:, None]
            + tl.arange(0, D)[None, :],
            v0,
            mask=None)
        tl.store(
            vo_ptr
            + pid * h * DD
            + hid * DD
            + D
            + DD * tl.arange(0, 1)[:, None]
            + tl.arange(0, D)[None, :],
            v1,
            mask=None)


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

    num_stages = 3
    num_warps = 2

    PH = triton.next_power_of_2(H)
    ph = triton.next_power_of_2(h)

    if h >= 2 and T < 128:
        B = h
    else:
        B = 1
    grid = (T, B)

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
        B,
        D // 2,
        D // 4,
        interleaved,
        silu,
        linear_scale,
        num_stages=num_stages,
        num_warps=num_warps)
    return qo, ko, vo


@triton.jit
def mla_rope_kernel(q_ptr,
                            k_ptr,
                            freqs_ptr,
                            position_ids_ptr,
                            q_stride_0,
                            q_stride_1,
                            k_stride_0,
                            k_stride_1,
                            H: tl.constexpr,
                            h: tl.constexpr,
                            SINGLE: tl.constexpr,
                            D: tl.constexpr,
                            d: tl.constexpr,
                            INTERLEAVE: tl.constexpr,
                            ):
    pid = tl.program_id(0)
    hid = tl.program_id(1)

    pos = tl.load(position_ids_ptr + pid)

    cos = tl.load(freqs_ptr + pos * D + tl.arange(0, D) % d).to(tl.float32)
    sin = tl.load(freqs_ptr + pos * D + d + tl.arange(0, D) % d).to(tl.float32)
    if INTERLEAVE:
        cos = tl.reshape(tl.trans(tl.reshape(cos,
                                        (2, d))),
            (D, ))
        sin = tl.reshape(tl.trans(tl.reshape(sin,
                                        (2, d))),
            (D, ))

    signs = tl.arange(0, 2).to(tl.float32) * 2 - 1

    q = tl.load(q_ptr
                 + pid * q_stride_0
                 + hid * h * q_stride_1 
                 + tl.arange(0, h)[:, None] * q_stride_1
                 + tl.arange(0, D)[None, :]).to(tl.float32)

    if INTERLEAVE:
        qr = tl.reshape(
            tl.flip(tl.reshape(q,
                                        (h, d, 2)),
                    dim=2) * signs,
            (h, D))
    else:
        qr = tl.reshape(tl.permute(
            tl.flip(tl.permute(tl.reshape(q,
                                        (h, 2, d)),
                            (0, 2, 1)),
                    dim=2) * signs,
            (0, 2, 1)),
            (h, D))
    q = q * cos + qr * sin
    tl.store(
        q_ptr
        + pid * q_stride_0
        + hid * h * q_stride_1
        + tl.arange(0, h)[:, None] * q_stride_1
        + tl.arange(0, D)[None, :],
        q)
    
    if SINGLE:
        if hid == 0:
            k = tl.load(k_ptr
                        + pid * k_stride_0
                        + tl.arange(0, D)).to(tl.float32)

            if INTERLEAVE:
                kr = tl.reshape(tl.flip(tl.reshape(k,
                                                (d, 2)),
                            dim=1) * signs,
                    (D, ))
            else:
                kr = tl.reshape(tl.permute(
                    tl.flip(tl.permute(tl.reshape(k,
                                                (2, d)),
                                    (1, 0)),
                            dim=1) * signs,
                    (1, 0)),
                    (D, ))
            k = k * cos + kr * sin
            tl.store(
                k_ptr
                + pid * k_stride_0
                + tl.arange(0, D),
                k)
    else:

        k = tl.load(k_ptr
                    + pid * k_stride_0
                    + hid * h * k_stride_1 
                    + tl.arange(0, h)[:, None] * k_stride_1
                    + tl.arange(0, D)[None, :]).to(tl.float32)

        if INTERLEAVE:
            kr = tl.reshape(
                tl.flip(tl.reshape(k,
                                            (h, d, 2)),
                        dim=2) * signs,
                (h, D))
        else:
            kr = tl.reshape(tl.permute(
                tl.flip(tl.permute(tl.reshape(k,
                                            (h, 2, d)),
                                (0, 2, 1)),
                        dim=2) * signs,
                (0, 2, 1)),
                (h, D))
        k = k * cos + kr * sin
        tl.store(
            k_ptr
            + pid * k_stride_0
            + hid * h * k_stride_1
            + tl.arange(0, h)[:, None] * k_stride_1
            + tl.arange(0, D)[None, :],
            k)


def triton_mla_rope(q,
                            k,
                            freqs,
                            position_ids,
                            interleave=True):
    """
    apply MLA-type rope
    Args:
        q: query tensor, [t, n_heads, 64]
        k: key tensor, [t, 1, 64]
        freqs: rope freqs, [len, 64]
        position_ids: position_ids for rope
        interleave: whether q/k is interleaved
    Returns:

    """

    assert freqs.is_contiguous()

    N, H, D = q.shape
    hk = k.shape[1]
    assert hk == H or hk == 1
    SINGLE = k.shape[1] == 1
    q_stride_0 = q.stride(0)
    q_stride_1 = q.stride(1)
    k_stride_0 = k.stride(0)
    k_stride_1 = k.stride(1)

    num_stages = 3
    num_warps = 4
    if N <= 64 and H % 4 == 0:
        B = 4
        h = H // B
    else:
        B = 1
        h = H

    grid = (N, B)
    mla_rope_kernel[grid](
        q,
        k,
        freqs,
        position_ids,
        q_stride_0,
        q_stride_1,
        k_stride_0,
        k_stride_1,
        H,
        h,
        SINGLE,
        D,
        D//2,
        interleave,
        num_stages=num_stages,
        num_warps=num_warps)
    return q, k
