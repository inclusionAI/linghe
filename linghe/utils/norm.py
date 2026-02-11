# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def rms_norm_forward_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    rms_ptr,
    eps,
    M,
    T,
    n,
    N: tl.constexpr,
    W: tl.constexpr,
    REUSE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    weight = tl.load(weight_ptr + tl.arange(0, N), mask=tl.arange(0, N) < n).to(
        tl.float32
    )[None, :]

    offs = pid * W * T * n + tl.arange(0, W)[:, None] * n + tl.arange(0, N)[None, :]
    for i in range(T):
        mask = (pid * W * T + i * W + tl.arange(0, W)[:, None] < M) & (
            tl.arange(0, N) < n
        )
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        if REUSE:
            rms = tl.load(
                rms_ptr + pid * W * T + i * W + tl.arange(0, W),
                mask=pid * W * T + i * W + tl.arange(0, W) < M,
                other=1.0,
            )
        else:
            rms = tl.rsqrt(tl.sum(x * x, axis=1) / n + eps)
            tl.store(
                rms_ptr + pid * W * T + i * W + tl.arange(0, W),
                rms,
                mask=pid * W * T + i * W + tl.arange(0, W) < M,
            )

        x = (x * rms[:, None]) * weight

        tl.store(out_ptr + offs, x, mask=mask)
        offs += n * W


def triton_rms_norm_forward(x, weight, eps=1e-6, out=None, rms=None):
    """
    rms norm
    Args:
        x: input tensor
        weight: weight of rms norm
        eps: epsilon of rms norm
        rms: use x*rms to calculate output if rms is not None,
            it will accelerate recompute of rms norm
    Returns:
        out: output tensor
        rms: 1/rms of input tensor
    """
    assert x.is_contiguous() and weight.is_contiguous()
    shape = x.shape
    assert len(shape) in (2, 3)
    if len(shape) == 3:
        M, n = shape[0] * shape[1], shape[2]
    else:
        M, n = x.shape
    N = triton.next_power_of_2(n)
    W = 8192 // N
    T = 4
    assert N <= 8192
    device = x.device
    if out is None:
        out = torch.empty_like(x)
    REUSE = rms is not None
    if not REUSE:
        rms = torch.empty((M,), device=device, dtype=torch.float32)

    grid = (triton.cdiv(M, T * W),)
    rms_norm_forward_kernel[grid](
        x, weight, out, rms, eps, M, T, n, N, W, REUSE, num_stages=3, num_warps=4
    )
    return out, rms


@triton.jit
def rms_norm_backward_kernel(
    grad_output_ptr,
    x_ptr,
    w_ptr,
    rms_ptr,
    dx_ptr,
    dw_ptr,
    eps,
    M,
    T,
    n,
    N: tl.constexpr,
    W: tl.constexpr,
    REUSE: tl.constexpr,
):
    pid = tl.program_id(0)

    w = tl.load(w_ptr + tl.arange(0, N), mask=tl.arange(0, N) < n).to(tl.float32)

    offs = pid * W * T * n + tl.arange(0, W)[:, None] * n + tl.arange(0, N)[None, :]
    w_grads = tl.zeros((N,), dtype=tl.float32)
    for i in range(T):
        mask = (pid * W * T + i * W + tl.arange(0, W)[:, None] < M) & (
            tl.arange(0, N) < n
        )

        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        g = tl.load(grad_output_ptr + offs, mask=mask).to(tl.float32)
        if REUSE:
            r = tl.load(
                rms_ptr + pid * W * T + i * W + tl.arange(0, W),
                mask=pid * W * T + i * W + tl.arange(0, W) < M,
            )[:, None]
        else:
            r = tl.rsqrt(tl.sum(x * x, 1) / n + eps)[:, None]
        w_grad = x * g * r
        w_grads += tl.sum(w_grad, 0)

        dx = r * g * w - r * r * r * x * tl.sum(x * g * w, 1, keep_dims=True) / n

        tl.store(dx_ptr + offs, dx, mask=mask)

        offs += n * W

    tl.store(dw_ptr + pid * n + tl.arange(0, N), w_grads, mask=tl.arange(0, N) < n)


def triton_rms_norm_backward(grad_output, x, w, eps=1e-6, rms=None):
    assert grad_output.is_contiguous()
    shape = x.shape
    if len(shape) == 3:
        M, n = shape[0] * shape[1], shape[2]
    else:
        M, n = x.shape
    N = triton.next_power_of_2(n)
    assert N <= 8192

    dx = torch.empty_like(x)
    REUSE = rms is not None

    W = 8192 // N
    T = 16
    g = triton.cdiv(M, T * W)
    tmp_dw = torch.empty(g, n, dtype=torch.float32, device=w.device)
    grid = (g,)
    rms_norm_backward_kernel[grid](
        grad_output,
        x,
        w,
        rms,
        dx,
        tmp_dw,
        eps,
        M,
        T,
        n,
        N,
        W,
        REUSE,
        num_stages=3,
        num_warps=4,
    )
    return dx, tmp_dw.sum(dim=0)


# output non-transposed and transposed together
# performance is bad with batchsize < 16384
@triton.jit
def rms_norm_and_block_quant_forward_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    scale_ptr,
    transpose_output_ptr,
    transpose_scale_ptr,
    rms_ptr,
    eps,
    M,
    n,
    N: tl.constexpr,
    T: tl.constexpr,
    W: tl.constexpr,
    H: tl.constexpr,
    ROUND: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    NB: tl.constexpr = N // 128
    nb = n // 128

    mask = tl.arange(0, N) < n
    weight = tl.load(weight_ptr + tl.arange(0, N), mask=mask).to(tl.float32)[None, :]
    offs = pid * W * T * n + tl.arange(0, W)[:, None] * n + tl.arange(0, N)[None, :]
    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        masks = (indices[:, None] < M) & (tl.arange(0, N) < n)
        x = tl.load(x_ptr + offs, mask=masks).to(tl.float32)
        rms = tl.rsqrt(tl.sum(x * x, axis=1) / n + eps)
        tl.store(rms_ptr + indices, rms, mask=indices < M)
        x = (x * rms[:, None]) * weight
        x = tl.reshape(x, [W, NB, 128])
        scale = tl.maximum(tl.max(tl.abs(x), 2) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        x = (x / scale[:, :, None]).to(out_ptr.dtype.element_ty)
        x = tl.reshape(x, [W, N])
        tl.store(
            scale_ptr + tl.arange(0, NB)[:, None] * M + indices[None, :],
            tl.trans(scale),
            mask=(indices[None, :] < M) & (tl.arange(0, NB)[:, None] < nb),
        )
        tl.store(out_ptr + offs, x, mask=masks)
        offs += n * W

    offs = pid * W * T * n + tl.arange(0, 128)[:, None] * n + tl.arange(0, H)[None, :]
    toffs = pid * 128 + tl.arange(0, H)[:, None] * M + tl.arange(0, 128)[None, :]
    indices = pid * W * T + tl.arange(0, 128)
    tl.debug_barrier()
    rms = tl.load(rms_ptr + indices, mask=indices < M)[:, None]
    for i in range(n // H):
        x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32)
        wgt = tl.load(weight_ptr + i * H + tl.arange(0, H)).to(tl.float32)
        x = x * rms * wgt
        scale = tl.maximum(tl.max(x.abs(), 0) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(transpose_scale_ptr + pid * n + i * H + tl.arange(0, H), scale)

        x = (x / scale).to(transpose_output_ptr.dtype.element_ty)
        tl.store(transpose_output_ptr + toffs, tl.trans(x), mask=indices[None, :] < M)
        offs += H
        toffs += M * H


# output non-transposed tensor only
@triton.jit
def rms_norm_and_block_quant_forward_n_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    scale_ptr,
    rms_ptr,
    eps,
    M,
    n,
    N: tl.constexpr,
    T: tl.constexpr,
    W: tl.constexpr,
    ROUND: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    NB: tl.constexpr = N // 128

    mask = tl.arange(0, N) < n
    weight = tl.load(weight_ptr + tl.arange(0, N), mask=mask).to(tl.float32)[None, :]
    offs = pid * W * T * n + tl.arange(0, W)[:, None] * n + tl.arange(0, N)[None, :]
    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        masks = (indices[:, None] < M) & (tl.arange(0, N) < n)
        x = tl.load(x_ptr + offs, mask=masks).to(tl.float32)
        rms = tl.rsqrt(tl.sum(x * x, axis=1) / n + eps)
        tl.store(rms_ptr + indices, rms, mask=indices < M)

        x = x * rms[:, None] * weight
        x = tl.reshape(x, [W, NB, 128])
        scale = tl.maximum(tl.max(tl.abs(x), 2) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        # x = (x / scale[:,:, None]).to(out_ptr.dtype.element_ty)
        # x = tl.reshape(x, [W, N])
        x = x / scale[:, :, None]
        x = tl.reshape(x, [W, N])

        tl.store(
            scale_ptr + tl.arange(0, NB)[:, None] * M + indices[None, :],
            tl.trans(scale),
            mask=(indices[None, :] < M) & (tl.arange(0, NB)[:, None] < n // 128),
        )
        tl.store(out_ptr + offs, x, mask=masks)
        offs += n * W


# output transposed tensor only
@triton.jit
def rms_norm_and_block_quant_forward_t_kernel(
    x_ptr,
    weight_ptr,
    transpose_output_ptr,
    transpose_scale_ptr,
    rms_ptr,
    M,
    N,
    W: tl.constexpr,
    ROUND: tl.constexpr,
):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)

    offs = (
        rid * 128 * N
        + cid * W
        + tl.arange(0, 128)[:, None] * N
        + tl.arange(0, W)[None, :]
    )
    toffs = (
        rid * 128
        + cid * M * W
        + tl.arange(0, W)[:, None] * M
        + tl.arange(0, 128)[None, :]
    )

    weight = tl.load(weight_ptr + cid * W + tl.arange(0, W)).to(tl.float32)
    indices = rid * 128 + tl.arange(0, 128)
    rms = tl.load(rms_ptr + indices, mask=indices < M)[:, None]
    x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32)
    x = x * rms * weight
    scale = tl.maximum(tl.max(x.abs(), 0) / 448.0, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(transpose_scale_ptr + rid * N + cid * W + tl.arange(0, W), scale)

    x = (tl.trans(x / scale)).to(transpose_output_ptr.dtype.element_ty)
    tl.store(transpose_output_ptr + toffs, x, mask=indices[None, :] < M)


def triton_rms_norm_and_block_quant_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    rms: Optional[torch.Tensor] = None,
    round_scale: bool = False,
    output_mode: int = 2,
):
    """
    Fused RMSNorm forward and block quantization.
    Args:
        x: Input tensor, shape [M, N]
        weight: RMSNorm weight,  shape [N]
        eps: epsilon value for L2 normalization.
        out: output of quantization data
        scale: output of quantization scale.
        rms: output of rms
        round_scale: Set whether to force power of 2 scales.
        output_mode: one of {0, 1, 2}.
            0: only output non-transpose tensor
            1: only output transposed tensor
            2: return both
    Returns:
        - out: quantization data.
        - scale: quantization scale.
        - rms: Reciprocal of the root mean square of the
            input calculated over the last dimension.
        - transpose_output: quantization data of transposed gradient.
        - transpose_scale: quantization scale of transposed gradient.
    """
    # row-wise read, row-wise write
    assert x.is_contiguous() and weight.is_contiguous()
    M, n = x.shape
    N = triton.next_power_of_2(n)
    assert N <= 8192
    device = x.device

    if out is None and output_mode in (0, 2):
        out = torch.empty((M, n), device=device, dtype=torch.float8_e4m3fn)

    if scale is None and output_mode in (0, 2):
        scale = torch.empty((n // 128, M), device=device, dtype=torch.float32)

    # transpose_output should be initialized, or else can not make splitted tensors
    transpose_output = torch.empty((n, M), device=device, dtype=torch.float8_e4m3fn)
    transpose_scale = torch.empty(
        ((M + 127) // 128, n), device=device, dtype=torch.float32
    )
    if output_mode == 0:  # only output non-transpose tensor
        assert rms is None
        rms = torch.empty((M,), dtype=torch.float32, device=device)
        W = 8192 // N
        T = 16 // W
        grid = (triton.cdiv(M, 16),)
        rms_norm_and_block_quant_forward_n_kernel[grid](
            x,
            weight,
            out,
            scale,
            rms,
            eps,
            M,
            n,
            N,
            T,
            W,
            round_scale,
            num_stages=3,
            num_warps=4,
        )

    elif output_mode == 1:  # only output transposed tensor
        assert rms is not None
        W = 32
        assert n % W == 0
        grid = (triton.cdiv(M, 128), n // W)
        rms_norm_and_block_quant_forward_t_kernel[grid](
            x,
            weight,
            transpose_output,
            transpose_scale,
            rms,
            M,
            n,
            W,
            round_scale,
            num_stages=3,
            num_warps=4,
        )

    elif output_mode == 2:  # output non-transposed and transposed tensor together
        # we force set output_mode=2 when recompute qkv, but it has rms
        # assert rms is None
        rms = torch.empty((M,), dtype=torch.float32, device=device)
        if M >= 1048576:  # not used
            W = 4096 // N
            T = 128 // W
            H = 32
            assert n % 128 == 0
            grid = (triton.cdiv(M, 128),)
            rms_norm_and_block_quant_forward_kernel[grid](
                x,
                weight,
                out,
                scale,
                transpose_output,
                transpose_scale,
                rms,
                eps,
                M,
                n,
                N,
                T,
                W,
                H,
                round_scale,
                num_stages=2,
                num_warps=4,
            )
        else:
            W = 8192 // N
            T = 16 // W
            grid = (triton.cdiv(M, 16),)
            rms_norm_and_block_quant_forward_n_kernel[grid](
                x,
                weight,
                out,
                scale,
                rms,
                eps,
                M,
                n,
                N,
                T,
                W,
                round_scale,
                num_stages=3,
                num_warps=4,
            )

            W = 32
            assert n % W == 0, f" {n=} {W=}"
            grid = (triton.cdiv(M, 128), n // W)
            rms_norm_and_block_quant_forward_t_kernel[grid](
                x,
                weight,
                transpose_output,
                transpose_scale,
                rms,
                M,
                n,
                W,
                round_scale,
                num_stages=3,
                num_warps=4,
            )

    return out, scale, rms, transpose_output, transpose_scale


@triton.jit
def rms_norm_and_smooth_quant_forward_kernel(
    x_ptr,
    weight_ptr,
    smooth_scale_ptr,
    out_ptr,
    scale_ptr,
    max_ptr,
    rms_ptr,
    eps,
    M,
    T,
    N: tl.constexpr,
    W: tl.constexpr,
    CALIBRATE: tl.constexpr,
    OUTPUT: tl.constexpr,
    ROUND: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    weight = tl.load(weight_ptr + tl.arange(0, N)).to(tl.float32)[None, :]
    smooth_scale = tl.load(smooth_scale_ptr + tl.arange(0, N))[None, :]
    smooth_scale = 1.0 / tl.maximum(smooth_scale, 1e-30)
    if CALIBRATE:
        # triton 3.3.1 has bug with N = 2048 and calibrate=True
        maxs = tl.zeros((N,), dtype=tl.float32)
    offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, N)[None, :]
    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32)
        rms = tl.rsqrt(tl.sum(x * x, axis=1) / N + eps)
        if OUTPUT:
            tl.store(rms_ptr + indices, rms, mask=indices < M)
        x = x * rms[:, None] * weight

        if CALIBRATE:
            maxs = tl.maximum(maxs, tl.max(tl.abs(x), 0))

        x = x * smooth_scale
        scale = tl.maximum(tl.max(tl.abs(x), 1) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        q = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(scale_ptr + indices, scale, mask=indices < M)
        tl.store(out_ptr + offs, q, mask=indices[:, None] < M)
        offs += N * W

    if CALIBRATE:
        tl.store(max_ptr + pid * N + tl.arange(0, N), maxs)


# rms is used for moe routing, it is stored as 1/rms
def triton_rms_norm_and_smooth_quant_forward(
    x,
    weight,
    smooth_scale=None,
    eps=1e-6,
    out=None,
    scale=None,
    rms=None,
    calibrate=False,
    output_rms=False,
    round_scale=False,
):
    """"""
    assert x.is_contiguous() and weight.is_contiguous()
    M, N = x.shape
    assert N <= 8192 and 8192 % N == 0
    device = x.device

    if out is None:
        out = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)

    if scale is None:
        scale = torch.empty((M,), device=device, dtype=torch.float32)
    W = 8192 // N
    T = 8 if M // W >= 4096 else 4
    assert M % (T * W) == 0
    g = M // (T * W)
    if calibrate:
        maxs = torch.empty((g, N), dtype=torch.float32, device=device)
    else:
        maxs = None
    if output_rms and rms is None:
        rms = torch.empty((M,), dtype=torch.float32, device=device)
    grid = (g,)
    rms_norm_and_smooth_quant_forward_kernel[grid](
        x,
        weight,
        smooth_scale,
        out,
        scale,
        maxs,
        rms,
        eps,
        M,
        T,
        N,
        W,
        calibrate,
        output_rms,
        round_scale,
        num_stages=3,
        num_warps=2 if N == 2048 else 4,
    )
    if calibrate:
        maxs = maxs.amax(0)

    return out, scale, maxs, rms


@triton.jit
def rms_norm_fp32_gemm_block_quant_forward_n_kernel(
    x_ptr,
    norm_weight_ptr,
    route_weight_ptr,
    y_ptr,
    rms_ptr,
    logit_ptr,
    xq_ptr,
    xs_ptr,
    eps,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    ROUND: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_offs = offs_m[:, None] * K + offs_k[None, :]

    rms = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for i in range(k):
        x = tl.load(x_ptr + x_offs).to(tl.float32)
        rms += tl.sum(x * x, axis=1)
        x_offs += BLOCK_SIZE_K

    rms = tl.rsqrt(rms / K + eps)

    tl.store(rms_ptr + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M), rms)

    x_offs = offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = route_weight_ptr + offs_n[None, :] * K + offs_k[:, None]
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        norm_weight = tl.load(norm_weight_ptr + i * BLOCK_SIZE_K + offs_k).to(
            tl.float32
        )
        x = tl.load(x_ptr + x_offs).to(tl.float32)
        w = tl.load(w_ptrs).to(tl.float32)

        x = x * rms[:, None] * norm_weight
        tl.store(y_ptr + x_offs, x)

        c = tl.dot(x, w, c)

        scale = tl.maximum(tl.max(tl.abs(x), 1) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))

        x = x / scale[:, None]

        tl.store(
            xs_ptr + M * i + pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M), scale
        )
        tl.store(xq_ptr + x_offs, x)

        x_offs += BLOCK_SIZE_K
        w_ptrs += BLOCK_SIZE_K

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = logit_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, c)


def triton_rms_norm_fp32_gemm_block_quant_forward(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    route_weight: torch.Tensor,
    rms: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    output_mode: int = 0,
    round_scale=False,
):
    """
    y = rms_norm(x)
    logits = y@w_route
    x_q, x_s, xt_q, xt_s = quantization(y)
    Args:
        x: input tensor
        norm weight: weight tensor of rms norm
        route_weight: moe router weight
        eps: epsilon of rms norm
        output_mode: 0 or 1
            0: only output non-transpose quantizatino tensor
            1: only output transposed quantizatino tensor

    Returns:
        - y: rms normed tensor
        - rms: 1/rms
        - logits: router logit
        - x_q:
        - x_s:
        - xt_q:
        - xt_s:
    """
    assert (
        x.is_contiguous()
        and norm_weight.is_contiguous()
        and route_weight.is_contiguous()
    )
    assert output_mode in (0, 1)
    M, K = x.size()
    N, K = route_weight.size()
    assert M % 128 == 0 and K % 128 == 0 and N % 128 == 0
    device = x.device
    y = torch.empty(M, K, dtype=x.dtype, device=device)
    if rms is None:
        assert output_mode == 0
        rms = torch.empty(M, dtype=torch.float32, device=device)
    logits = torch.empty(M, N, dtype=torch.float32, device=device)

    x_q = torch.empty((M, K), device=device, dtype=torch.float8_e4m3fn)
    x_s = torch.empty((K // 128, M), device=device, dtype=torch.float32)
    xt_q = torch.empty((K, M), device=device, dtype=torch.float8_e4m3fn)
    xt_s = torch.empty((M // 128, K), device=device, dtype=torch.float32)

    if output_mode == 0:
        BLOCK_SIZE_K = 128  # MUST BE 128 (quantization block size)
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = N
        num_warps = 4
        num_stages = 2
        grid = (M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)
        rms_norm_fp32_gemm_block_quant_forward_n_kernel[grid](
            x,
            norm_weight,
            route_weight,
            y,
            rms,
            logits,
            x_q,
            x_s,
            eps,
            M,
            N,
            K,
            BLOCK_SIZE_K,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            round_scale,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        W = 32
        grid = (triton.cdiv(M, 128), K // W)
        rms_norm_and_block_quant_forward_t_kernel[grid](
            x,
            norm_weight,
            xt_q,
            xt_s,
            rms,
            M,
            K,
            W,
            round_scale,
            num_stages=3,
            num_warps=4,
        )

    return y, rms, logits, x_q, x_s, xt_q, xt_s
