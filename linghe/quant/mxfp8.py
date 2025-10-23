# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch 
import triton 
import triton.language as tl



@triton.jit
def mxfp8_quant_kernel(x_ptr,
                                        out_ptr, scale_ptr,
                                        transpose_output_ptr,
                                        transpose_scale_ptr,
                                        M,
                                        m,
                                        N: tl.constexpr,
                                        OUTPUT_MODE: tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)

    offs = rid * 32 * N + cid * 32 + tl.arange(0, 32)[:,
                                           None] * N + tl.arange(0, 32)[
                                                           None, :]
    indices = rid * 32 + tl.arange(0, 32)
    mask = indices[:, None] < m
    b = N // 32

    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    
    if OUTPUT_MODE % 2 == 0:
        scale = tl.maximum(tl.max(x.abs(), 1) / 448, 1e-30)
        log_scale = tl.ceil(tl.log2(scale))
        scale = tl.exp2(log_scale)
        tl.store(scale_ptr + rid * 32 * b + cid + tl.arange(0, 32) * b, log_scale+127)
        xq = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + rid * 32 * N + cid * 32 + \
             tl.arange(0, 32)[:,None] * N + tl.arange(0,32)[None, :], xq,
             mask=mask)

    if OUTPUT_MODE > 0:
        scale = tl.maximum(tl.max(x.abs(), 0) / 448, 1e-30)
        log_scale = tl.ceil(tl.log2(scale))
        scale = tl.exp2(log_scale)
        tl.store(transpose_scale_ptr + rid * N + cid * 32 + tl.arange(0, 32),
                 log_scale + 127)
        xq = (x / scale).to(out_ptr.dtype.element_ty)
        tl.store(transpose_output_ptr + rid * 32 * N + \
             cid * 32 + tl.arange(0, 32)[:, None] * N + \
                 tl.arange(0, 32)[None, :],
                 xq, mask=mask)


def triton_mxfp8_quant(x,
                                        out=None,
                                        scale=None,
                                        output_mode=2):
    """
    fused silu and mxfp8 quantization, used in shared expert
    Args:
        x: input tensor
        output_mode: one of {0, 1, 2}
            0: only output non-transposed quantized tensor
            1: only output transposed quantized tensor
            2: output both

    Returns:
        - out: quantized tensor
        - scale: quantization scale
        - transpose_output: quantized tensor of transposed output
        - transpose_scale: quantization scale of transposed output
    """
    m, N = x.shape
    M = (m + 127) // 128 * 128
    assert N % 128 == 0  # transposed scaled should be multiplier of 128
    device = x.device
    if out is None:
        out = torch.empty((m, N), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.empty((M, N // 32), device=device,
                            dtype=torch.uint8)

    transpose_output = torch.empty((m, N), device=device,
                                   dtype=torch.float8_e4m3fn)
    transpose_scale = torch.empty((M // 32, N), device=device,
                                  dtype=torch.uint8)

    grid = (M // 32, N // 32)
    mxfp8_quant_kernel[grid](
        x,
        out,
        scale,
        transpose_output,
        transpose_scale,
        M,
        m,
        N,
        output_mode,
        num_stages=2,
        num_warps=2
    )

    return out, scale, transpose_output, transpose_scale
