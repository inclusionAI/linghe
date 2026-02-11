# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def hadamard_quant_row_kernel(
    x_ptr,
    hm_ptr,
    x_q_ptr,
    x_scale_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    R: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * R * BLOCK_SIZE
    rows = row_start + tl.arange(0, R * BLOCK_SIZE)
    mask_rows = rows < M

    hm = tl.load(
        hm_ptr
        + tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)[None, :]
    )

    max_val = tl.zeros((R * BLOCK_SIZE,), dtype=tl.float32) + 1.17e-38

    num_col_blocks = tl.cdiv(N, BLOCK_SIZE)
    for col_block in range(num_col_blocks):
        col_start = col_block * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask_cols = cols < N

        offs = rows[:, None] * N + cols[None, :]
        x = tl.load(
            x_ptr + offs, mask=mask_rows[:, None] & mask_cols[None, :], other=0.0
        )
        x_transformed = tl.dot(x, hm)
        current_max = tl.max(tl.abs(x_transformed), axis=1)
        max_val = tl.maximum(max_val, current_max)

    scale = max_val / 448.0
    tl.store(x_scale_ptr + rows, scale, mask=mask_rows)
    s = 448.0 / tl.where(max_val > 0, max_val, 1.0)

    for col_block in range(num_col_blocks):
        col_start = col_block * BLOCK_SIZE
        cols = col_start + tl.arange(0, BLOCK_SIZE)
        mask_cols = cols < N

        offs = rows[:, None] * N + cols[None, :]
        x = tl.load(
            x_ptr + offs, mask=mask_rows[:, None] & mask_cols[None, :], other=0.0
        )
        x_transformed = tl.dot(x, hm)
        quantized = (x_transformed * s[:, None]).to(x_q_ptr.dtype.element_ty)
        tl.store(
            x_q_ptr + offs, quantized, mask=mask_rows[:, None] & mask_cols[None, :]
        )


@triton.jit
def hadamard_quant_col_kernel(
    x_ptr,
    hm_ptr,
    xt_q_ptr,
    xt_scale_ptr,
    M,
    N,
    BLOCK_SIZE: tl.constexpr,
    R: tl.constexpr,
):
    pid = tl.program_id(0)
    col_start = pid * R * BLOCK_SIZE
    cols = col_start + tl.arange(0, R * BLOCK_SIZE)
    mask_cols = cols < N

    hm = tl.load(
        hm_ptr
        + tl.arange(0, BLOCK_SIZE)[:, None] * BLOCK_SIZE
        + tl.arange(0, BLOCK_SIZE)[None, :]
    )

    max_val = tl.zeros((R * BLOCK_SIZE,), dtype=tl.float32) + 1.17e-38

    num_row_blocks = tl.cdiv(M, BLOCK_SIZE)
    for row_block in range(num_row_blocks):
        row_start = row_block * BLOCK_SIZE
        rows = row_start + tl.arange(0, BLOCK_SIZE)
        mask_rows = rows < M

        offs = rows[:, None] * N + cols[None, :]
        x = tl.load(
            x_ptr + offs, mask=mask_rows[:, None] & mask_cols[None, :], other=0.0
        )
        x_transformed = tl.dot(hm, x)
        current_max = tl.max(tl.abs(x_transformed), axis=0)
        max_val = tl.maximum(max_val, current_max)

    scale = max_val / 448.0
    tl.store(xt_scale_ptr + cols, scale, mask=mask_cols)
    s = 448.0 / tl.where(max_val > 0, max_val, 1.0)

    for row_block in range(num_row_blocks):
        row_start = row_block * BLOCK_SIZE
        rows = row_start + tl.arange(0, BLOCK_SIZE)
        mask_rows = rows < M

        offs = rows[:, None] * N + cols[None, :]
        x = tl.load(
            x_ptr + offs, mask=mask_rows[:, None] & mask_cols[None, :], other=0.0
        )
        x_transformed = tl.dot(hm, x)
        quantized = (x_transformed * s[None, :]).to(xt_q_ptr.dtype.element_ty)
        quantized_t = tl.trans(quantized)
        store_offs = cols[:, None] * M + rows[None, :]
        tl.store(
            xt_q_ptr + store_offs,
            quantized_t,
            mask=mask_cols[:, None] & mask_rows[None, :],
        )


def triton_hadamard_quant(x, hm):
    """
    apply hadamard transformation and then quantize transformed tensor
    Args:
        x: input tensor
        hm: hamadard matrix
    Returns:
        - x_q: rowwise quantized tensor of non-transposed x
        - x_scale: rowwise quantization scale of non-transposed x
        - xt_q: columnwise quantized tensor of transposed x
        - xt_scale: columnwise quantization scale of transposed x
    """
    M, N = x.shape
    device = x.device
    BLOCK_SIZE = hm.size(0)
    R = 1
    x_q = torch.empty((M, N), dtype=torch.float8_e4m3fn, device=device)
    xt_q = torch.empty((N, M), dtype=torch.float8_e4m3fn, device=device)
    x_scale = torch.empty((M,), dtype=torch.float32, device=device)
    xt_scale = torch.empty((N,), dtype=torch.float32, device=device)

    grid_row = (triton.cdiv(M, R * BLOCK_SIZE),)
    hadamard_quant_row_kernel[grid_row](
        x, hm, x_q, x_scale, M, N, BLOCK_SIZE, R, num_stages=6, num_warps=4
    )

    grid_col = (triton.cdiv(N, R * BLOCK_SIZE),)
    hadamard_quant_col_kernel[grid_col](
        x, hm, xt_q, xt_scale, M, N, BLOCK_SIZE, R, num_stages=6, num_warps=4
    )

    return x_q, x_scale, xt_q, xt_scale
