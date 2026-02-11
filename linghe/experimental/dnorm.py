# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl

from linghe.experimental.symm_mem_barrier import symm_mem_sync


@triton.jit
def sp_rms_norm_forward_kernel(
        x_ptr,
        weight_ptr,
        out_ptr,
        rms_ptr,
        eps,
        M,
        T,
        n,
        buffer_ptrs,
        signal_ptrs,
        N: tl.constexpr,
        W: tl.constexpr,
        REUSE: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        GROUP_RANK: tl.constexpr,
        ):
    pid = tl.program_id(axis=0)
    buffer_ptrs = buffer_ptrs.to(tl.pointer_type(tl.uint64))

    weight = tl.load(weight_ptr + tl.arange(0, N), mask=tl.arange(0, N) < n).to(
        tl.float32
        )[None, :]

    offs = pid * W * T * n + tl.arange(0, W)[:, None] * n + tl.arange(0, N)[
                                                            None, :]
    for i in range(T):
        buffer_ptr = tl.load(buffer_ptrs + GROUP_RANK).to(
            tl.pointer_type(tl.bfloat16))
        buffer_ptr = tl.multiple_of(buffer_ptr, 16)

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

        offs_buffer = GROUP_RANK * M * n + offs
        tl.store(buffer_ptr + offs_buffer, x)
        symm_mem_sync(
            signal_ptrs,
            None,
            GROUP_RANK,
            GROUP_SIZE,
            hasPreviousMemAccess=True,
            hasSubsequentMemAccess=True,
            )
        # tl.store(out_ptr + offs, x, mask=mask)
        offs += n * W

    for i in tl.static_range(GROUP_SIZE):
        for j in range(T):
            offs = i * M * n + pid * W * T * n + j * W * n + tl.arange(0, W)[:,
                                                             None] * n + tl.arange(
                0, N)[None, :]
            buffer_ptr = tl.load(buffer_ptrs + i).to(
                tl.pointer_type(tl.bfloat16))
            buffer_ptr = tl.multiple_of(buffer_ptr, 16)
            tmp = tl.load(buffer_ptr + offs)
            tl.store(out_ptr + offs, tmp)


def triton_sp_rms_norm_forward(x, weight, hdl, eps=1e-6, rms=None):
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
    T = 1
    assert N <= 8192
    device = x.device
    out = torch.empty((hdl.world_size * M, n), dtype=torch.bfloat16,
                      device=device)
    REUSE = rms is not None
    if not REUSE:
        rms = torch.empty((M,), device=device, dtype=torch.float32)

    grid = (triton.cdiv(M, T * W),)
    sp_rms_norm_forward_kernel[grid](
        x,
        weight,
        out,
        rms,
        eps,
        M,
        T,
        n,
        hdl.buffer_ptrs_dev,
        hdl.signal_pad_ptrs_dev,
        N,
        W,
        REUSE,
        hdl.world_size,
        hdl.rank,
        num_stages=3,
        num_warps=4
        )
    # print(out)
    return out, rms
