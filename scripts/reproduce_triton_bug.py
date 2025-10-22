# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""


import triton 
import triton.language as tl
import torch




@triton.jit
def rms_norm_and_smooth_quant_forward_kernel(x_ptr, weight_ptr, smooth_scale_ptr,
                                      out_ptr, scale_ptr, max_ptr, rms_ptr, 
                                      eps,
                                      M, 
                                      T, 
                                      N: tl.constexpr, 
                                      W: tl.constexpr,
                                      CALIBRATE: tl.constexpr,
                                      ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    weight = tl.load(weight_ptr + tl.arange(0, N)).to(tl.float32)[None, :]
    smooth_scale = tl.load(smooth_scale_ptr + tl.arange(0, N))[None, :]
    smooth_scale = 1.0 / tl.maximum(smooth_scale, 1e-30)
    # triton 3.3.1 has bug with N = 2048 and calibrate=True 
    if CALIBRATE:
        maxs = tl.zeros((N, ), dtype=tl.float32)
    offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, N)[
                                                            None, :]
    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32)
        rms = 1/tl.sqrt(tl.sum(x * x, axis=1) / N + eps)
        tl.store(rms_ptr + indices, rms, mask=indices < M)
        x = x * rms[:, None] * weight

        if CALIBRATE:
            maxs = tl.maximum(maxs, tl.max(tl.abs(x),0))

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
def triton_rms_norm_and_quant_forward(x, weight, smooth_scale, eps=1e-6,
                                      calibrate=False,
                                      round_scale=False):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192 and 8192 % N == 0
    device = x.device

    out = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)

    scale = torch.empty((M,), device=device, dtype=torch.float32)
    W = 8192 // N
    T = 8 if M // W >= 4096 else 4
    assert M % (T * W) == 0
    g = M // (T*W)
    if calibrate:
        maxs = torch.empty((g, N), dtype=torch.float32, device=device)
    else:
        maxs = None
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
        round_scale,
        num_stages=3,
        num_warps=4
    )
    if calibrate:
        maxs = maxs.amax(0)
    return out, scale, maxs, rms 



if __name__ == '__main__':
    M = 4096
    N = 2048
    dtype = torch.bfloat16
    device = 'cuda:0'
    calibrate = True 
    # bug condition: triton=3.3.1 N=2048 calibrate=True num_warps=4

    x = torch.randn(M, N, dtype=dtype, requires_grad=True, device=device)
    weight = torch.randn(N, dtype=dtype, requires_grad=True, device=device)
    smooth_scale = torch.rand(N, dtype=torch.float32, requires_grad=False,
                              device=device) + 0.1

    q, scale, maxs, rms = triton_rms_norm_and_quant_forward(x, weight,
                                                            smooth_scale=smooth_scale,
                                                            calibrate=calibrate,
                                                            round_scale=True)
    print(f'{q=}\n{scale=}')