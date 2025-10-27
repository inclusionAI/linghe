# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import torch.nn.functional as F

from linghe.utils.norm import (triton_rms_norm_and_smooth_quant_forward,
                               triton_rms_norm_and_block_quant_forward,
                               triton_rms_norm_and_mxfp8_quant_forward,
                               triton_rms_norm_backward,
                               triton_rms_norm_forward)
from linghe.tools.util import (output_check,
                               torch_smooth_quant,
                               torch_group_quant,
                               torch_mxfp8_quant)
from linghe.tools.benchmark import benchmark_func


def torch_rms_forward(x, weight):
    x = x.float()
    weight = weight.float()
    N = x.shape[-1]
    rmsnorm = torch.nn.RMSNorm(
        normalized_shape=N,
        eps=1e-6,
        dtype=torch.float32,
        device=x.device
    )
    with torch.no_grad():
        rmsnorm.weight.copy_(weight)
    return rmsnorm(x)


def torch_rms_backward(x, weight, dy):
    x = x.float()
    weight = weight.float()
    dy = dy.float()
    N = x.shape[-1]
    rmsnorm = torch.nn.RMSNorm(
        normalized_shape=N,
        eps=1e-6,
        dtype=torch.float32,
        device=x.device
    )
    with torch.no_grad():
        rmsnorm.weight.copy_(weight)
    x = x.clone().detach().requires_grad_()
    y = rmsnorm(x)
    y.backward(gradient=dy)
    return x.grad, rmsnorm.weight.grad


def torch_rms_and_smooth_quant_forward(x, weight, smooth_scale=None,
                                       round_scale=False):
    x = x.float()
    weight = weight.float()
    smooth_scale = smooth_scale.float()
    N = x.shape[-1]
    rmsnorm = torch.nn.RMSNorm(
        normalized_shape=N,
        eps=1e-6,
        dtype=torch.float32,
        device=x.device
    )
    with torch.no_grad():
        rmsnorm.weight.copy_(weight)
    y = rmsnorm(x)
    # smooth
    y_q, y_scale, y_maxs = torch_smooth_quant(y, smooth_scale, reverse=False,
                                              round_scale=round_scale)
    return y_q, y_scale, y_maxs


def torch_rms_and_block_quant_forward(x, weight, round_scale=False):
    x = x.float()
    weight = weight.float()
    N = x.shape[-1]
    rmsnorm = torch.nn.RMSNorm(
        normalized_shape=N,
        eps=1e-6,
        dtype=torch.float32,
        device=x.device
    )
    with torch.no_grad():
        rmsnorm.weight.copy_(weight)
    y = rmsnorm(x)
    # blockwise
    y_q, y_scale = torch_group_quant(y, round_scale=round_scale)
    yt_q, yt_scale = torch_group_quant(y.t(), round_scale=round_scale)
    return y_q, y_scale, yt_q, yt_scale

def torch_rms_and_mxfp8_quant_forward(x, weight):
    x = x.float()
    weight = weight.float()
    N = x.shape[-1]
    rmsnorm = torch.nn.RMSNorm(
        normalized_shape=N,
        eps=1e-6,
        dtype=torch.float32,
        device=x.device
    )
    with torch.no_grad():
        rmsnorm.weight.copy_(weight)
    y = rmsnorm(x)
    # mxfp8
    y_q, y_scale, yt_q, yt_scale = torch_mxfp8_quant(y)
    return y_q, y_scale, yt_q, yt_scale

def test_rmsnorm(M=4096, N=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.randn(M, N, dtype=dtype, requires_grad=True, device=device) ** 3
    weight = torch.randn(N, dtype=dtype, requires_grad=True, device=device)
    dy = torch.randn(M, N, dtype=dtype, device=device)

    y_ref = torch_rms_forward(x, weight)
    y = triton_rms_norm_forward(x, weight)
    output_check(y_ref.float(), y.float(), 'y')

    dx_ref, dw_ref = torch_rms_backward(x, weight, dy)
    dx, dw = triton_rms_norm_backward(dy, x, weight)
    output_check(dx_ref, dx, mode="dx")
    output_check(dw_ref, dw, mode='dw')

    if bench:
        benchmark_func(triton_rms_norm_forward, x, weight, ref_bytes=M * N * 3)
        benchmark_func(triton_rms_norm_backward, dy, x, weight,
                       ref_bytes=M * N * 3)


def test_rmsnorm_and_smooth_quant(M=4096, N=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.randn(M, N, dtype=dtype, requires_grad=True, device=device)
    weight = torch.randn(N, dtype=dtype, requires_grad=True, device=device)
    smooth_scale = torch.rand(N, dtype=torch.float32, requires_grad=False,
                              device=device) + 0.1
    calibrate = True

    # smooth
    q_ref, scale_ref, maxs_ref = torch_rms_and_smooth_quant_forward(x, weight,
                                                                    smooth_scale=smooth_scale,
                                                                    round_scale=True)

    q, scale, maxs, rms = triton_rms_norm_and_smooth_quant_forward(x, weight,
                                                                   smooth_scale=smooth_scale,
                                                                   calibrate=calibrate,
                                                                   output_rms=True,
                                                                   round_scale=True)
    output_check(q_ref, q, mode="smooth.data")
    output_check(scale_ref, scale, mode='smooth.scale')
    if calibrate:
        output_check(maxs_ref, maxs, mode="smooth.maxs")

    if bench:
        benchmark_func(triton_rms_norm_and_smooth_quant_forward, x, weight,
                       smooth_scale=smooth_scale,
                       calibrate=True,
                       round_scale=True,
                       output_rms=True,
                       ref_bytes=M * N * 3)


def test_rmsnorm_and_block_quant(M=4096, N=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.randn(M, N, dtype=dtype, requires_grad=True, device=device) ** 2
    weight = torch.randn(N, dtype=dtype, requires_grad=True, device=device)

    # blockwise
    q_ref, scale_ref, qt_ref, scale_t_ref = torch_rms_and_block_quant_forward(x,
                                                                              weight,
                                                                              round_scale=True)
    q, scale, rms, q_t, scale_t = triton_rms_norm_and_block_quant_forward(x,
                                                                          weight,
                                                                          round_scale=True,
                                                                          output_mode=2)
    output_check(q_ref, q, mode="2.block.data")
    output_check(scale_ref.t(), scale, mode='2.block.scale')
    output_check(qt_ref, q_t, mode='2.block.t_data')
    output_check(scale_t_ref.t(), scale_t, mode="2.block.t_scale")

    q, scale, _, _, _ = triton_rms_norm_and_block_quant_forward(x, weight,
                                                                round_scale=True,
                                                                output_mode=0)
    output_check(q_ref, q, mode="0.block.data")
    output_check(scale_ref.t(), scale, mode='0.block.scale')

    _, _, _, q_t, scale_t = triton_rms_norm_and_block_quant_forward(x, weight,
                                                                    round_scale=True,
                                                                    rms=rms,
                                                                    output_mode=1)
    output_check(qt_ref, q_t, mode='0.block.t_data')
    output_check(scale_t_ref.t(), scale_t, mode="0.block.t_scale")

    if bench:
        benchmark_func(triton_rms_norm_and_block_quant_forward, x, weight,
                       round_scale=True,
                       output_mode=0,
                       ref_bytes=M * N * 3)

        benchmark_func(triton_rms_norm_and_block_quant_forward, x, weight,
                       round_scale=True,
                       output_mode=1,
                       ref_bytes=M * N * 3)

        benchmark_func(triton_rms_norm_and_block_quant_forward, x, weight,
                       round_scale=True,
                       output_mode=2,
                       ref_bytes=M * N * 4)

def test_rmsnorm_and_mxfp8_quant(M=4096, N=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.randn(M, N, dtype=dtype, requires_grad=True, device=device) ** 2
    weight = torch.randn(N, dtype=dtype, requires_grad=True, device=device)

    # mxfp8
    q_ref, scale_ref, qt_ref, scale_t_ref = torch_rms_and_mxfp8_quant_forward(x,
                                                                              weight)
    q, scale, rms, q_t, scale_t = triton_rms_norm_and_mxfp8_quant_forward(x,
                                                                          weight,
                                                                    output_mode=2)
    output_check(q_ref, q, mode="2.block.data")
    output_check(scale_ref.t(), scale, mode='2.block.scale')
    output_check(qt_ref, q_t, mode='2.block.t_data')
    output_check(scale_t_ref, scale_t, mode="2.block.t_scale")

    q, scale, _, _, _ = triton_rms_norm_and_mxfp8_quant_forward(x, weight,
                                                                output_mode=0)
    output_check(q_ref, q, mode="0.block.data")
    output_check(scale_ref.t(), scale, mode='0.block.scale')

    _, _, _, q_t, scale_t = triton_rms_norm_and_mxfp8_quant_forward(x, weight,
                                                                    rms=rms,
                                                                    output_mode=1)
    output_check(qt_ref, q_t, mode='0.block.t_data')
    output_check(scale_t_ref, scale_t, mode="0.block.t_scale")

    if bench:
        benchmark_func(triton_rms_norm_and_mxfp8_quant_forward, x, weight,
                       output_mode=0,
                       ref_bytes=M * N * 3)

        benchmark_func(triton_rms_norm_and_mxfp8_quant_forward, x, weight,
                       output_mode=1,
                       ref_bytes=M * N * 3)

        benchmark_func(triton_rms_norm_and_mxfp8_quant_forward, x, weight,
                       output_mode=2,
                       ref_bytes=M * N * 4)

if __name__ == '__main__':
    test_rmsnorm(M=16384, N=2048, bench=False)
    test_rmsnorm(M=16384, N=1664, bench=False)
    test_rmsnorm(M=1664, N=1664, bench=False)

    test_rmsnorm_and_mxfp8_quant(M=2048, N=1664, bench=False)

    test_rmsnorm(M=8192, N=4096, bench=False)
    test_rmsnorm(M=4096, N=8192, bench=False)
    test_rmsnorm_and_smooth_quant(M=16384, N=2048, bench=False)
    test_rmsnorm_and_smooth_quant(M=8192, N=4096, bench=False)
    test_rmsnorm_and_smooth_quant(M=4096, N=8192, bench=False)
    test_rmsnorm_and_block_quant(M=128, N=2048, bench=False)
    test_rmsnorm_and_block_quant(M=8192, N=4096, bench=False)


