# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import random

random.seed(7)

import torch

from linghe.tools.benchmark import benchmark_func
from linghe.utils.silu import (triton_weighted_silu_forward,
                               triton_weighted_silu_backward,
                               triton_batch_weighted_silu_and_smooth_quant_backward,
                               triton_batch_weighted_silu_and_smooth_quant_forward,
                               triton_batch_weighted_silu_and_block_quant_backward,
                               triton_batch_weighted_silu_and_block_quant_forward,
                               triton_silu_and_smooth_quant_backward,
                               triton_silu_and_smooth_quant_forward,
                               triton_silu_and_block_quant_backward,
                               triton_silu_and_block_quant_forward,
                               )
from linghe.tools.util import (torch_smooth_quant,
                               torch_group_quant)
from linghe.tools.check import output_check


def torch_silu(x):
    M, N = x.shape
    x1, x2 = torch.split(x, N // 2, dim=1)
    y = torch.sigmoid(x1) * x1 * x2
    return y


def torch_weighted_silu(x, weight):
    dtype = x.dtype
    x = x.float()
    weight = weight.float()
    M, N = x.shape
    x1, x2 = torch.split(x, N // 2, dim=1)
    y = torch.sigmoid(x1) * x1 * x2 * weight
    return y.to(dtype)


def torch_weighted_silu_backward(dy, x, weight):
    dtype = x.dtype
    x = x.float()
    x = x.clone().detach().requires_grad_()
    weight = weight.clone().detach().requires_grad_()
    y = torch_weighted_silu(x, weight)
    y.backward(gradient=dy)
    return x.grad.to(dtype), weight.grad


def torch_silu_and_smooth_quant_forward(x, smooth_scale=None, round_scale=True):
    M, N = x.shape
    x = x.float()
    x1, x2 = torch.split(x, N // 2, dim=1)
    y = torch.sigmoid(x1) * x1 * x2

    # smooth
    y_q, y_scale, x_maxs = torch_smooth_quant(y, smooth_scale, reverse=False,
                                              round_scale=round_scale)
    # y_smooth = y / smooth_scale
    # x_maxs = y.abs().float().amax(0)
    # y_scale = y_smooth.abs().amax(1) / 448
    # if round_scale:
    #     y_scale = torch.exp2(torch.ceil(torch.log2(y_scale)))
    # y_q = (y_smooth / y_scale[:, None]).to(torch.float8_e4m3fn)
    return y_q, y_scale, x_maxs


def torch_silu_and_block_quant_forward(x, round_scale=True):
    M, N = x.shape
    x = x.float()
    x1, x2 = torch.split(x, N // 2, dim=1)
    y = torch.sigmoid(x1) * x1 * x2
    # blockwise
    y_q, y_scale = torch_group_quant(y, round_scale=round_scale)
    yt_q, yt_scale = torch_group_quant(y.t(), round_scale=round_scale)

    return y_q, y_scale, yt_q, yt_scale


def torch_silu_and_smooth_quant_backward(grad, x, smooth_scale=None,
                                         transpose_smooth_scale=None,
                                         round_scale=True, reverse=True):
    grad = grad.float()
    x = x.float().detach().clone().requires_grad_()
    y = torch_silu(x)
    y.backward(gradient=grad)
    dx = x.grad

    q, dx_scale, ms = torch_smooth_quant(dx, smooth_scale, reverse=reverse,
                                         round_scale=round_scale)
    yt_q, yt_scale, ms = torch_smooth_quant(dx.t().contiguous(),
                                            transpose_smooth_scale,
                                            reverse=reverse,
                                            round_scale=round_scale)
    return q, dx_scale, yt_q, yt_scale


def torch_silu_and_block_quant_backward(grad, x, round_scale=True):
    grad = grad.float()
    x = x.float().detach().clone().requires_grad_()
    y = torch_silu(x)
    y.backward(gradient=grad)
    dx = x.grad
    # blockwise
    q, dx_scale = torch_group_quant(dx, round_scale=round_scale)
    yt_q, yt_scale = torch_group_quant(dx.t(), round_scale=round_scale)

    return q, dx_scale, yt_q, yt_scale



def torch_batch_weighted_silu_and_smooth_quant_forward(xs, weight,
                                                       counts,
                                                       smooth_scales=None,
                                                       round_scale=True,
                                                       reverse=False):
    counts = counts.tolist()
    N = xs.shape[1]
    if sum(counts) == 0:
        device = xs.device
        qs = torch.empty((0, N // 2), device=device, dtype=torch.float8_e4m3fn)
        scales = torch.empty((0,), device=device, dtype=torch.float32)
        maxs = torch.zeros((len(counts), N), device=device, dtype=torch.float32)
        return qs, scales, maxs

    xs = xs.float()
    weight = weight.float()
    smooth_scales = smooth_scales.float()

    qs = []
    scales = []
    maxs = []
    s = 0
    for i, c in enumerate(counts):
        x = xs[s:s + c]
        y = torch_weighted_silu(x, weight[s:s + c])
        q, scale, ms = torch_smooth_quant(y, smooth_scales[i], reverse=reverse,
                                          round_scale=round_scale)
        qs.append(q)
        scales.append(scale)
        maxs.append(ms)

        s += c
    qs = torch.cat(qs, 0)
    scales = torch.cat(scales, 0)
    maxs = torch.cat(maxs, 0)
    return qs, scales, maxs


def torch_batch_weighted_silu_and_block_quant_forward(xs, weight,
                                                      counts,
                                                      round_scale=True):
    counts = counts.tolist()
    N = xs.shape[1]
    if sum(counts) == 0:
        device = xs.device
        qs = torch.empty((0, N // 2), device=device, dtype=torch.float8_e4m3fn)
        scales = torch.empty((0,), device=device, dtype=torch.float32)
        qts = torch.empty((0,), device=device, dtype=torch.float8_e4m3fn)
        qtscales = torch.zeros((0,), device=device, dtype=torch.float32)
        return qs, scales, qts, qtscales

    xs = xs.float()
    weight = weight.float()

    qs = []
    scales = []
    qts = []
    qtscales = []
    s = 0
    for i, c in enumerate(counts):
        x = xs[s:s + c]
        y = torch_weighted_silu(x, weight[s:s + c])
        q, scale = torch_group_quant(y, round_scale=round_scale)
        qt, qtscale = torch_group_quant(y.t(), round_scale=round_scale)
        qs.append(q)
        scales.append(scale.t().contiguous().view(-1))
        qts.append(qt.view(-1))
        qtscales.append(qtscale.t().contiguous().view(-1))

        s += c
    qs = torch.cat(qs, 0)
    scales = torch.cat(scales, 0)
    qts = torch.cat(qts, 0)
    qtscales = torch.cat(qtscales, 0)
    return qs, scales, qts, qtscales


def torch_batch_weighted_silu_and_smooth_quant_backward(grad_output, x, weight,
                                                        counts,
                                                        smooth_scales=None,
                                                        transpose_smooth_scale=None,
                                                        round_scale=True,
                                                        reverse=False):
    if sum(counts) == 0:
        device = x.device
        N = x.shape[1]
        dx_q = torch.empty((0, N), device=device, dtype=torch.float8_e4m3fn)
        dx_scale = torch.empty((0,), device=device, dtype=torch.float32)
        dw = torch.empty_like(weight)
        qts = torch.empty((0,), device=device, dtype=torch.float8_e4m3fn)
        qtscales = torch.zeros((N * len(counts),), device=device,
                               dtype=torch.float32)
        return dx_q, dx_scale, dw, qts, qtscales

    grad_output = grad_output.float()
    x = x.float()
    weight = weight.float()
    smooth_scales = smooth_scales.float()
    transpose_smooth_scale = transpose_smooth_scale.float()

    dx, dw = torch_weighted_silu_backward(grad_output, x, weight)
    qs = []
    scales = []
    qts = []
    qtscales = []
    s = 0
    for i, c in enumerate(counts):
        q, scale, dx_max = torch_smooth_quant(dx[s:s + c], smooth_scales[i],
                                              reverse=reverse,
                                              round_scale=round_scale)
        dxt = dx[s:s + c].t().contiguous()
        dxt_s = transpose_smooth_scale[s:s + c]
        padding_size = (c + 31) // 32 * 32 - c
        if padding_size > 0:
            dxt = torch.nn.functional.pad(dxt, (0, padding_size, 0, 0))
            dxt_s = torch.nn.functional.pad(dxt_s, (0, padding_size))
        qt, t_scale, dx_max = torch_smooth_quant(dxt, dxt_s,
                                                 reverse=reverse,
                                                 round_scale=round_scale)

        qs.append(q)
        scales.append(scale)
        qts.append(qt.view(-1))
        qtscales.append(t_scale.view(-1))
        s += c
    dx_q = torch.cat(qs, 0)
    dx_scale = torch.cat(scales, 0)
    qts = torch.cat(qts, 0)
    qtscales = torch.cat(qtscales, 0)
    return dx_q, dx_scale, dw, qts, qtscales


def torch_batch_weighted_silu_and_block_quant_backward(grad_output, x, weight,
                                                       counts,
                                                       round_scale=True):
    if sum(counts) == 0:
        device = x.device
        N = x.shape[1]
        dx_q = torch.empty((0, N), device=device, dtype=torch.float8_e4m3fn)
        dx_scale = torch.empty((0,), device=device, dtype=torch.float32)
        dw = torch.empty_like(weight)
        qts = torch.empty((0,), device=device, dtype=torch.float8_e4m3fn)
        qtscales = torch.zeros((0,), device=device,
                               dtype=torch.float32)
        return dx_q, dx_scale, dw, qts, qtscales

    grad_output = grad_output.float()
    x = x.float()
    weight = weight.float()

    dx, dw = torch_weighted_silu_backward(grad_output, x, weight)
    qs = []
    scales = []
    qts = []
    qtscales = []
    s = 0
    for i, c in enumerate(counts):
        q, scale = torch_group_quant(dx[s:s + c], round_scale=round_scale)
        qt, qtscale = torch_group_quant(dx[s:s + c].t(),
                                        round_scale=round_scale)
        qs.append(q)
        scales.append(scale.t().contiguous().view(-1))
        qts.append(qt.view(-1))
        qtscales.append(qtscale.t().contiguous().view(-1))

        s += c
    dx_q = torch.cat(qs, 0)
    dx_scale = torch.cat(scales, 0)
    qts = torch.cat(qts, 0)
    qtscales = torch.cat(qtscales, 0)
    return dx_q, dx_scale, dw, qts, qtscales



def test_weighted_silu(M=4096, N=4096, asm=False, coef=1.0, bench=False):
    x = torch.randn((M, N), dtype=torch.bfloat16, device='cuda:0')
    x = (x * coef).clone().detach().requires_grad_()
    weight = torch.randn((M, 1), dtype=torch.float32, device='cuda:0')
    grad_output = torch.randn((M, N // 2), dtype=torch.bfloat16,
                              device='cuda:0')
    ref_y = torch_weighted_silu(x, weight)
    y = triton_weighted_silu_forward(x, weight, asm=asm)
    output_check(ref_y, y, 'y')

    dx_ref, dw_ref = torch_weighted_silu_backward(grad_output, x, weight)
    dx, dw = triton_weighted_silu_backward(grad_output, x, weight)
    output_check(dx_ref, dx, 'dx')
    output_check(dw_ref, dw, 'dw', rtol=3e-3, atol=3e-3)

    if bench:
        benchmark_func(triton_weighted_silu_forward, x, weight, asm=asm,
                       n_repeat=100,
                       ref_bytes=M * N * 3)
        benchmark_func(triton_weighted_silu_backward, grad_output, x, weight,
                       n_repeat=100, ref_bytes=M * N * 5)


def test_silu_and_smooth_quant(M=4096, N=4096, coef=1.0, grad_coef=1.0,
                               bench=False):
    x = torch.randn((M, N), dtype=torch.bfloat16, device='cuda:0')
    x = (x * coef).clone().detach().requires_grad_()
    grad_output = torch.randn((M, N // 2), dtype=torch.bfloat16,
                              device='cuda:0') * grad_coef
    smooth_scale = 1 + torch.rand((N // 2,), dtype=torch.float32,
                                  device='cuda:0')
    grad_smooth_scale = 1 + torch.rand((N,), dtype=torch.float32,
                                       device='cuda:0')
    transpose_grad_smooth_scale = 1 + torch.rand((M,), dtype=torch.float32,
                                                 device='cuda:0')

    round_scale = False
    y_q_ref, y_scale_ref, y_maxs_ref = torch_silu_and_smooth_quant_forward(x,
                                                                           smooth_scale=smooth_scale,
                                                                           round_scale=round_scale)
    y_q, y_scale, y_maxs = triton_silu_and_smooth_quant_forward(x,
                                                                smooth_scale=smooth_scale,
                                                                round_scale=round_scale,
                                                                calibrate=True)
    output_check(y_q_ref, y_q, 'smooth.y_q', rtol=0.125)
    output_check(y_scale_ref, y_scale, 'smooth.y_scale')
    output_check(y_maxs_ref, y_maxs, 'smooth.y_max')

    dx_q_ref, dx_scale_ref, dxt_q_ref, dxt_scale_ref = torch_silu_and_smooth_quant_backward(
        grad_output, x,
        smooth_scale=grad_smooth_scale,
        transpose_smooth_scale=transpose_grad_smooth_scale,
        reverse=True,
        round_scale=True)
    dx_q, dx_scale, dxt_q, dxt_scale = triton_silu_and_smooth_quant_backward(
        grad_output, x,
        smooth_scale=grad_smooth_scale,
        transpose_smooth_scale=transpose_grad_smooth_scale,
        reverse=True,
        round_scale=True)

    output_check(dx_q_ref, dx_q, 'smooth.dx_data', rtol=0.125)
    output_check(dx_scale_ref, dx_scale, 'smooth.dx_scale')
    output_check(dxt_q_ref, dxt_q, 'smooth.dxt_data', rtol=0.125)
    output_check(dxt_scale_ref, dxt_scale, 'smooth.dxt_scale')

    if bench:
        benchmark_func(torch_silu_and_smooth_quant_forward, x,
                       smooth_scale=smooth_scale,
                       n_repeat=100, ref_bytes=M * N * 2.5)
        benchmark_func(triton_silu_and_smooth_quant_forward, x,
                       smooth_scale=smooth_scale,
                       n_repeat=100, ref_bytes=M * N * 2.5)
        benchmark_func(triton_silu_and_smooth_quant_backward, grad_output, x,
                       smooth_scale=grad_smooth_scale,
                       transpose_smooth_scale=transpose_grad_smooth_scale,
                       n_repeat=100, ref_bytes=M * N * 5)


def test_silu_and_block_quant(M=4096, N=4096, coef=1.0, grad_coef=1.0,
                              bench=False):
    x = torch.randn((M, N), dtype=torch.bfloat16, device='cuda:0')
    x = (x * coef).clone().detach().requires_grad_()
    grad_output = torch.randn((M, N // 2), dtype=torch.bfloat16,
                              device='cuda:0') * grad_coef

    round_scale = False
    y_q_ref, y_scale_ref, yt_q_ref, yt_scale_ref = torch_silu_and_block_quant_forward(
        x, round_scale=round_scale)

    y_q, y_scale, yt_q, yt_scale = triton_silu_and_block_quant_forward(x,
                                                                       round_scale=round_scale,
                                                                       output_mode=0)
    output_check(y_q_ref, y_q, 'block.0.y_q', rtol=0.125)
    output_check(y_scale_ref, y_scale.t(), 'block.0.y_scale')

    y_q, y_scale, yt_q, yt_scale = triton_silu_and_block_quant_forward(x,
                                                                       round_scale=round_scale,
                                                                       output_mode=1)
    output_check(yt_q_ref, yt_q, 'block.1.yt_q', rtol=0.125)
    output_check(yt_scale_ref, yt_scale.t(), 'block.1.yt_scale')

    y_q, y_scale, yt_q, yt_scale = triton_silu_and_block_quant_forward(x,
                                                                       round_scale=round_scale,
                                                                       output_mode=2)
    output_check(y_q_ref, y_q, 'block.2.y_q', rtol=0.125)
    output_check(y_scale_ref, y_scale.t(), 'block.2.y_scale')
    output_check(yt_q_ref, yt_q, 'block.2.yt_q', rtol=0.125)
    output_check(yt_scale_ref, yt_scale.t(), 'block.2.yt_scale')

    dx_q_ref, dx_scale_ref, dxt_q_ref, dxt_scale_ref = torch_silu_and_block_quant_backward(
        grad_output, x,
        round_scale=round_scale)
    dx_q, dx_scale, dxt_q, dxt_scale = triton_silu_and_block_quant_backward(
        grad_output, x,
        round_scale=round_scale)
    output_check(dx_q_ref, dx_q, 'block.dx_q', rtol=0.125)
    output_check(dx_scale_ref.t(), dx_scale, 'block.dx_scale')
    output_check(dxt_q_ref, dxt_q, 'block.dxt_q', rtol=0.125)
    output_check(dxt_scale_ref.t(), dxt_scale, 'block.dxt_scale')

    if bench:
        benchmark_func(triton_silu_and_block_quant_forward, x,
                       round_scale=round_scale, output_mode=0,
                       n_repeat=100, ref_bytes=M * N * 3)
        benchmark_func(triton_silu_and_block_quant_forward, x,
                       round_scale=round_scale, output_mode=1,
                       n_repeat=100, ref_bytes=M * N * 3)
        benchmark_func(triton_silu_and_block_quant_forward, x,
                       round_scale=round_scale, output_mode=2,
                       n_repeat=100, ref_bytes=M * N * 3)
        benchmark_func(triton_silu_and_block_quant_backward, grad_output, x,
                       n_repeat=100, ref_bytes=M * N * 5)


def test_triton_batch_weighted_silu_and_smooth_quant(M=1024, N=4096,
                                                     n_experts=32,
                                                     coef=1.0,
                                                     grad_coef=1.0,
                                                     bench=False):
    count_list = [random.randint(M // 2, M // 2 * 3) // 16 * 16 for _ in
                  range(n_experts)]
    counts = torch.tensor(count_list, device='cuda:0', dtype=torch.int32)
    bs = sum(count_list)

    x = torch.randn((bs, N), dtype=torch.bfloat16, device='cuda:0') * coef
    weight = torch.randn((bs, 1), dtype=torch.float32, device='cuda:0')
    smooth_scales = 1 + torch.rand((n_experts, N // 2), dtype=torch.float32,
                                   device='cuda:0') * 10

    grad_output = torch.randn((bs, N // 2), dtype=torch.bfloat16,
                              device='cuda:0') * grad_coef
    grad_smooth_scales = 1 + torch.rand((n_experts, N), dtype=torch.float32,
                                        device='cuda:0') * 10
    transpose_grad_smooth_scales = 1 + torch.rand((bs,), dtype=torch.float32,
                                                  device='cuda:0') * 10
    round_scale = True
    rtol = 2 if round_scale else 0.125
    x_q_ref, x_scale_ref, x_max_ref = torch_batch_weighted_silu_and_smooth_quant_forward(
        x,
        weight,
        counts,
        smooth_scales=smooth_scales,
        round_scale=round_scale,
        reverse=False)
    x_q, x_scale, maxs = triton_batch_weighted_silu_and_smooth_quant_forward(x,
                                                                             weight,
                                                                             counts,
                                                                             smooth_scale=smooth_scales,
                                                                             round_scale=round_scale,
                                                                             reverse=False)
    output_check(x_q_ref, x_q, 'smooth.data', rtol=rtol)
    output_check(x_scale_ref, x_scale, 'smooth.scale')

    dx_ref, dx_scale_ref, dw_ref, dxt_ref, dxt_scale_ref = torch_batch_weighted_silu_and_smooth_quant_backward(
        grad_output, x, weight, count_list,
        smooth_scales=grad_smooth_scales,
        transpose_smooth_scale=transpose_grad_smooth_scales,
        round_scale=round_scale, reverse=False)
    dx, dx_scale, dw, dxt, dxt_scale = triton_batch_weighted_silu_and_smooth_quant_backward(
        grad_output, x, weight, counts,
        smooth_scale=grad_smooth_scales,
        transpose_smooth_scale=transpose_grad_smooth_scales,
        splits=count_list,
        round_scale=round_scale,
        reverse=False)
    output_check(dx_ref, dx, 'smooth.dx', rtol=rtol)
    output_check(dx_scale_ref, dx_scale, 'smooth.dx_scale')
    rate = coef ** 0.75 if coef > 1 else 1
    output_check(dw_ref, dw, 'smooth.dw', rtol=1e-3 * rate, atol=1e-3 * rate)
    output_check(dxt_ref, dxt, 'smooth.dxt', rtol=rtol)
    output_check(dxt_scale_ref, dxt_scale.view(-1), 'smooth.dxt_scale')

    if bench:
        ref_time = None
        benchmark_func(triton_batch_weighted_silu_and_smooth_quant_forward, x,
                       weight,
                       counts, smooth_scale=smooth_scales, round_scale=True,
                       ref_bytes=n_experts * M * N * 2.5, ref_time=ref_time)
        benchmark_func(triton_batch_weighted_silu_and_smooth_quant_backward,
                       grad_output, x, weight, counts,
                       smooth_scale=smooth_scales,
                       transpose_smooth_scale=transpose_grad_smooth_scales,
                       splits=count_list,
                       round_scale=True,
                       ref_bytes=n_experts * M * N * 4, ref_time=ref_time)


def test_triton_batch_weighted_silu_and_block_quant(M=1024, N=4096,
                                                    n_experts=32,
                                                    bench=False,
                                                    coef=1.0,
                                                    grad_coef=1.0):
    count_list = [random.randint(M // 2, M // 2 * 3) // 16 * 16 for _ in
                  range(n_experts)]
    counts = torch.tensor(count_list, device='cuda:0', dtype=torch.int32)
    bs = sum(count_list)

    x = torch.randn((bs, N), dtype=torch.bfloat16,
                    device='cuda:0') * coef
    if bs > 3:
        x[:3] = 0.0
    weight = torch.randn((bs, 1), dtype=torch.float32, device='cuda:0')

    grad_output = torch.randn((bs, N // 2), dtype=torch.bfloat16,
                              device='cuda:0') * grad_coef
    round_scale = False
    rtol = 2 if round_scale else 0.125

    x_q_ref, x_scale_ref, xt_q_ref, xt_scale_ref = torch_batch_weighted_silu_and_block_quant_forward(
        x,
        weight,
        counts,
        round_scale=round_scale)
    x_q, x_scale, xt_q, xt_scale = triton_batch_weighted_silu_and_block_quant_forward(
        x,
        weight,
        counts,
        count_list,
        round_scale=round_scale,
        output_mode=2)

    output_check(x_q_ref, x_q, 'block.q', rtol=rtol)
    output_check(x_scale_ref, x_scale.view(-1), 'block.scale')
    output_check(xt_q_ref, xt_q.view(-1), 'block.qt', rtol=rtol)
    output_check(xt_scale_ref, xt_scale.view(-1), 'block.t_scale')

    x_q, x_scale, xt_q, xt_scale = triton_batch_weighted_silu_and_block_quant_forward(
        x,
        weight,
        counts,
        count_list,
        round_scale=round_scale,
        output_mode=0)
    output_check(x_q_ref, x_q, 'block.q', rtol=rtol)
    output_check(x_scale_ref, x_scale.view(-1), 'block.scale')

    x_q, x_scale, xt_q, xt_scale = triton_batch_weighted_silu_and_block_quant_forward(
        x,
        weight,
        counts,
        count_list,
        round_scale=round_scale,
        output_mode=1)
    output_check(xt_q_ref, xt_q.view(-1), 'block.qt', rtol=rtol)
    output_check(xt_scale_ref, xt_scale.view(-1), 'block.t_scale')

    dx_ref, dx_scale_ref, dw_ref, dxt_ref, dxt_scale_ref = torch_batch_weighted_silu_and_block_quant_backward(
        grad_output, x, weight, counts,
        round_scale=round_scale)
    dx, dx_scale, dw, dxt, dxt_scale = triton_batch_weighted_silu_and_block_quant_backward(
        grad_output, x, weight, counts, splits=count_list,
        round_scale=round_scale)
    output_check(dx_ref, dx, 'block.dx', rtol=rtol)
    output_check(dx_scale_ref, dx_scale.view(-1), 'block.dx_scale')
    rate = (coef * grad_coef) ** 0.75 if coef * grad_coef > 1 else 1
    output_check(dw_ref, dw, 'block.dw', rtol=1e-3 * rate, atol=1e-3 * rate)
    output_check(dxt_ref, dxt.view(-1), 'block.dxt', rtol=rtol)
    output_check(dxt_scale_ref, dxt_scale.view(-1), 'block.dxt_scale')

    if bench:
        ref_time = None
        benchmark_func(triton_batch_weighted_silu_and_block_quant_forward, x,
                       weight,
                       counts, round_scale=True, splits=count_list,
                       output_mode=0, n_repeat=100,
                       ref_bytes=n_experts * M * N * 2.5, ref_time=ref_time)
        benchmark_func(triton_batch_weighted_silu_and_block_quant_forward, x,
                       weight,
                       counts, round_scale=True, splits=count_list,
                       output_mode=1, n_repeat=100,
                       ref_bytes=n_experts * M * N * 3, ref_time=ref_time)
        benchmark_func(triton_batch_weighted_silu_and_block_quant_forward, x,
                       weight,
                       counts, round_scale=True, splits=count_list,
                       output_mode=2, n_repeat=100,
                       ref_bytes=n_experts * M * N * 3, ref_time=ref_time)
        benchmark_func(triton_batch_weighted_silu_and_block_quant_backward,
                       grad_output, x, weight, counts,
                       round_scale=True, splits=count_list, n_repeat=100,
                       ref_bytes=n_experts * M * N * 4, ref_time=ref_time)



if __name__ == '__main__':
    test_weighted_silu(M=16384, N=4096, coef=1.0, asm=False, bench=False)
    test_weighted_silu(M=16384, N=4096, coef=1.0, asm=True, bench=False)
    test_weighted_silu(M=8192, N=1536, bench=False)
    test_weighted_silu(M=0, N=1536, bench=False)

    test_silu_and_smooth_quant(M=16384, N=1024, bench=False)
    test_silu_and_smooth_quant(M=8192, N=2048, bench=False)
    test_silu_and_smooth_quant(M=4096, N=10240, bench=False)
    test_silu_and_smooth_quant(M=4096, N=5120, bench=False)

    test_silu_and_block_quant(M=16384, N=1024, bench=False)
    test_silu_and_block_quant(M=8192, N=4096, bench=False)
    test_silu_and_block_quant(M=16384, N=1536, bench=False)
    test_silu_and_block_quant(M=4096, N=1536 * 8, bench=False)
    test_silu_and_block_quant(M=4096, N=1536 * 8, coef=100.0, grad_coef=100.0,
                              bench=False)
    test_silu_and_block_quant(M=4096, N=1536 * 8, coef=0.0, grad_coef=0.0,
                              bench=False)

    test_triton_batch_weighted_silu_and_smooth_quant(M=0, N=2048, n_experts=32,
                                                     bench=False)
    test_triton_batch_weighted_silu_and_smooth_quant(M=2048, N=2048,
                                                     n_experts=32, bench=False)

    test_triton_batch_weighted_silu_and_block_quant(M=0, N=1536, n_experts=32,
                                                    bench=False)
    test_triton_batch_weighted_silu_and_block_quant(M=2048, N=8192,
                                                    n_experts=32, bench=False)
    test_triton_batch_weighted_silu_and_block_quant(M=12080, N=1536,
                                                    n_experts=32, coef=100.0,
                                                    grad_coef=100.0,
                                                    bench=False)
    test_triton_batch_weighted_silu_and_block_quant(M=12080, N=1536,
                                                    n_experts=32, coef=0.0,
                                                    grad_coef=0.0, bench=False)
