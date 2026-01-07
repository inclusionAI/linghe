# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import pytest
import random

random.seed(7)

import torch
import torch_musa

from linghe.tools.benchmark import benchmark_func
from linghe.utils.silu import (
    triton_weighted_silu_forward,
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
from linghe.tools.util import output_check, torch_smooth_quant, torch_group_quant


def torch_silu(x):
    M, N = x.shape
    x1, x2 = torch.split(x, N // 2, dim=1)
    y = torch.sigmoid(x1) * x1 * x2
    return y


def torch_weighted_silu(x, weight):
    M, N = x.shape
    x1, x2 = torch.split(x, N // 2, dim=1)
    y = torch.sigmoid(x1) * x1 * x2 * weight
    return y


def torch_weighted_silu_backward(dy, x, weight):
    x = x.clone().detach().requires_grad_()
    weight = weight.clone().detach().requires_grad_()
    y = torch_weighted_silu(x, weight)
    y.backward(gradient=dy)
    return x.grad, weight.grad


def torch_silu_and_smooth_quant_forward(x, smooth_scale=None, round_scale=True):
    M, N = x.shape
    x = x.float()
    x1, x2 = torch.split(x, N // 2, dim=1)
    y = torch.sigmoid(x1) * x1 * x2

    # smooth
    y_q, y_scale, x_maxs = torch_smooth_quant(
        y, smooth_scale, reverse=False, round_scale=round_scale
    )
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


def torch_silu_and_smooth_quant_backward(
    grad,
    x,
    smooth_scale=None,
    transpose_smooth_scale=None,
    round_scale=True,
    reverse=True,
):
    grad = grad.float()
    x = x.float().detach().clone().requires_grad_()
    y = torch_silu(x)
    y.backward(gradient=grad)
    dx = x.grad

    q, dx_scale, ms = torch_smooth_quant(
        dx, smooth_scale, reverse=reverse, round_scale=round_scale
    )
    yt_q, yt_scale, ms = torch_smooth_quant(
        dx.t().contiguous(),
        transpose_smooth_scale,
        reverse=reverse,
        round_scale=round_scale,
    )
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


def torch_batch_weighted_silu_and_smooth_quant_forward(
    xs, weight, counts, smooth_scales=None, round_scale=True, reverse=False
):
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
        x = xs[s : s + c]
        y = torch_weighted_silu(x, weight[s : s + c])
        q, scale, ms = torch_smooth_quant(
            y, smooth_scales[i], reverse=reverse, round_scale=round_scale
        )
        qs.append(q)
        scales.append(scale)
        maxs.append(ms)

        s += c
    qs = torch.cat(qs, 0)
    scales = torch.cat(scales, 0)
    maxs = torch.cat(maxs, 0)
    return qs, scales, maxs


def torch_batch_weighted_silu_and_block_quant_forward(
    xs, weight, counts, round_scale=True
):
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
        x = xs[s : s + c]
        y = torch_weighted_silu(x, weight[s : s + c])
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


def torch_batch_weighted_silu_and_smooth_quant_backward(
    grad_output,
    x,
    weight,
    counts,
    smooth_scales=None,
    transpose_smooth_scale=None,
    round_scale=True,
    reverse=False,
):
    if sum(counts) == 0:
        device = x.device
        N = x.shape[1]
        dx_q = torch.empty((0, N), device=device, dtype=torch.float8_e4m3fn)
        dx_scale = torch.empty((0,), device=device, dtype=torch.float32)
        dw = torch.empty_like(weight)
        qts = torch.empty((0,), device=device, dtype=torch.float8_e4m3fn)
        qtscales = torch.zeros((N * len(counts),), device=device, dtype=torch.float32)
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
        q, scale, dx_max = torch_smooth_quant(
            dx[s : s + c], smooth_scales[i], reverse=reverse, round_scale=round_scale
        )
        dxt = dx[s : s + c].t().contiguous()
        dxt_s = transpose_smooth_scale[s : s + c]
        padding_size = (c + 31) // 32 * 32 - c
        if padding_size > 0:
            dxt = torch.nn.functional.pad(dxt, (0, padding_size, 0, 0))
            dxt_s = torch.nn.functional.pad(dxt_s, (0, padding_size))
        qt, t_scale, dx_max = torch_smooth_quant(
            dxt, dxt_s, reverse=reverse, round_scale=round_scale
        )

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


def torch_batch_weighted_silu_and_block_quant_backward(
    grad_output, x, weight, counts, round_scale=True
):
    if sum(counts) == 0:
        device = x.device
        N = x.shape[1]
        dx_q = torch.empty((0, N), device=device, dtype=torch.float8_e4m3fn)
        dx_scale = torch.empty((0,), device=device, dtype=torch.float32)
        dw = torch.empty_like(weight)
        qts = torch.empty((0,), device=device, dtype=torch.float8_e4m3fn)
        qtscales = torch.zeros((N * len(counts),), device=device, dtype=torch.float32)
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
        q, scale = torch_group_quant(dx[s : s + c], round_scale=round_scale)
        qt, qtscale = torch_group_quant(dx[s : s + c].t(), round_scale=round_scale)
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


@pytest.mark.parametrize("M,N", [(16384, 1024)])
@pytest.mark.parametrize("bench", [True])
def test_weighted_silu(M, N, bench):
    x = torch.randn((M, N), dtype=torch.bfloat16, device="musa:0")
    x = (x**3 // 10).clone().detach().requires_grad_()
    weight = torch.randn((M, 1), dtype=torch.bfloat16, device="musa:0")
    grad_output = torch.randn((M, N // 2), dtype=torch.bfloat16, device="musa:0")
    ref_y = torch_weighted_silu(x, weight)
    y = triton_weighted_silu_forward(x, weight)
    assert output_check(ref_y, y, "y")

    dx_ref, dw_ref = torch_weighted_silu_backward(grad_output, x, weight)
    dx, dw = triton_weighted_silu_backward(grad_output, x, weight)
    assert output_check(dx_ref, dx, "dx")
    assert output_check(dw_ref, dw, "dw")

    if bench:
        benchmark_func(
            triton_weighted_silu_forward, x, weight, n_repeat=100, ref_bytes=M * N * 3
        )
        benchmark_func(
            triton_weighted_silu_backward,
            grad_output,
            x,
            weight,
            n_repeat=100,
            ref_bytes=M * N * 5,
        )


@pytest.mark.parametrize("M,N", [(16384, 1024), (8192, 2048), (4096, 10240), (4096, 5120)])
@pytest.mark.parametrize("bench", [True])
def test_silu_and_smooth_quant(M, N, bench):
    if True:
        x = torch.randn((M, N), dtype=torch.bfloat16, device="musa:0")
        x = (x * 10).clone().detach().requires_grad_()
        grad_output = torch.randn((M, N // 2), dtype=torch.bfloat16, device="musa:0")
        smooth_scale = 1 + torch.rand((N // 2,), dtype=torch.float32, device="musa:0")
        grad_smooth_scale = 1 + torch.rand((N,), dtype=torch.float32, device="musa:0")
        transpose_grad_smooth_scale = 1 + torch.rand(
            (M,), dtype=torch.float32, device="musa:0"
        )
    else:
        d = torch.load("/ossfs/workspace/tmp/vis/silu.bin")
        x = d["x"].clone().detach().to("musa:0").requires_grad_()
        grad_output = d["g"].to("musa:0")
        grad_smooth_scale = d["smooth_scale"].to("musa:0")
        N = x.shape[-1]
        M = x.shape[0]
        smooth_scale = 1 + torch.rand((N // 2,), dtype=torch.float32, device="musa:0")

    y_q_ref, y_scale_ref, y_maxs_ref = torch_silu_and_smooth_quant_forward(
        x, smooth_scale=smooth_scale
    )
    y_q, y_scale, y_maxs = triton_silu_and_smooth_quant_forward(
        x, smooth_scale=smooth_scale, round_scale=True, calibrate=True
    )
    assert output_check(y_q_ref.float(), y_q.float(), "smooth.y_q")
    assert output_check(y_scale_ref, y_scale, "smooth.y_scale")
    assert output_check(y_maxs_ref, y_maxs, "smooth.y_max")

    dx_q_ref, dx_scale_ref, dxt_q_ref, dxt_scale_ref = (
        torch_silu_and_smooth_quant_backward(
            grad_output,
            x,
            smooth_scale=grad_smooth_scale,
            transpose_smooth_scale=transpose_grad_smooth_scale,
            reverse=True,
            round_scale=True,
        )
    )
    dx_q, dx_scale, dxt_q, dxt_scale = triton_silu_and_smooth_quant_backward(
        grad_output,
        x,
        smooth_scale=grad_smooth_scale,
        transpose_smooth_scale=transpose_grad_smooth_scale,
        reverse=True,
        round_scale=True,
    )

    assert output_check(dx_q_ref.float(), dx_q.float(), "smooth.dx_data")
    assert output_check(dx_scale_ref, dx_scale, "smooth.dx_scale")
    assert output_check(dxt_q_ref.float(), dxt_q.float(), "smooth.dxt_data")
    assert output_check(dxt_scale_ref, dxt_scale, "smooth.dxt_scale")

    if bench:
        benchmark_func(
            torch_silu_and_smooth_quant_forward,
            x,
            smooth_scale=smooth_scale,
            n_repeat=100,
            ref_bytes=M * N * 2.5,
        )
        benchmark_func(
            triton_silu_and_smooth_quant_forward,
            x,
            smooth_scale=smooth_scale,
            n_repeat=100,
            ref_bytes=M * N * 2.5,
        )
        benchmark_func(
            triton_silu_and_smooth_quant_backward,
            grad_output,
            x,
            smooth_scale=grad_smooth_scale,
            transpose_smooth_scale=transpose_grad_smooth_scale,
            reverse=True,
            round_scale=True,
        )


@pytest.mark.parametrize("M,N", [(16384, 1024)])
@pytest.mark.parametrize("bench", [True])
def test_silu_and_block_quant(M, N, bench):
    if True:
        x = torch.randn((M, N), dtype=torch.bfloat16, device="musa:0")
        x = (x * 10).clone().detach().requires_grad_()
        grad_output = torch.randn((M, N // 2), dtype=torch.bfloat16, device="musa:0")
    else:
        d = torch.load("/ossfs/workspace/tmp/vis/silu.bin")
        x = d["x"].clone().detach().to("musa:0").requires_grad_()
        grad_output = d["g"].to("musa:0")
        N = x.shape[-1]
        M = x.shape[0]

    y_q_ref, y_scale_ref, yt_q_ref, yt_scale_ref = torch_silu_and_block_quant_forward(
        x, round_scale=True
    )
    y_q, y_scale, yt_q, yt_scale = triton_silu_and_block_quant_forward(
        x, round_scale=True, output_mode=2
    )
    assert output_check(y_q_ref.float(), y_q.float(), "block.2.y_q")
    assert output_check(y_scale_ref, y_scale.t(), "block.2.y_scale")
    assert output_check(yt_q_ref, yt_q, "block.2.yt_q")
    assert output_check(yt_scale_ref, yt_scale.t(), "block.2.yt_scale")

    y_q, y_scale, yt_q, yt_scale = triton_silu_and_block_quant_forward(
        x, round_scale=True, output_mode=0
    )
    assert output_check(y_q_ref.float(), y_q.float(), "block.0.y_q")
    assert output_check(y_scale_ref, y_scale.t(), "block.0.y_scale")

    y_q, y_scale, yt_q, yt_scale = triton_silu_and_block_quant_forward(
        x, round_scale=True, output_mode=1
    )
    assert output_check(yt_q_ref.float(), yt_q.float(), "block.1.yt_q")
    assert output_check(yt_scale_ref, yt_scale.t(), "block.1.yt_scale")

    dx_q_ref, dx_scale_ref, dxt_q_ref, dxt_scale_ref = (
        torch_silu_and_block_quant_backward(grad_output, x, round_scale=True)
    )
    dx_q, dx_scale, dxt_q, dxt_scale = triton_silu_and_block_quant_backward(
        grad_output, x, round_scale=True
    )
    assert output_check(dx_q_ref.float(), dx_q.float(), "block.dx_q")
    assert output_check(dx_scale_ref.t(), dx_scale, "block.dx_scale")
    assert output_check(dxt_q_ref.float(), dxt_q.float(), "block.dxt_q")
    assert output_check(dxt_scale_ref.t(), dxt_scale, "block.dxt_scale")

    if bench:
        benchmark_func(
            triton_silu_and_block_quant_forward, x, n_repeat=100, ref_bytes=M * N * 3
        )
        benchmark_func(
            triton_silu_and_block_quant_backward,
            grad_output,
            x,
            n_repeat=100,
            ref_bytes=M * N * 5,
        )


@pytest.mark.parametrize("M,N", [(2048, 2048), (800, 2048), (0, 2048)])
@pytest.mark.parametrize("n_experts", [32])
@pytest.mark.parametrize("bench", [True])
def test_triton_batch_weighted_silu_and_smooth_quant(M, N, n_experts, bench):
    if True:
        count_list = [
            random.randint(M // 2, M // 2 * 3) // 16 * 16 for _ in range(n_experts)
        ]
        counts = torch.tensor(count_list, device="musa:0", dtype=torch.int32)
        bs = sum(count_list)

        x = torch.randn((bs, N), dtype=torch.bfloat16, device="musa:0") ** 3 / 4
        weight = torch.randn((bs, 1), dtype=torch.float32, device="musa:0")
        smooth_scales = (
            1
            + torch.rand((n_experts, N // 2), dtype=torch.float32, device="musa:0") * 10
        )
    else:
        d = torch.load("/ossfs/workspace/Megatron-LM/silu.bin")
        counts = d["counts"].cuda()
        x = d["x"].cuda()
        weight = d["weight"].cuda()
        smooth_scales = d["smooth_scale"].cuda()
        bs = sum(counts.tolist())
        N = x.shape[-1]
        n_experts = counts.shape[0]

    grad_output = torch.randn((bs, N // 2), dtype=torch.bfloat16, device="musa:0") ** 3
    grad_smooth_scales = (
        1 + torch.rand((n_experts, N), dtype=torch.float32, device="musa:0") * 10
    )
    transpose_grad_smooth_scales = (
        1 + torch.rand((bs,), dtype=torch.float32, device="musa:0") * 10
    )
    round_scale = True

    x_q_ref, x_scale_ref, x_max_ref = (
        torch_batch_weighted_silu_and_smooth_quant_forward(
            x,
            weight,
            counts,
            smooth_scales=smooth_scales,
            round_scale=round_scale,
            reverse=False,
        )
    )
    x_q, x_scale, maxs = triton_batch_weighted_silu_and_smooth_quant_forward(
        x,
        weight,
        counts,
        smooth_scale=smooth_scales,
        round_scale=round_scale,
        reverse=False,
    )
    assert output_check(x_q_ref.float(), x_q.float(), "smooth.data")
    assert output_check(x_scale_ref, x_scale, "smooth.scale")

    dx_ref, dx_scale_ref, dw_ref, dxt_ref, dxt_scale_ref = (
        torch_batch_weighted_silu_and_smooth_quant_backward(
            grad_output,
            x,
            weight,
            count_list,
            smooth_scales=grad_smooth_scales,
            transpose_smooth_scale=transpose_grad_smooth_scales,
            round_scale=round_scale,
            reverse=False,
        )
    )
    dx, dx_scale, dw, dxt, dxt_scale = (
        triton_batch_weighted_silu_and_smooth_quant_backward(
            grad_output,
            x,
            weight,
            counts,
            smooth_scale=grad_smooth_scales,
            transpose_smooth_scale=transpose_grad_smooth_scales,
            splits=count_list,
            round_scale=round_scale,
            reverse=False,
        )
    )
    assert output_check(dx_ref.float(), dx.float(), "smooth.dx")
    assert output_check(dx_scale_ref, dx_scale, "smooth.dx_scale")
    assert output_check(dw_ref, dw, "smooth.dw")
    assert output_check(dxt_ref.float(), dxt.float(), "smooth.dxt")
    assert output_check(dxt_scale_ref, dxt_scale.view(-1), "smooth.dxt_scale")

    if bench:
        ref_time = None
        benchmark_func(
            triton_batch_weighted_silu_and_smooth_quant_forward,
            x,
            weight,
            counts,
            smooth_scale=smooth_scales,
            round_scale=True,
            ref_bytes=n_experts * M * N * 2.5,
            ref_time=ref_time,
        )
        benchmark_func(
            triton_batch_weighted_silu_and_smooth_quant_backward,
            grad_output,
            x,
            weight,
            counts,
            smooth_scale=smooth_scales,
            transpose_smooth_scale=transpose_grad_smooth_scales,
            splits=count_list,
            round_scale=True,
            ref_bytes=n_experts * M * N * 4,
            ref_time=ref_time,
        )


@pytest.mark.parametrize("M,N", [(4096, 2048), (1008, 2048)])
@pytest.mark.parametrize("n_experts", [32])
@pytest.mark.parametrize("bench", [True])
def test_triton_batch_weighted_silu_and_block_quant(M, N, n_experts, bench):
    if True:
        count_list = [
            random.randint(M // 2, M // 2 * 3) // 16 * 16 for _ in range(n_experts)
        ]
        counts = torch.tensor(count_list, device="musa:0", dtype=torch.int32)
        bs = sum(count_list)

        x = torch.randn((bs, N), dtype=torch.bfloat16, device="musa:0") ** 3 / 10
        weight = torch.randn((bs, 1), dtype=torch.float32, device="musa:0")
    else:
        d = torch.load("/ossfs/workspace/Megatron-LM/silu.bin")
        counts = d["counts"].cuda()
        x = d["x"].cuda()
        weight = d["weight"].cuda()
        smooth_scales = d["smooth_scale"].cuda()
        bs = sum(counts.tolist())
        N = x.shape[-1]
        n_experts = counts.shape[0]

    grad_output = torch.randn((bs, N // 2), dtype=torch.bfloat16, device="musa:0") ** 3
    round_scale = True

    x_q_ref, x_scale_ref, xt_q_ref, xt_scale_ref = (
        torch_batch_weighted_silu_and_block_quant_forward(
            x, weight, counts, round_scale=round_scale
        )
    )
    x_q, x_scale, xt_q, xt_scale = triton_batch_weighted_silu_and_block_quant_forward(
        x, weight, counts, count_list, round_scale=round_scale, output_mode=2
    )

    assert output_check(x_q_ref.float(), x_q.float(), "block.q")
    assert output_check(x_scale_ref, x_scale, "block.scale")
    assert output_check(xt_q_ref.float(), xt_q.float(), "block.qt")
    assert output_check(xt_scale_ref, xt_scale, "block.t_scale")

    dx_ref, dx_scale_ref, dw_ref, dxt_ref, dxt_scale_ref = (
        torch_batch_weighted_silu_and_block_quant_backward(
            grad_output, x, weight, counts, round_scale=round_scale
        )
    )
    dx, dx_scale, dw, dxt, dxt_scale = (
        triton_batch_weighted_silu_and_block_quant_backward(
            grad_output, x, weight, counts, splits=count_list, round_scale=round_scale
        )
    )
    assert output_check(dx_ref.float(), dx.float(), "block.dx")
    assert output_check(dx_scale_ref, dx_scale, "block.dx_scale")
    assert output_check(dw_ref, dw, "block.dw")
    assert output_check(dxt_ref.float(), dxt.float(), "block.dxt")
    assert output_check(dxt_scale_ref, dxt_scale, "block.dxt_scale")

    if bench:
        ref_time = None
        benchmark_func(
            triton_batch_weighted_silu_and_block_quant_forward,
            x,
            weight,
            counts,
            round_scale=True,
            splits=count_list,
            output_mode=0,
            n_repeat=100,
            ref_bytes=n_experts * M * N * 2.5,
            ref_time=ref_time,
        )
        benchmark_func(
            triton_batch_weighted_silu_and_block_quant_forward,
            x,
            weight,
            counts,
            round_scale=True,
            splits=count_list,
            output_mode=2,
            n_repeat=100,
            ref_bytes=n_experts * M * N * 3,
            ref_time=ref_time,
        )
        benchmark_func(
            triton_batch_weighted_silu_and_block_quant_backward,
            grad_output,
            x,
            weight,
            counts,
            round_scale=True,
            splits=count_list,
            n_repeat=100,
            ref_bytes=n_experts * M * N * 4,
            ref_time=ref_time,
        )