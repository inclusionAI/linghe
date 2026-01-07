# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import pytest

import torch
import torch.nn.functional as F
import torch_musa

from linghe.utils.norm import (
    triton_rms_norm_and_smooth_quant_forward,
    triton_rms_norm_and_block_quant_forward,
    triton_rms_norm_backward,
    triton_rms_norm_forward,
    triton_group_rms_norm_gate_forward,
    triton_group_rms_norm_gate_backward,
)
from linghe.tools.util import output_check, torch_smooth_quant, torch_group_quant
from linghe.tools.benchmark import benchmark_func


def torch_rms_forward(x, weight):
    x = x.float()
    weight = weight.float()
    N = x.shape[-1]
    rmsnorm = torch.nn.RMSNorm(
        normalized_shape=N, eps=1e-6, dtype=torch.float32, device=x.device
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
        normalized_shape=N, eps=1e-6, dtype=torch.float32, device=x.device
    )
    with torch.no_grad():
        rmsnorm.weight.copy_(weight)
    x = x.clone().detach().requires_grad_()
    y = rmsnorm(x)
    y.backward(gradient=dy)
    return x.grad, rmsnorm.weight.grad


def torch_rms_and_smooth_quant_forward(x, weight, smooth_scale=None, round_scale=False):
    x = x.float()
    weight = weight.float()
    smooth_scale = smooth_scale.float()
    N = x.shape[-1]
    rmsnorm = torch.nn.RMSNorm(
        normalized_shape=N, eps=1e-6, dtype=torch.float32, device=x.device
    )
    with torch.no_grad():
        rmsnorm.weight.copy_(weight)
    y = rmsnorm(x)
    # smooth
    y_q, y_scale, y_maxs = torch_smooth_quant(
        y, smooth_scale, reverse=False, round_scale=round_scale
    )
    return y_q, y_scale, y_maxs


def torch_rms_and_block_quant_forward(x, weight, round_scale=False):
    x = x.float()
    weight = weight.float()
    N = x.shape[-1]
    rmsnorm = torch.nn.RMSNorm(
        normalized_shape=N, eps=1e-6, dtype=torch.float32, device=x.device
    )
    with torch.no_grad():
        rmsnorm.weight.copy_(weight)
    y = rmsnorm(x)
    # blockwise
    y_q, y_scale = torch_group_quant(y, round_scale=round_scale)
    yt_q, yt_scale = torch_group_quant(y.t(), round_scale=round_scale)
    return y_q, y_scale, yt_q, yt_scale


@torch.compile
def torch_group_rms_norm_gate_forward(
    x, gate, weight, eps=1e-6, group_size=4, transpose=True
):
    x = x.float()
    gate = gate.float()
    weight = weight.float()
    if transpose:
        length, bs, dim = gate.shape
    else:
        bs, length, dim = gate.shape
    d = dim // group_size
    attn_output = x.view(bs, length, group_size, d).transpose(0, 1)
    outputs = []
    for i in range(group_size):
        outputs.append(
            F.rms_norm(
                attn_output[:, :, i], [d], weight=weight[i * d : (i + 1) * d], eps=eps
            )
        )
    outputs = torch.stack(outputs, 2)
    if transpose:
        outputs = outputs.view(length, bs, dim)
    else:
        outputs = outputs.view(bs, length, dim)
    gate = F.sigmoid(gate)
    return outputs * gate


def torch_group_rms_norm_gate_backward(
    grad_output, x, gate, weight, eps=1e-6, group_size=4, transpose=True
):
    grad_output = grad_output.float()
    x = x.float().clone().detach().requires_grad_()
    gate = gate.float().clone().detach().requires_grad_()
    weight = weight.float().clone().detach().requires_grad_()
    y = torch_group_rms_norm_gate_forward(
        x, gate, weight, eps=eps, group_size=group_size, transpose=transpose
    )
    y.backward(gradient=grad_output)
    return x.grad, gate.grad, weight.grad

@pytest.mark.parametrize("M,N", [(16384, 2048), (8192, 4096), (4096, 8192)])
@pytest.mark.parametrize("bench", [True])
def test_rmsnorm(M, N, bench):
    dtype = torch.bfloat16
    device = "musa:0"

    x = torch.randn(M, N, dtype=dtype, requires_grad=True, device=device) ** 3
    weight = torch.randn(N, dtype=dtype, requires_grad=True, device=device)
    dy = torch.randn(M, N, dtype=dtype, device=device)

    y_ref = torch_rms_forward(x, weight)
    y = triton_rms_norm_forward(x, weight)
    assert output_check(y_ref.float(), y.float(), "y")

    dx_ref, dw_ref = torch_rms_backward(x, weight, dy)
    dx, dw = triton_rms_norm_backward(dy, x, weight)
    assert output_check(dx_ref, dx, mode="dx")
    assert output_check(dw_ref, dw, mode="dw")

    if bench:
        benchmark_func(torch_rms_forward, x, weight)
        benchmark_func(triton_rms_norm_forward, x, weight, ref_bytes=M * N * 3)
        benchmark_func(torch_rms_backward, x, weight, dy)
        benchmark_func(triton_rms_norm_backward, dy, x, weight, ref_bytes=M * N * 3)


@pytest.mark.parametrize("M,N", [(16384, 2048), (8192, 4096), (4096, 8192)])
@pytest.mark.parametrize("bench", [True])
def test_rmsnorm_and_smooth_quant(M, N, bench):
    dtype = torch.bfloat16
    device = "musa:0"

    x = torch.randn(M, N, dtype=dtype, requires_grad=True, device=device)
    weight = torch.randn(N, dtype=dtype, requires_grad=True, device=device)
    smooth_scale = torch.rand(N, dtype=torch.float32, requires_grad=False, device=device) + 0.1
    calibrate = True

    # smooth
    q_ref, scale_ref, maxs_ref = torch_rms_and_smooth_quant_forward(
        x, weight, smooth_scale=smooth_scale, round_scale=True
    )

    q, scale, maxs, rms = triton_rms_norm_and_smooth_quant_forward(
        x,
        weight,
        smooth_scale=smooth_scale,
        calibrate=calibrate,
        output_rms=True,
        round_scale=True,
    )
    assert output_check(q_ref, q, mode="smooth.data")
    assert output_check(scale_ref, scale, mode="smooth.scale")
    if calibrate:
        assert output_check(maxs_ref, maxs, mode="smooth.maxs")

    if bench:
        benchmark_func(
            triton_rms_norm_and_smooth_quant_forward,
            x,
            weight,
            smooth_scale=smooth_scale,
            calibrate=True,
            round_scale=True,
            output_rms=True,
            ref_bytes=M * N * 3,
        )


@pytest.mark.parametrize("M,N", [(128, 2048), (8192, 4096)])
@pytest.mark.parametrize("bench", [True])
def test_rmsnorm_and_block_quant(M, N, bench):
    dtype = torch.bfloat16
    device = "musa:0"

    x = torch.randn(M, N, dtype=dtype, requires_grad=True, device=device) ** 2
    weight = torch.randn(N, dtype=dtype, requires_grad=True, device=device)

    # blockwise
    q_ref, scale_ref, qt_ref, scale_t_ref = torch_rms_and_block_quant_forward(
        x, weight, round_scale=True
    )
    q, scale, rms, q_t, scale_t = triton_rms_norm_and_block_quant_forward(
        x, weight, round_scale=True, output_mode=2
    )
    assert output_check(q_ref, q, mode="2.block.data")
    assert output_check(scale_ref.t(), scale, mode="2.block.scale")
    assert output_check(qt_ref, q_t, mode="2.block.t_data")
    assert output_check(scale_t_ref.t(), scale_t, mode="2.block.t_scale")

    q, scale, _, _, _ = triton_rms_norm_and_block_quant_forward(
        x, weight, round_scale=True, output_mode=0
    )
    assert output_check(q_ref, q, mode="0.block.data")
    assert output_check(scale_ref.t(), scale, mode="0.block.scale")

    _, _, _, q_t, scale_t = triton_rms_norm_and_block_quant_forward(
        x, weight, round_scale=True, rms=rms, output_mode=1
    )
    assert output_check(qt_ref, q_t, mode="0.block.t_data")
    assert output_check(scale_t_ref.t(), scale_t, mode="0.block.t_scale")

    if bench:
        benchmark_func(
            triton_rms_norm_and_block_quant_forward,
            x,
            weight,
            round_scale=True,
            output_mode=0,
            ref_bytes=M * N * 3,
        )

        benchmark_func(
            triton_rms_norm_and_block_quant_forward,
            x,
            weight,
            round_scale=True,
            output_mode=1,
            ref_bytes=M * N * 3,
        )

        benchmark_func(
            triton_rms_norm_and_block_quant_forward,
            x,
            weight,
            round_scale=True,
            output_mode=2,
            ref_bytes=M * N * 4,
        )


@pytest.mark.parametrize("bs,length,dim,group_size", [(2, 4096, 2048,4), (1, 4096, 4096,4)])
@pytest.mark.parametrize("transpose", [True, False])
@pytest.mark.parametrize("bench", [False])
def test_group_rms_norm_gate(bs, length, dim, group_size, transpose, bench):
    dtype = torch.bfloat16
    device = "musa:0"
    x = (
        torch.randn(bs, length, dim, dtype=dtype, requires_grad=True, device=device)
        ** 2
    )
    weight = torch.randn(dim, dtype=dtype, requires_grad=True, device=device)
    if transpose:
        gate = torch.randn(
            length, bs, dim, dtype=dtype, requires_grad=True, device=device
        )
        grad_output = torch.randn(
            length, bs, dim, dtype=dtype, requires_grad=True, device=device
        )
    else:
        gate = torch.randn(
            bs, length, dim, dtype=dtype, requires_grad=True, device=device
        )
        grad_output = torch.randn(
            bs, length, dim, dtype=dtype, requires_grad=True, device=device
        )

    output_ref = torch_group_rms_norm_gate_forward(
        x, gate, weight, group_size=group_size, transpose=transpose
    )
    output = triton_group_rms_norm_gate_forward(
        x, gate, weight, group_size=group_size, transpose=transpose
    )
    assert output_check(output_ref, output.float(), mode="group_norm_gate.y")

    dx_ref, dg_ref, dw_ref = torch_group_rms_norm_gate_backward(
        grad_output, x, gate, weight, group_size=group_size, transpose=transpose
    )
    dx, dg, dw = triton_group_rms_norm_gate_backward(
        grad_output, x, gate, weight, group_size=group_size, transpose=transpose
    )
    assert output_check(dx_ref, dx.float(), mode="group_norm_gate.dx")
    assert output_check(dg_ref, dg.float(), mode="group_norm_gate.dg")
    assert output_check(dw_ref, dw.float(), mode="group_norm_gate.dw")

    if bench:
        benchmark_func(
            torch_group_rms_norm_gate_forward,
            x,
            gate,
            weight,
            group_size=group_size,
            ref_bytes=bs * length * dim * 6,
        )

        benchmark_func(
            triton_group_rms_norm_gate_forward,
            x,
            gate,
            weight,
            group_size=group_size,
            ref_bytes=bs * length * dim * 6,
        )

        benchmark_func(
            triton_group_rms_norm_gate_backward,
            grad_output,
            x,
            gate,
            weight,
            group_size=group_size,
            ref_bytes=bs * length * dim * 10,
        )
