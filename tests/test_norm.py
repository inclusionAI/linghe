# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import torch.nn.functional as F

from linghe.tools.benchmark import benchmark_func
from linghe.tools.util import (output_check,
                              torch_group_quant)
from linghe.utils.norm import (triton_rms_norm_and_block_quant_forward,
                               triton_rms_norm_and_mxfp8_quant_forward,
                               triton_rms_norm_backward,
                               triton_rms_norm_forward,
                               triton_group_norm_gate_forward,
                               triton_group_norm_gate_backward)


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


def torch_rms_and_quant_forward(x, weight, smooth_scale=None,
                                round_scale=False):
    x = x.float()
    weight = weight.float()
    if smooth_scale is not None:
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
    # blockwise
    y_q, y_scale = torch_group_quant(y, round_scale=round_scale)
    yt_q, yt_scale = torch_group_quant(y.t(), round_scale=round_scale)
    return y_q, y_scale, yt_q, yt_scale

def torch_rms_and_mxfp8_quant_forward(x, weight, smooth_scale=None,
                                                round_scale=False):
    x = x.float()
    weight = weight.float()
    if smooth_scale is not None:
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
    # mxfp8
    y_q, y_scale = torch_group_quant(y, B=32, round_scale=round_scale)
    yt_q, yt_scale = torch_group_quant(y.t(), B=32, round_scale=round_scale)
    return y_q, y_scale, yt_q, yt_scale


# backward of rms is bf16, do not need quant
def torch_rms_and_quant_backward(x, weight, dy, smooth_scale=None,
                                 round_scale=False):
    x = x.float()
    weight = weight.float()
    dy = dy.float()
    if smooth_scale is not None:
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
    x = x.clone().detach().requires_grad_()
    y = rmsnorm(x)
    y.backward(gradient=dy)
    dx = x.grad
    dw = rmsnorm.weight.grad
    dx_q, dx_scale = torch_group_quant(dx, round_scale=round_scale)
    dxt_q, dxt_scale = torch_group_quant(dx.t(), round_scale=round_scale)

    return dx_q, dx_scale, dw, dxt_q, dxt_scale


@torch.compile
def torch_group_norm_gate_forward(x, gate, weight, eps=1e-6, group_size=4):
    x = x.float()
    gate = gate.float()
    weight = weight.float()
    length, bs, dim = gate.shape
    d = dim // group_size
    attn_output = x.view(bs, length, group_size, d).transpose(0, 1)
    outputs = []
    for i in range(group_size):
        outputs.append(F.rms_norm(attn_output[:, :, i], [d],
                                  weight=weight[i * d:(i + 1) * d], eps=eps))
    outputs = torch.stack(outputs, 2)
    outputs = outputs.view(length, bs, dim)
    gate = F.sigmoid(gate)
    return outputs * gate


def torch_group_norm_gate_backward(grad_output, x, gate, weight, eps=1e-6,
                                   group_size=4):
    grad_output = grad_output.float()
    x = x.float().clone().detach().requires_grad_()
    gate = gate.float().clone().detach().requires_grad_()
    weight = weight.float().clone().detach().requires_grad_()
    y = torch_group_norm_gate_forward(x, gate, weight, eps=eps,
                                      group_size=group_size)
    y.backward(gradient=grad_output)
    return x.grad, gate.grad, weight.grad


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


def test_rmsnorm_and_block_quant(M=4096, N=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.randn(M, N, dtype=dtype, requires_grad=True, device=device) ** 2
    weight = torch.randn(N, dtype=dtype, requires_grad=True, device=device)

    # blockwise
    q_ref, scale_ref, qt_ref, scale_t_ref = torch_rms_and_quant_forward(x,
                                                                        weight,
                                                                        smooth_scale=None,
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

    # blockwise
    q_ref, scale_ref, qt_ref, scale_t_ref = torch_rms_and_mxfp8_quant_forward(x,
                                                                        weight,
                                                                        smooth_scale=None,
                                                                        round_scale=True)
    q, scale, rms, q_t, scale_t = triton_rms_norm_and_mxfp8_quant_forward(x,
                                                                          weight,
                                                                          round_scale=True,
                                                                          output_mode=2)
    output_check(q_ref, q, mode="2.block.data")
    output_check(scale_ref.t(), scale, mode='2.block.scale')
    output_check(qt_ref.t(), q_t, mode='2.block.t_data')
    output_check(scale_t_ref.t(), scale_t, mode="2.block.t_scale")

    q, scale, _, _, _ = triton_rms_norm_and_mxfp8_quant_forward(x, weight,
                                                                round_scale=True,
                                                                output_mode=0)
    output_check(q_ref, q, mode="0.block.data")
    output_check(scale_ref.t(), scale, mode='0.block.scale')

    _, _, _, q_t, scale_t = triton_rms_norm_and_mxfp8_quant_forward(x, weight,
                                                                    round_scale=True,
                                                                    rms=rms,
                                                                    output_mode=1)
    output_check(qt_ref.t(), q_t, mode='0.block.t_data')
    output_check(scale_t_ref.t(), scale_t, mode="0.block.t_scale")

    if bench:
        benchmark_func(triton_rms_norm_and_mxfp8_quant_forward, x, weight,
                       round_scale=True,
                       output_mode=0,
                       ref_bytes=M * N * 3)

        benchmark_func(triton_rms_norm_and_mxfp8_quant_forward, x, weight,
                       round_scale=True,
                       output_mode=1,
                       ref_bytes=M * N * 3)

        benchmark_func(triton_rms_norm_and_mxfp8_quant_forward, x, weight,
                       round_scale=True,
                       output_mode=2,
                       ref_bytes=M * N * 4)


def test_group_norm_gate_quant(bs=1, length=4096, dim=4096, group_size=4,
                               bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'
    x = torch.randn(bs, length, dim, dtype=dtype, requires_grad=True,
                    device=device) ** 2
    weight = torch.randn(dim, dtype=dtype, requires_grad=True, device=device)
    gate = torch.randn(length, bs, dim, dtype=dtype, requires_grad=True,
                       device=device)
    grad_output = torch.randn(length, bs, dim, dtype=dtype, requires_grad=True,
                              device=device)

    output_ref = torch_group_norm_gate_forward(x, gate, weight,
                                               group_size=group_size)
    output = triton_group_norm_gate_forward(x, gate, weight,
                                            group_size=group_size)
    output_check(output_ref, output.float(), mode='group_norm_gate.y')

    dx_ref, dg_ref, dw_ref = torch_group_norm_gate_backward(grad_output, x,
                                                            gate, weight,
                                                            group_size=group_size)
    dx, dg, dw = triton_group_norm_gate_backward(grad_output, x, gate, weight,
                                                 group_size=group_size)
    output_check(dx_ref, dx.float(), mode='group_norm_gate.dx')
    output_check(dg_ref, dg.float(), mode='group_norm_gate.dg')
    output_check(dw_ref, dw.float(), mode='group_norm_gate.dw')

    if bench:
        benchmark_func(torch_group_norm_gate_forward, x, gate, weight,
                       group_size=group_size,
                       ref_bytes=bs * length * dim * 6)

        benchmark_func(triton_group_norm_gate_forward, x, gate, weight,
                       group_size=group_size,
                       ref_bytes=bs * length * dim * 6)

        benchmark_func(triton_group_norm_gate_backward, grad_output, x, gate,
                       weight, group_size=group_size,
                       ref_bytes=bs * length * dim * 10)


if __name__ == '__main__':
    # test_rmsnorm(M=16384, N=2048, bench=False)
    # test_rmsnorm(M=8192, N=4096, bench=False)
    # test_rmsnorm(M=4096, N=8192, bench=False)
    # test_rmsnorm_and_block_quant(M=128, N=2048, bench=False)
    # test_rmsnorm_and_block_quant(M=8192, N=4096, bench=False)
    test_rmsnorm_and_mxfp8_quant(M=2048, N=1664, bench=False)
    # test_group_norm_gate_quant(bs=2, length=4096, dim=2048, group_size=4,
    #                            bench=True)
    # test_group_norm_gate_quant(bs=1, length=4096, dim=4096, group_size=4,
    #                            bench=True)
