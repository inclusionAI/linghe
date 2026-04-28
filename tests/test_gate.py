# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import pytest
import torch
import torch.nn.functional as F

from linghe.tools.check import output_check
from linghe.utils.gate import (
    triton_group_rms_norm_gate_forward,
    triton_group_rms_norm_gate_backward,
    triton_group_rms_norm_gate_and_mxfp8_quant_forward,
    triton_group_rms_norm_gate_and_mxfp8_quant_backward,
)
from linghe.tools.util import torch_mxfp8_quant


# @torch.compile
def torch_group_rms_norm_gate_forward(
    x, gate, weight, eps=1e-6, group_size=4, native=True, high_precison=False
):
    dtype = x.dtype
    if not native:
        x = torch.permute(x, [1, 0, 2])
    x = x.float()
    gate = gate.float()
    weight = weight.float()
    length, bs, dim = gate.shape
    d = dim // group_size
    attn_output = x.view(bs, length, group_size, d)
    outputs = []
    for i in range(group_size):
        if weight.size(0) == dim:
            o = F.rms_norm(
                attn_output[:, :, i], [d], weight=weight[i * d : (i + 1) * d], eps=eps
            )
        else:
            o = F.rms_norm(attn_output[:, :, i], [d], weight=weight, eps=eps)
        outputs.append(o)
    outputs = torch.stack(outputs, 2).view(bs, length, dim)
    outputs = outputs.transpose(0, 1)
    gate = F.sigmoid(gate)
    if high_precison:
        outputs = outputs * gate
    else:
        outputs = (outputs * gate).to(dtype)
    return outputs


def torch_group_rms_norm_gate_mxfp8_quant_forward(
    x, gate, weight, eps=1e-6, group_size=4, native=True
):
    out = torch_group_rms_norm_gate_forward(
        x, gate, weight, eps, group_size, native, high_precison=True
    )
    out = out.reshape(-1, x.size(-1))
    x_q_ref, x_scale_ref, xt_q_ref, xt_scale_ref = torch_mxfp8_quant(out)
    return x_q_ref, x_scale_ref, xt_q_ref, xt_scale_ref


def torch_group_rms_norm_gate_mxfp8_quant_backward(
    grad_output, x, gate, weight, eps=1e-6, group_size=4, native=True
):
    dtype = grad_output.dtype
    grad_output = grad_output.float()
    x = x.float().clone().detach().requires_grad_()
    gate = gate.float().clone().detach().requires_grad_()
    weight = weight.float().clone().detach().requires_grad_()
    y = torch_group_rms_norm_gate_forward(
        x,
        gate,
        weight,
        eps=eps,
        group_size=group_size,
        native=native,
        high_precison=True,
    )
    y.backward(gradient=grad_output)

    dx = x.grad.to(dtype)
    dg = gate.grad
    dw = weight.grad.to(dtype)
    dg = dg.reshape(-1, gate.size(-1))
    g_q, g_scale, gt_q, gt_scale = torch_mxfp8_quant(dg)

    return dx, g_q, g_scale, gt_q, gt_scale, dw


def torch_group_rms_norm_gate_backward(
    grad_output, x, gate, weight, eps=1e-6, group_size=4, native=True
):
    dtype = grad_output.dtype
    grad_output = grad_output.float()
    x = x.float().clone().detach().requires_grad_()
    gate = gate.float().clone().detach().requires_grad_()
    weight = weight.float().clone().detach().requires_grad_()
    y = torch_group_rms_norm_gate_forward(
        x, gate, weight, eps=eps, group_size=group_size
    )
    y.backward(gradient=grad_output)
    return x.grad.to(dtype), gate.grad.to(dtype), weight.grad.to(dtype)


@pytest.mark.parametrize(
    "bs,length,dim,group_size,contiguous,share,coef,grad_coef,native",
    [
        (2, 4096, 2048, 4, True, False, 1.0, 1.0, True),
        (2, 4096, 2048, 4, True, True, 1.0, 1.0, True),
        (1, 4096, 4096, 4, True, False, 1.0, 1.0, True),
        (2, 4096, 1536, 4, True, False, 1.0, 1.0, True),
        (2, 4096, 1536, 4, True, False, 10000.0, 10000.0, True),
        (2, 4096, 1536, 4, True, False, 0.0, 0.0, True),
        (2, 4096, 1536, 4, False, False, 1.0, 1.0, True),
        (2, 4096, 1536, 4, False, False, 1.0, 1.0, False),
    ],
)
def test_group_rms_norm_gate(
    bs, length, dim, group_size, contiguous, share, coef, grad_coef, native, benchmark
):
    dtype = torch.bfloat16
    device = "cuda:0"
    if native:
        x = torch.randn(bs, length, dim, dtype=dtype, requires_grad=True, device=device)
    else:
        x = torch.randn(length, bs, dim, dtype=dtype, requires_grad=True, device=device)
    weight = torch.randn(
        dim // group_size if share else dim,
        dtype=dtype,
        requires_grad=True,
        device=device,
    )
    if contiguous:
        gate = (
            torch.randn(length, bs, dim, dtype=dtype, device=device) * coef
        ).requires_grad_()
    else:
        tmp = torch.randn(length, bs, 3 * dim, dtype=dtype, device=device) * coef
        split_sizes = [dim, dim, dim]
        _, _, gate = torch.split(tmp, split_sizes, dim=-1)
        gate = gate.requires_grad_()

    grad_output = torch.randn(length, bs, dim, dtype=dtype, device=device) * grad_coef

    output_ref = torch_group_rms_norm_gate_forward(
        x, gate, weight, group_size=group_size, native=native
    )
    output_ref.backward(gradient=grad_output)
    dx_ref = x.grad.to(dtype)
    dg_ref = gate.grad.to(dtype)
    dw_ref = weight.grad.to(dtype)

    output = triton_group_rms_norm_gate_forward(x, gate, weight, group_size=group_size)
    output_check(output_ref, output, name="group_norm_gate.y")

    dx, dg, dw = triton_group_rms_norm_gate_backward(
        grad_output, x, gate, weight, group_size=group_size
    )
    output_check(dx_ref, dx, name="group_norm_gate.dx")
    output_check(dg_ref, dg, name="group_norm_gate.dg")
    output_check(dw_ref, dw.to(dtype), name="group_norm_gate.dw")

    benchmark(
        torch_group_rms_norm_gate_forward,
        x,
        gate,
        weight,
        group_size=group_size,
        ref_bytes=bs * length * dim * 6,
    )

    benchmark(
        triton_group_rms_norm_gate_forward,
        x,
        gate,
        weight,
        group_size=group_size,
        ref_bytes=bs * length * dim * 6,
    )

    benchmark(
        triton_group_rms_norm_gate_backward,
        grad_output,
        x,
        gate,
        weight,
        group_size=group_size,
        ref_bytes=bs * length * dim * 10,
    )


@pytest.mark.parametrize(
    "bs,length,dim,group_size,contiguous,share,coef,grad_coef,native",
    [
        (2, 4096, 2048, 4, False, False, 1.0, 1.0, False),
        (2, 4096, 2048, 4, False, False, 1.0, 1.0, True),
        (2, 4096, 2048, 4, True, False, 1.0, 1.0, False),
        (2, 4096, 2048, 4, False, False, 1.0, 1.0, False),
        (2, 4096, 2048, 4, True, False, 1.0, 1.0, True),
    ],
)
def test_group_rms_norm_gate_quant(
    bs, length, dim, group_size, contiguous, share, coef, grad_coef, native, benchmark
):

    dtype = torch.bfloat16
    device = "cuda:0"
    if native:
        x = torch.randn(bs, length, dim, dtype=dtype, requires_grad=True, device=device)
    else:
        x = torch.randn(length, bs, dim, dtype=dtype, requires_grad=True, device=device)
    weight = torch.randn(
        dim // group_size if share else dim,
        dtype=dtype,
        requires_grad=True,
        device=device,
    )
    if contiguous:
        gate = (
            torch.randn(length, bs, dim, dtype=dtype, device=device) * coef
        ).requires_grad_()
    else:
        tmp = torch.randn(length, bs, 3 * dim, dtype=dtype, device=device) * coef
        split_sizes = [dim, dim, dim]
        _, _, gate = torch.split(tmp, split_sizes, dim=-1)
        gate = gate.requires_grad_()

    grad_output = torch.randn(length, bs, dim, dtype=dtype, device=device) * grad_coef

    x_q_ref, x_scale_ref, xt_q_ref, xt_scale_ref = (
        torch_group_rms_norm_gate_mxfp8_quant_forward(
            x, gate, weight, group_size=group_size, native=native
        )
    )
    x_q, x_scale, xt_q, xt_scale = triton_group_rms_norm_gate_and_mxfp8_quant_forward(
        x, gate, weight, group_size=group_size
    )

    output_check(x_q_ref, x_q, "x_q")
    output_check(x_scale_ref, x_scale, "x_scale")
    output_check(xt_q_ref, xt_q, "xt_q")
    output_check(xt_scale_ref, xt_scale, "xt_scale")

    dx_ref, g_q_ref, g_scale_ref, gt_q_ref, gt_scale_ref, dw_ref = (
        torch_group_rms_norm_gate_mxfp8_quant_backward(
            grad_output, x, gate, weight, group_size=group_size, native=native
        )
    )

    dx, g_q, g_scale, gt_q, gt_scale, dw = (
        triton_group_rms_norm_gate_and_mxfp8_quant_backward(
            grad_output, x, gate, weight, group_size=group_size
        )
    )

    output_check(dx_ref, dx, "dx")
    output_check(dw_ref, dw, "dw")
    output_check(g_q_ref, g_q, "gq")
    output_check(g_scale_ref, g_scale, "x_scale")
    output_check(gt_q_ref, gt_q, "xt_q")
    output_check(gt_scale_ref, gt_scale, "xt_scale")

    benchmark(
        triton_group_rms_norm_gate_and_mxfp8_quant_forward,
        x,
        gate,
        weight,
        1e-6,
        group_size,
        n_repeat=100,
    )

    benchmark(
        triton_group_rms_norm_gate_and_mxfp8_quant_backward,
        grad_output,
        x,
        gate,
        weight,
        1e-6,
        group_size,
        n_repeat=100,
    )
