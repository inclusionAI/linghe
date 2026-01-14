# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import torch.nn.functional as F

from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.utils.gate import (triton_group_rms_norm_gate_forward,
                               triton_group_rms_norm_gate_backward)


# @torch.compile
def torch_group_rms_norm_gate_forward(x, gate, weight, eps=1e-6, group_size=4,
                                      transpose=True):
    dtype = x.dtype
    x = x.float()
    gate = gate.float()
    weight = weight.float()
    if transpose:
        length, bs, dim = gate.shape
    else:
        bs, length, dim = gate.shape
    d = dim // group_size
    attn_output = x.view(bs, length, group_size, d)
    outputs = []
    for i in range(group_size):
        if weight.size(0) == dim:
            o = F.rms_norm(attn_output[:, :, i], [d],
                           weight=weight[i * d:(i + 1) * d], eps=eps)
        else:
            o = F.rms_norm(attn_output[:, :, i], [d],
                           weight=weight, eps=eps)
        outputs.append(o)
    outputs = torch.stack(outputs, 2).view(bs, length, dim)
    if transpose:
        outputs = outputs.transpose(0, 1)
    gate = F.sigmoid(gate)
    outputs = (outputs * gate).to(dtype)
    return outputs


def torch_group_rms_norm_gate_backward(grad_output, x, gate, weight, eps=1e-6,
                                       group_size=4,
                                       transpose=True):
    dtype = grad_output.dtype
    grad_output = grad_output.float()
    x = x.float().clone().detach().requires_grad_()
    gate = gate.float().clone().detach().requires_grad_()
    weight = weight.float().clone().detach().requires_grad_()
    y = torch_group_rms_norm_gate_forward(x, gate, weight, eps=eps,
                                          group_size=group_size,
                                          transpose=transpose)
    y.backward(gradient=grad_output)
    return x.grad.to(dtype), gate.grad.to(dtype), weight.grad.to(dtype)


def test_group_rms_norm_gate(bs=1, length=4096, dim=4096, group_size=4,
                             transpose=True, share=False, coef=1.0,
                             grad_coef=1.0,
                             bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'
    x = torch.randn(bs, length, dim, dtype=dtype, requires_grad=True,
                    device=device)
    weight = torch.randn(dim // group_size if share else dim, dtype=dtype,
                         requires_grad=True, device=device)
    if transpose:
        gate = (torch.randn(length, bs, dim, dtype=dtype,
                            device=device) * coef).requires_grad_()
        grad_output = torch.randn(length, bs, dim, dtype=dtype,
                                  device=device) * grad_coef
    else:
        gate = (torch.randn(bs, length, dim, dtype=dtype,
                            device=device) * coef).requires_grad_()
        grad_output = torch.randn(bs, length, dim, dtype=dtype,
                                  device=device) * grad_coef

    output_ref = torch_group_rms_norm_gate_forward(x, gate, weight,
                                                   group_size=group_size,
                                                   transpose=transpose)
    output = triton_group_rms_norm_gate_forward(x, gate, weight,
                                                group_size=group_size,
                                                transpose=transpose)
    output_check(output_ref, output, name='group_norm_gate.y')

    dx_ref, dg_ref, dw_ref = torch_group_rms_norm_gate_backward(grad_output, x,
                                                                gate, weight,
                                                                group_size=group_size,
                                                                transpose=transpose)
    dx, dg, dw = triton_group_rms_norm_gate_backward(grad_output, x, gate,
                                                     weight,
                                                     group_size=group_size,
                                                     transpose=transpose)
    output_check(dx_ref, dx, name='group_norm_gate.dx')
    output_check(dg_ref, dg, name='group_norm_gate.dg')
    output_check(dw_ref, dw.to(dtype), name='group_norm_gate.dw')

    if bench:
        benchmark_func(torch_group_rms_norm_gate_forward, x, gate, weight,
                       group_size=group_size, transpose=transpose,
                       ref_bytes=bs * length * dim * 6)

        benchmark_func(triton_group_rms_norm_gate_forward, x, gate, weight,
                       group_size=group_size, transpose=transpose,
                       ref_bytes=bs * length * dim * 6)

        benchmark_func(triton_group_rms_norm_gate_backward, grad_output, x,
                       gate,
                       weight, group_size=group_size, transpose=transpose,
                       ref_bytes=bs * length * dim * 10)


if __name__ == '__main__':
    test_group_rms_norm_gate(bs=2, length=4096, dim=2048, group_size=4,
                             transpose=True,
                             bench=False)
    test_group_rms_norm_gate(bs=2, length=4096, dim=2048, group_size=4,
                             transpose=False,
                             bench=False)
    test_group_rms_norm_gate(bs=2, length=4096, dim=2048, group_size=4,
                             transpose=False, share=True,
                             bench=False)
    test_group_rms_norm_gate(bs=1, length=4096, dim=4096, group_size=4,
                             bench=False)
    test_group_rms_norm_gate(bs=2, length=4096, dim=1536, group_size=4,
                             transpose=True,
                             bench=False)
    test_group_rms_norm_gate(bs=2, length=4096, dim=1536, group_size=4,
                             transpose=False,
                             coef=10000.0,
                             grad_coef=10000.0,
                             bench=False)
    test_group_rms_norm_gate(bs=2, length=4096, dim=1536, group_size=4,
                             transpose=False,
                             coef=0.0,
                             grad_coef=0.0,
                             bench=False)
