# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import torch.nn.functional as F

from linghe.utils.gate import (triton_group_rms_norm_gate_forward,
                               triton_group_rms_norm_gate_backward)
from linghe.tools.util import output_check
from linghe.tools.benchmark import benchmark_func

# @torch.compile
def torch_group_rms_norm_gate_forward(x, gate, weight, eps=1e-6, group_size=4, transpose=True):
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
        if weight.size(0) == dim:
            o = F.rms_norm(attn_output[:, :, i], [d],
                                    weight=weight[i * d:(i + 1) * d], eps=eps)
        else:
            o = F.rms_norm(attn_output[:, :, i], [d],
                                    weight=weight, eps=eps)
        outputs.append(o)
    outputs = torch.stack(outputs, 2)
    if transpose:
        outputs = outputs.view(length, bs, dim)
    else:
        outputs = outputs.view(bs, length, dim)
    gate = F.sigmoid(gate)
    return outputs * gate


def torch_group_rms_norm_gate_backward(grad_output, x, gate, weight, eps=1e-6,
                                   group_size=4,
                                   transpose=True):
    grad_output = grad_output.float()
    x = x.float().clone().detach().requires_grad_()
    gate = gate.float().clone().detach().requires_grad_()
    weight = weight.float().clone().detach().requires_grad_()
    y = torch_group_rms_norm_gate_forward(x, gate, weight, eps=eps,
                                      group_size=group_size,
                                      transpose=transpose)
    y.backward(gradient=grad_output)
    return x.grad, gate.grad, weight.grad


def test_group_rms_norm_gate(bs=1, length=4096, dim=4096, group_size=4,
                               transpose=True, share=False,
                               bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'
    x = torch.randn(bs, length, dim, dtype=dtype, requires_grad=True,
                    device=device) ** 2
    weight = torch.randn(dim//group_size if share else dim, dtype=dtype, 
                         requires_grad=True, device=device)
    if transpose:
        gate = torch.randn(length, bs, dim, dtype=dtype, requires_grad=True,
                        device=device)
        grad_output = torch.randn(length, bs, dim, dtype=dtype, requires_grad=True,
                                device=device)
    else:
        gate = torch.randn(bs, length, dim, dtype=dtype, requires_grad=True,
                        device=device)
        grad_output = torch.randn(bs, length, dim, dtype=dtype, requires_grad=True,
                                device=device)

    output_ref = torch_group_rms_norm_gate_forward(x, gate, weight,
                                               group_size=group_size,
                                               transpose=transpose)
    output = triton_group_rms_norm_gate_forward(x, gate, weight,
                                            group_size=group_size,
                                               transpose=transpose)
    output_check(output_ref, output.float(), mode='group_norm_gate.y')

    dx_ref, dg_ref, dw_ref = torch_group_rms_norm_gate_backward(grad_output, x,
                                                            gate, weight,
                                                            group_size=group_size,
                                               transpose=transpose)
    dx, dg, dw = triton_group_rms_norm_gate_backward(grad_output, x, gate, weight,
                                                 group_size=group_size,
                                               transpose=transpose)
    output_check(dx_ref, dx.float(), mode='group_norm_gate.dx')
    output_check(dg_ref, dg.float(), mode='group_norm_gate.dg')
    output_check(dw_ref, dw.float(), mode='group_norm_gate.dw')

    if bench:
        benchmark_func(torch_group_rms_norm_gate_forward, x, gate, weight,
                       group_size=group_size,
                       ref_bytes=bs * length * dim * 6)

        benchmark_func(triton_group_rms_norm_gate_forward, x, gate, weight,
                       group_size=group_size,
                       ref_bytes=bs * length * dim * 6)

        benchmark_func(triton_group_rms_norm_gate_backward, grad_output, x, gate,
                       weight, group_size=group_size,
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


