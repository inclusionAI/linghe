# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import torch.nn.functional as F

from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.infer.gate import triton_group_rms_norm_gate, triton_block_group_rms_norm_gate
from linghe.tools.util import torch_group_quant


def torch_group_rms_norm_gate(x, gate, weight, eps=1e-6, group_size=4):
    dtype = x.dtype
    x = x.float()
    gate = gate.float()
    weight = weight.float()
    length, dim = gate.shape
    d = dim // group_size
    attn_output = x.view(length, group_size, d)
    outputs = []
    for i in range(group_size):
        if weight.size(0) == dim:
            o = F.rms_norm(attn_output[:, i], [d],
                           weight=weight[i * d:(i + 1) * d], eps=eps)
        else:
            o = F.rms_norm(attn_output[:, i], [d],
                           weight=weight, eps=eps)
        outputs.append(o)
    outputs = torch.stack(outputs, 1).view(length, dim)
    gate = F.sigmoid(gate)
    outputs = (outputs * gate).to(dtype)
    return outputs

def torch_block_group_rms_norm_gate(x, gate, weight, eps=1e-6, group_size=4):
    outputs = torch_group_rms_norm_gate(x, gate, weight, eps=eps, group_size=group_size)
    y_q, y_scale = torch_group_quant(outputs, round_scale=False)
    return y_q, y_scale

def test_group_rms_norm_gate(length=4096, dim=4096, group_size=4,
                             transpose=True, share=False, coef=1.0,
                             grad_coef=1.0,
                             bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'
    x = torch.randn(length, dim, dtype=dtype, requires_grad=True,
                    device=device)
    weight = torch.randn(dim // group_size if share else dim, dtype=dtype,
                         requires_grad=True, device=device)

    gate = (torch.randn(length, dim, dtype=dtype,
                        device=device) * coef).requires_grad_()

    output_ref = torch_group_rms_norm_gate(x, gate, weight,
                                                   group_size=group_size)
    output = triton_group_rms_norm_gate(x, gate, weight,
                                                group_size=group_size)
    output_check(output_ref, output, name='group_norm_gate.y')

    if bench:
        benchmark_func(torch_group_rms_norm_gate, x, gate, weight,
                       group_size=group_size, 
                       ref_bytes=length * dim * 6)

        benchmark_func(triton_group_rms_norm_gate, x, gate, weight,
                       group_size=group_size, 
                       ref_bytes=length * dim * 6,
                       n_profile=10)



def test_block_group_rms_norm_gate(length=4096, dim=4096, group_size=4,
                             transpose=True, share=False, coef=1.0,
                             grad_coef=1.0,
                             bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'
    x = torch.randn(length, dim, dtype=dtype, requires_grad=True,
                    device=device)
    weight = torch.randn(dim // group_size if share else dim, dtype=dtype,
                         requires_grad=True, device=device)

    gate = (torch.randn(length, dim, dtype=dtype,
                        device=device) * coef).requires_grad_()

    output_ref, scale_ref = torch_block_group_rms_norm_gate(x, gate, weight,
                                                   group_size=group_size)
    output, scale = triton_block_group_rms_norm_gate(x, gate, weight,
                                                group_size=group_size)
    output_check(output_ref, output, name='group_norm_gate.data')
    output_check(scale_ref, scale, name='group_norm_gate.scale', atol=1e-4, rtol=3e-2)

    if bench:
        benchmark_func(torch_block_group_rms_norm_gate, x, gate, weight,
                       group_size=group_size, 
                       ref_bytes=length * dim * 6)

        benchmark_func(triton_block_group_rms_norm_gate, x, gate, weight,
                       group_size=group_size, 
                       ref_bytes=length * dim * 6)


if __name__ == '__main__':
    test_group_rms_norm_gate(length=4096, dim=2048, group_size=4,
                             bench=False)
    test_group_rms_norm_gate(length=3432, dim=2048, group_size=4,
                             bench=False)
    test_group_rms_norm_gate(length=4, dim=1024, group_size=1,
                             bench=True)
    test_block_group_rms_norm_gate(length=4096, dim=2048, group_size=4,
                             bench=False)
    test_block_group_rms_norm_gate(length=3431, dim=2048, group_size=1,
                             bench=False)
