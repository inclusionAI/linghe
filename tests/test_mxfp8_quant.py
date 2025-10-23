# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.quant.mxfp8 import triton_mxfp8_quant

from linghe.tools.benchmark import benchmark_func
from linghe.tools.util import output_check



def torch_mxfp8_quant(x):
    m, N = x.shape 
    if m % 128 != 0:
        M = (m + 127) // 128 * 128
        x = torch.cat([x, torch.zeros((M-m,N), dtype=x.dtype, device=x.device)], 0)
    else:
        M = m
    xs = x.view(M, N//32, 32)
    xm = xs.abs().amax(2)
    scale = torch.maximum(xm/448, 1e-30*torch.ones_like(xm)) 
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    x_q = (xs/scale[:,:,None]).to(torch.float8_e4m3fn).view(M,N)
    x_scale = scale.to(torch.float8_e8m0fnu).view(torch.uint8)

    xs = x.view(M//32, 32, N)
    xm = xs.abs().amax(1)
    scale = torch.maximum(xm/448, 1e-30*torch.ones_like(xm)) 
    scale = torch.exp2(torch.ceil(torch.log2(scale)))
    xt_q = (xs/scale[:,None,:]).to(torch.float8_e4m3fn).view(M,N)
    xt_scale = scale.to(torch.float8_e8m0fnu).view(torch.uint8)

    return x_q, x_scale, xt_q, xt_scale

def test_mxfp8_quant(M=4096, N=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.randn(M, N, dtype=dtype, device=device)

    x_q_ref, x_scale_ref, xt_q_ref, xt_scale_ref = torch_mxfp8_quant(x)
    x_q, x_scale, xt_q, xt_scale = triton_mxfp8_quant(x)


    output_check(x_q_ref, x_q, 'x_q')
    output_check(x_scale_ref, x_scale, 'x_scale')
    output_check(xt_q_ref, xt_q, 'xt_q')
    output_check(xt_scale_ref, xt_scale, 'xt_scale')


    if bench:
        ref_bytes = M * N * 4
        benchmark_func(triton_mxfp8_quant, x, ref_flops=ref_bytes)



if __name__ == '__main__':
    test_mxfp8_quant(M=4096, N=8192, bench=True)


