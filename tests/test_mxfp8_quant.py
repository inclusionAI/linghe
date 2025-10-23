# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.quant.mxfp8 import triton_mxfp8_quant

from linghe.tools.benchmark import benchmark_func
from linghe.tools.util import output_check, torch_mxfp8_quant




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
        benchmark_func(triton_mxfp8_quant, x, ref_bytes=ref_bytes)



if __name__ == '__main__':
    test_mxfp8_quant(M=4096, N=8192, bench=True)
    # test_mxfp8_quant(M=4031, N=8192, bench=False)
    # test_mxfp8_quant(M=4031, N=512, bench=False)




