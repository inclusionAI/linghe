# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import random

import torch

from linghe.quant.mxfp8 import triton_mxfp8_quant, triton_batch_mxfp8_quant
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.tools.util import torch_mxfp8_quant, torch_batch_mxfp8_quant


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
        benchmark_func(torch_mxfp8_quant, x)


def test_batch_mxfp8_quant(M=4096, N=4096, n_experts=32, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    splits = [max(random.randint(M - 256, M + 256), 0) for x in
              range(n_experts)]
    splits = [(x + 32) // 32 * 32 for x in splits]
    token_count_per_expert = torch.tensor(splits, device=device)

    x = torch.randn((sum(splits), N), dtype=dtype, device=device)

    x_q_ref, x_scale_ref, xt_q_ref, xt_scale_ref = torch_batch_mxfp8_quant(x,
                                                                           splits)

    x_q, x_scale, xt_q, xt_scale = triton_batch_mxfp8_quant(x,
                                                            token_count_per_expert,
                                                            splits,
                                                            output_mode=2)

    output_check(x_q_ref, x_q, 'x_q')
    output_check(x_scale_ref, x_scale, 'x_scale')
    output_check(xt_q_ref, xt_q, 'xt_q')
    output_check(xt_scale_ref, xt_scale, 'xt_scale')

    if bench:
        ref_bytes = M * N * n_experts * 4
        benchmark_func(torch_batch_mxfp8_quant, x, splits)
        benchmark_func(triton_batch_mxfp8_quant, x, token_count_per_expert,
                       splits, output_mode=2, ref_bytes=ref_bytes)


if __name__ == '__main__':
    test_mxfp8_quant(M=4096, N=8192, bench=False)
    test_mxfp8_quant(M=4031, N=8192, bench=False)
    test_mxfp8_quant(M=4096, N=8192, bench=False)
    test_batch_mxfp8_quant(M=4096, N=8192, bench=False)
