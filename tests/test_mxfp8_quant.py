# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import random

import torch

from linghe.quant.mxfp8 import triton_mxfp8_quant, triton_batch_mxfp8_quant
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.tools.util import torch_mxfp8_quant


def torch_batch_mxfp8_quant(x, token_count_per_expert_list):
    M, DIM = x.shape
    q_refs = []
    s_refs = []
    qt_refs = []
    st_refs = []
    s = 0
    for i, c in enumerate(token_count_per_expert_list):
        c = token_count_per_expert_list[i]
        if c == 0:
            continue
        y = x[s:s + c]
        y = y.float()

        y_q, y_scale, yt_q, yt_scale = torch_mxfp8_quant(y)
        q_refs.append(y_q)
        s_refs.append(y_scale)
        qt_refs.append(yt_q)
        st_refs.append(yt_scale)
        s += c
    q_ref = torch.cat(q_refs, 0)
    s_ref = torch.cat(s_refs, 0)
    qt_ref = torch.cat(qt_refs, 0)
    st_ref = torch.cat(st_refs, 0)
    return q_ref, s_ref, qt_ref, st_ref


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
