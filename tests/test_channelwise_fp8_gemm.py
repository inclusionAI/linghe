# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.gemm.channelwise_fp8_gemm import triton_scaled_mm
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.utils.add import triton_inplace_add


def scaled_gemm_and_update(x_q, w_q, x_scales, w_scales, c=None, accum=False):
    o = torch._scaled_mm(x_q,
                         w_q.t(),
                         scale_a=x_scales.view(-1, 1),
                         scale_b=w_scales.view(1, -1),
                         out_dtype=torch.bfloat16,
                         use_fast_accum=True)
    if accum:
        assert c is not None
        triton_inplace_add(c, o, accum=accum)
    else:
        c = o
    return c


def test_triton_channelwise_gemm(M=4096, N=4096, K=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.randn(M, K, dtype=dtype, device=device)
    x_scales = torch.rand((M,), dtype=torch.float32, device=device)
    x_q = x.to(torch.float8_e4m3fn)

    w = torch.randn(N, K, dtype=dtype, device=device)
    w_scales = torch.rand((N,), dtype=torch.float32, device=device)
    w_q = w.to(torch.float8_e4m3fn)

    y_ref = (x_q.float() * x_scales[:, None]) @ (
                w_q.float() * w_scales[:, None]).t()
    y = triton_scaled_mm(x_q, w_q, x_scales, w_scales, c=None,
                         accum=False)

    output_check(y_ref, y, name='y', atol=-1)

    if bench:
        y_bf16 = torch.randn(M, N, dtype=dtype, device=device)
        y_fp16 = y.to(torch.float16)
        y_fp32 = y.to(torch.float32)

        n_repeat = 100
        ref_flops = M * N * K * 2

        benchmark_func(scaled_gemm_and_update, x_q, w_q, x_scales, w_scales,
                       c=y_bf16,
                       accum=False, n_repeat=n_repeat, ref_flops=ref_flops)

        benchmark_func(scaled_gemm_and_update, x_q, w_q, x_scales, w_scales,
                       c=y_bf16,
                       accum=True, n_repeat=n_repeat, ref_flops=ref_flops)
        benchmark_func(scaled_gemm_and_update, x_q, w_q, x_scales, w_scales,
                       c=y_fp16,
                       accum=True, n_repeat=n_repeat, ref_flops=ref_flops)
        benchmark_func(scaled_gemm_and_update, x_q, w_q, x_scales, w_scales,
                       c=y_fp32,
                       accum=True, n_repeat=n_repeat, ref_flops=ref_flops)

        benchmark_func(triton_scaled_mm, x_q, w_q, x_scales, w_scales, c=y_bf16,
                       accum=True, n_repeat=n_repeat, ref_flops=ref_flops)
        benchmark_func(triton_scaled_mm, x_q, w_q, x_scales, w_scales, c=y_fp16,
                       accum=True, n_repeat=n_repeat, ref_flops=ref_flops)
        benchmark_func(triton_scaled_mm, x_q, w_q, x_scales, w_scales, c=y_fp32,
                       accum=True, n_repeat=n_repeat, ref_flops=ref_flops)


if __name__ == '__main__':
    test_triton_channelwise_gemm(M=4096, N=4096, K=4096, bench=False)
