# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.gemm.blockwise_fp8_gemm import triton_bb_fp8_gemm, \
    triton_tt_fp8_gemm
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check


def test_triton_bb_gemm(M=4096, N=4096, K=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'
    B = 64

    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)

    x_scales = torch.rand((M // B, K // B), dtype=torch.float32, device=device)
    w_scales = torch.rand((N // B, K // B), dtype=torch.float32, device=device)

    x_q = x.to(torch.float8_e4m3fn)
    w_q = w.to(torch.float8_e4m3fn)

    x_dq = (x_q.float().view(M // B, B, K // B, B) * x_scales[:, None, :,
                                                     None]).view(M, K)
    w_dq = (w_q.float().view(N // B, B, K // B, B) * w_scales[:, None, :,
                                                     None]).view(N, K)

    y_ref = x_dq @ w_dq.t()
    y = triton_bb_fp8_gemm(x_q, w_q, x_scales, w_scales,
                           out_dtype=dtype, block_size=B)
    output_check(y_ref.to(dtype), y, name='y', rtol=0.05, atol=1.0)

    if bench:
        n_repeat = 100
        ref_flops = M * N * K * 2

        benchmark_func(triton_bb_fp8_gemm, x_q, w_q, x_scales, w_scales,
                       out_dtype=dtype, block_size=B,
                       n_repeat=n_repeat, ref_flops=ref_flops)


def test_triton_tt_gemm(M=4096, N=4096, K=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'
    B = 64

    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)

    x_scales = torch.rand((M, K // B), dtype=torch.float32, device=device)
    w_scales = torch.rand((N, K // B), dtype=torch.float32, device=device)

    x_q = x.to(torch.float8_e4m3fn)
    w_q = w.to(torch.float8_e4m3fn)

    x_dq = (x_q.float().view(M, K // B, B) * x_scales[:, :, None]).view(M, K)
    w_dq = (w_q.float().view(N, K // B, B) * w_scales[:, :, None]).view(N, K)

    y_ref = x_dq @ w_dq.t()
    y = triton_tt_fp8_gemm(x_q, w_q, x_scales, w_scales,
                           out_dtype=dtype, block_size=B)
    output_check(y_ref.to(dtype), y, 'y', atol=1.0, rtol=0.05)

    if bench:
        n_repeat = 100
        ref_flops = M * N * K * 2

        benchmark_func(triton_tt_fp8_gemm, x_q, w_q, x_scales, w_scales,
                       out_dtype=dtype, block_size=B,
                       n_repeat=n_repeat, ref_flops=ref_flops)


if __name__ == '__main__':
    test_triton_bb_gemm(M=4096, N=8192, K=2048, bench=False)
    test_triton_tt_gemm(M=4096, N=8192, K=2048, bench=False)
