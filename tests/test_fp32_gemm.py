# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.gemm.fp32_gemm import (triton_fp32_gemm,
                                  triton_fp32_gemm_for_backward,
                                  triton_fp32_gemm_for_update)
from linghe.tools.benchmark import benchmark_func
from linghe.tools.util import output_check



def torch_fp32_matmul(x, w):
    return torch.nn.functional.linear(x.float(), w.float())

def torch_fp32_matmul_backward(dy, w):
    return (dy @ w).to(torch.bfloat16)

def torch_fp32_matmul_update(y, x):
    return (y.t() @ x).to(torch.bfloat16)


def test_fp32_matmul(M=2048, N=256, K=8192, bench=False):
    # M, N, K = 4096, 256, 8192
    dtype = torch.bfloat16
    device = 'cuda:0'
    n_repeat = 100

    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    scale = torch.randn(M, dtype=torch.float32, device=device)
    dy = torch.randn(M, N, dtype=torch.float32, device=device)

    y_ref = torch_fp32_matmul(x, w)
    y = triton_fp32_gemm(x, w)
    output_check(y_ref, y.float(), mode='fp32_gemm')

    dx = torch.zeros(M, K, dtype=dtype, device=device)
    dx = triton_fp32_gemm_for_backward(dy, w)
    dx_ref = dy @ w.float()
    output_check(dx_ref, dx.float(), mode='backward')

    main_grad = triton_fp32_gemm_for_update(y, x)
    main_grad_ref = y.t() @ (x.float())
    output_check(main_grad_ref, main_grad.float(), mode='update')

    if bench:
        print('\nbenchmark\n')
        ref_time = benchmark_func(torch_fp32_matmul, x, w, n_repeat=n_repeat,
                                  ref_bytes=M * K * 6 + N * K * 6 + M * N * 4,
                                  ref_linghe=2 * M * N * K)
        benchmark_func(triton_fp32_gemm, x, w, n_repeat=n_repeat,
                       ref_bytes=M * K * 6 + N * K * 6 + M * N * 4,
                       ref_linghe=2 * M * N * K, ref_time=ref_time)
        benchmark_func(triton_scaled_fp32_gemm, x, w, scale, n_repeat=n_repeat,
                       ref_bytes=M * K * 6 + N * K * 6 + M * N * 4,
                       ref_linghe=2 * M * N * K, ref_time=ref_time)

        ref_time = benchmark_func(torch_fp32_matmul_backward, dy, w.float(),
                                  n_repeat=n_repeat,
                                  ref_bytes=M * K * 10 + N * K * 4 + M * N * 4,
                                  ref_linghe=2 * M * N * K)
        benchmark_func(triton_fp32_gemm_for_backward, dy, w,
                       n_repeat=n_repeat,
                       ref_bytes=M * K * 2 + N * K * 2 + M * N * 4,
                       ref_linghe=2 * M * N * K, ref_time=ref_time)

        ref_time = benchmark_func(torch_fp32_matmul_update, dy, x.float(),
                                  n_repeat=n_repeat,
                                  ref_bytes=M * K * 4 + N * K * 12 + M * N * 4,
                                  ref_linghe=2 * M * N * K)
        benchmark_func(triton_fp32_gemm_for_update, dy, x, n_repeat=n_repeat,
                       ref_bytes=M * K * 2 + N * K * 8 + M * N * 4,
                       ref_linghe=2 * M * N * K, ref_time=ref_time)
        benchmark_func(triton_scaled_fp32_gemm_for_update, dy, x, scale,
                       n_repeat=n_repeat,
                       ref_bytes=M * K * 2 + N * K * 8 + M * N * 4,
                       ref_linghe=2 * M * N * K, ref_time=ref_time)


if __name__ == '__main__':
    test_fp32_matmul(M=2048, N=256, K=8192)
