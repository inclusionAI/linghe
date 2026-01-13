# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.facade.fp32_gemm import fp32_gemm
from linghe.gemm.fp32_gemm import (triton_fp32_gemm,
                                   triton_fp32_gemm_for_backward,
                                   triton_fp32_gemm_for_update,
                                   triton_split_fp32_gemm,
                                   triton_split_fp32_gemm_for_backward,
                                   triton_split_fp32_gemm_for_update)
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check


def torch_fp64_matmul(x, w):
    return torch.nn.functional.linear(x.to(torch.float64),
                                      w.to(torch.float64)).to(torch.float32)


def torch_fp32_matmul(x, w):
    return torch.nn.functional.linear(x.to(torch.float32), w.to(torch.float32))


def torch_fp32_matmul_backward(dy, w):
    return (dy @ w).to(torch.bfloat16)


def torch_fp32_matmul_update(dy, x):
    return (dy.transpose(-2, -1) @ x).to(torch.bfloat16)


def test_fp32_matmul(M=2048, N=256, K=8192, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)
    w = torch.randn(N, K, dtype=dtype, device=device, requires_grad=True)
    dy = torch.randn(M, N, dtype=torch.float32, device=device)

    y_ref = torch_fp32_matmul(x, w)
    y_ref.backward(gradient=dy)
    dx_ref = x.grad
    dw_ref = w.grad

    y = triton_fp32_gemm(x, w)
    dx = triton_fp32_gemm_for_backward(dy, w)
    dw = triton_fp32_gemm_for_update(dy, x)

    output_check(y_ref, y, name='y', atol=5e-3, rtol=2e-3)
    output_check(dx_ref, dx, name='dx', atol=2e-2, rtol=2e-2)
    output_check(dw_ref, dw.to(dtype), name='dw', atol=2e-1, rtol=2e-2)

    y = triton_split_fp32_gemm(x, w)
    dx = triton_split_fp32_gemm_for_backward(dy, w)
    dw = triton_split_fp32_gemm_for_update(dy, x)
    output_check(y_ref, y, name='split.y', atol=5e-3, rtol=2e-3)
    output_check(dx_ref, dx, name='split.dx', atol=2e-2, rtol=2e-2)
    output_check(dw_ref, dw.to(dtype), name='split.dw', atol=2e-1, rtol=2e-2)

    x.grad = None
    w.grad = None
    y = fp32_gemm(x, w)
    y.backward(gradient=dy)
    dx = x.grad
    dw = w.grad
    output_check(y_ref, y, name='y', atol=5e-3, rtol=2e-3)
    output_check(dx_ref, dx, name='dx', atol=2e-2, rtol=2e-2)
    output_check(dw_ref, dw.to(dtype), name='dw', atol=2e-1, rtol=2e-2)

    if bench:
        ref_bytes = M * K * 6 + N * K * 6 + M * N * 4
        ref_flops = 2 * M * N * K
        ref_time = benchmark_func(torch_fp32_matmul, x, w,
                                  ref_bytes=ref_bytes,
                                  ref_flops=ref_flops)
        benchmark_func(triton_fp32_gemm, x, w,
                       ref_bytes=ref_bytes,
                       ref_flops=ref_flops, ref_time=ref_time)
        benchmark_func(triton_split_fp32_gemm, x, w,
                       ref_bytes=ref_bytes,
                       ref_flops=ref_flops, ref_time=ref_time)

        ref_bytes = M * K * 10 + N * K * 4 + M * N * 4
        ref_time = benchmark_func(torch_fp32_matmul_backward, dy, w.float(),
                                  ref_bytes=ref_bytes,
                                  ref_flops=ref_flops)
        benchmark_func(triton_fp32_gemm_for_backward, dy, w,
                       ref_bytes=ref_bytes,
                       ref_flops=ref_flops, ref_time=ref_time)
        benchmark_func(triton_split_fp32_gemm_for_backward, dy, w,
                       ref_bytes=ref_bytes,
                       ref_flops=ref_flops, ref_time=ref_time)

        ref_bytes = M * K * 4 + N * K * 12 + M * N * 4
        ref_time = benchmark_func(torch_fp32_matmul_update, dy, x.float(),
                                  ref_bytes=ref_bytes,
                                  ref_flops=ref_flops)
        benchmark_func(triton_fp32_gemm_for_update, dy, x,
                       ref_bytes=ref_bytes,
                       ref_flops=ref_flops, ref_time=ref_time)
        benchmark_func(triton_split_fp32_gemm_for_update, dy, x,
                       ref_bytes=ref_bytes,
                       ref_flops=ref_flops, ref_time=ref_time)


def test_BMK_fp32_matmul(B=2, M=2048, N=256, K=8192, bench=False):
    # M, N, K = 4096, 256, 8192
    dtype = torch.bfloat16
    device = 'cuda:0'
    n_repeat = 100

    x = torch.randn(B, M, K, dtype=dtype, device=device, requires_grad=True)
    w = torch.randn(N, K, dtype=dtype, device=device, requires_grad=True)
    dy = torch.randn(B, M, N, dtype=torch.float32, device=device)

    y_ref = torch_fp32_matmul(x, w)
    y_ref.backward(gradient=dy)
    dx_ref = x.grad
    dw_ref = w.grad

    x.grad = None
    w.grad = None
    y = fp32_gemm(x, w)
    y.backward(gradient=dy)
    dx = x.grad
    dw = w.grad
    output_check(y_ref, y, name='forward', atol=5e-3, rtol=2e-3)
    output_check(dx_ref, dx, name='backward', atol=1e-1, rtol=2e-2)
    output_check(dw_ref, dw, name='update', atol=1e-1, rtol=2e-2)

    if bench:
        print('\nbenchmark\n')
        ref_time = benchmark_func(torch_fp32_matmul, x, w, n_repeat=n_repeat,
                                  ref_bytes=M * K * 6 + N * K * 6 + M * N * 4,
                                  ref_flops=2 * M * N * K)
        benchmark_func(fp32_gemm, x, w, n_repeat=n_repeat,
                       ref_bytes=M * K * 6 + N * K * 6 + M * N * 4,
                       ref_flops=2 * M * N * K, ref_time=ref_time)


if __name__ == '__main__':
    test_fp32_matmul(M=4096, N=256, K=8192, bench=False)
    test_fp32_matmul(M=16384, N=256, K=2048, bench=False)
    test_fp32_matmul(M=128, N=16, K=128, bench=False)
    test_BMK_fp32_matmul(B=2, M=2048, N=16, K=8192, bench=False)
    test_BMK_fp32_matmul(B=2, M=2048, N=256, K=8192, bench=False)
    test_BMK_fp32_matmul(B=2, M=128, N=16, K=128, bench=False)
