# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.gemm.fp32_gemm import (triton_fp32_gemm,
                                  triton_fp32_gemm_for_backward,
                                  triton_fp32_gemm_for_update)
from linghe.facade.fp32_gemm import fp32_gemm
from linghe.tools.benchmark import benchmark_func
from linghe.tools.util import output_check



def torch_fp32_matmul(x, w):
    return torch.nn.functional.linear(x.float(), w.float())

def torch_fp32_matmul_backward(dy, w):
    return (dy @ w).to(torch.bfloat16)

def torch_fp32_matmul_update(dy, x):
    return (dy.transpose(-2, -1) @ x).to(torch.bfloat16)


def test_fp32_matmul(M=2048, N=256, K=8192, bench=False):
    # M, N, K = 4096, 256, 8192
    dtype = torch.bfloat16
    device = 'cuda:0'
    n_repeat = 100

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

    output_check(y_ref, y, mode='forward')
    output_check(dx_ref, dx, mode='backward')
    output_check(dw_ref, dw, mode='update')

    x.grad = None 
    w.grad = None
    y = fp32_gemm(x, w)
    y.backward(gradient=dy)
    dx = x.grad 
    dw = w.grad
    output_check(y_ref, y, mode='forward')
    output_check(dx_ref, dx, mode='backward')
    output_check(dw_ref, dw, mode='update')



    if bench:
        print('\nbenchmark\n')
        ref_time = benchmark_func(torch_fp32_matmul, x, w, n_repeat=n_repeat,
                                  ref_bytes=M * K * 6 + N * K * 6 + M * N * 4,
                                  ref_flops=2 * M * N * K)
        benchmark_func(triton_fp32_gemm, x, w, n_repeat=n_repeat,
                       ref_bytes=M * K * 6 + N * K * 6 + M * N * 4,
                       ref_flops=2 * M * N * K, ref_time=ref_time)

        ref_time = benchmark_func(torch_fp32_matmul_backward, dy, w.float(),
                                  n_repeat=n_repeat,
                                  ref_bytes=M * K * 10 + N * K * 4 + M * N * 4,
                                  ref_flops=2 * M * N * K)
        benchmark_func(triton_fp32_gemm_for_backward, dy, w,
                       n_repeat=n_repeat,
                       ref_bytes=M * K * 2 + N * K * 2 + M * N * 4,
                       ref_flops=2 * M * N * K, ref_time=ref_time)

        ref_time = benchmark_func(torch_fp32_matmul_update, dy, x.float(),
                                  n_repeat=n_repeat,
                                  ref_bytes=M * K * 4 + N * K * 12 + M * N * 4,
                                  ref_flops=2 * M * N * K)
        benchmark_func(triton_fp32_gemm_for_update, dy, x, n_repeat=n_repeat,
                       ref_bytes=M * K * 2 + N * K * 8 + M * N * 4,
                       ref_flops=2 * M * N * K, ref_time=ref_time)


def test_batch_fp32_matmul(B=2, M=2048, N=256, K=8192, bench=False):
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
    output_check(y_ref, y, mode='forward')
    output_check(dx_ref, dx, mode='backward')
    output_check(dw_ref, dw, mode='update')

    
    if bench:
        print('\nbenchmark\n')
        ref_time = benchmark_func(torch_fp32_matmul, x, w, n_repeat=n_repeat,
                                  ref_bytes=M * K * 6 + N * K * 6 + M * N * 4,
                                  ref_flops=2 * M * N * K)
        benchmark_func(fp32_gemm, x, w, n_repeat=n_repeat,
                       ref_bytes=M * K * 6 + N * K * 6 + M * N * 4,
                       ref_flops=2 * M * N * K, ref_time=ref_time)


if __name__ == '__main__':
    test_fp32_matmul(M=2048, N=256, K=8192, bench=False)
    test_fp32_matmul(M=2048, N=16, K=8192, bench=False)
    test_fp32_matmul(M=128, N=16, K=128, bench=False)
    test_batch_fp32_matmul(B=2, M=2048, N=256, K=8192, bench=False)
    test_batch_fp32_matmul(B=2,M=2048, N=16, K=8192, bench=False)
    test_batch_fp32_matmul(B=2,M=128, N=16, K=128, bench=False)


