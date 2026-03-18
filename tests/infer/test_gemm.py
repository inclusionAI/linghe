# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.infer.gemm import triton_split_fp32_gemm
from linghe.gemm.fp32_gemm import triton_tma_persistent_matmul
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check


def torch_fp64_matmul(x, w):
    return torch.nn.functional.linear(x.to(torch.float64),
                                      w.to(torch.float64)).to(torch.float32)


def torch_fp32_matmul(x, w):
    return torch.nn.functional.linear(x.to(torch.float32), w)


def test_fp32_matmul(M=2048, N=256, K=8192, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)
    w = torch.randn(N, K, dtype=dtype, device=device, requires_grad=True)

    y_ref = torch_fp32_matmul(x, w.float())

    y = triton_split_fp32_gemm(x, w)
    output_check(y_ref, y, name='split.y', atol=5e-3, rtol=2e-3)
    
    y = triton_tma_persistent_matmul(x, w)
    output_check(y_ref, y, name='tma.y', atol=5e-3, rtol=2e-3)

    if bench:
        ref_bytes = M * K * 6 + N * K * 6 + M * N * 4
        ref_flops = 2 * M * N * K
        ref_time = benchmark_func(torch_fp32_matmul, x, w.float(),
                                  ref_bytes=ref_bytes,
                                  ref_flops=ref_flops)
        benchmark_func(triton_split_fp32_gemm, x, w,
                       ref_bytes=ref_bytes,
                       ref_flops=ref_flops, ref_time=ref_time)
        benchmark_func(triton_tma_persistent_matmul, x, w,
                       ref_bytes=ref_bytes,
                       ref_flops=ref_flops, ref_time=ref_time)

if __name__ == '__main__':
    test_fp32_matmul(M=4096, N=256, K=8192, bench=True)
    test_fp32_matmul(M=16384, N=256, K=2048, bench=True)
    test_fp32_matmul(M=1235, N=256, K=8192, bench=True)

    test_fp32_matmul(M=2048, N=157184, K=2048, bench=True)
    test_fp32_matmul(M=2048-32, N=157184, K=4096, bench=True)
    test_fp32_matmul(M=2048-1, N=157184, K=8192, bench=True)

    test_fp32_matmul(M=128, N=256, K=8192, bench=True)
    test_fp32_matmul(M=33, N=256, K=8192, bench=True)
    test_fp32_matmul(M=0, N=256, K=8192, bench=True)

