# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.infer.gemm import (triton_split_fp32_gemm,
                               triton_tile_block_fp8_gemm,
                               triton_split_tile_block_fp8_gemm,
                               triton_tma_persistent_matmul)
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.tools.util import (torch_group_quant,
                               torch_block_quant,
                               torch_group_dequant,
                               torch_blockwise_dequant)

def torch_fp64_matmul(x, w):
    return torch.nn.functional.linear(x.to(torch.float64),
                                      w.to(torch.float64)).to(torch.float32)


def torch_fp32_matmul(x, w):
    return torch.nn.functional.linear(x.to(torch.float32), w)

def torch_fp16_matmul(x, w):
    return torch.nn.functional.linear(x, w)

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
        ref_time = benchmark_func(torch_fp16_matmul, x, w,
                                  ref_bytes=ref_bytes,
                                  ref_flops=ref_flops)
        benchmark_func(torch_fp32_matmul, x, w.float(),
                                  ref_bytes=ref_bytes,
                                  ref_flops=ref_flops,
                                  ref_time=ref_time)
        benchmark_func(triton_split_fp32_gemm, x, w,
                       ref_bytes=ref_bytes,
                       ref_flops=ref_flops,
                       ref_time=ref_time,
                       n_profile=0)
        benchmark_func(triton_tma_persistent_matmul, x, w,
                       ref_bytes=ref_bytes,
                       ref_flops=ref_flops,
                       ref_time=ref_time,
                       n_profile=0)


def test_fp8_matmul(M=2048, N=1024, K=8192, transpose_scale=True, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.randn(M, K, dtype=dtype, device=device, requires_grad=True)
    w = torch.randn(N, K, dtype=dtype, device=device, requires_grad=True)

    # y_ref = torch_fp32_matmul(x, w.float())

    x_q, x_s = torch_group_quant(x)
    if transpose_scale:
        p = (4 - M % 4) % 4
        x_s = torch.cat([x_s, torch.zeros(p, K//128, device=device, dtype=torch.float32)], 0)
        x_s = x_s.t().contiguous()[:,:M].t()
    w_q, w_s = torch_block_quant(w)

    x_dq = torch_group_dequant(x_q, x_s)
    w_dq = torch_blockwise_dequant(w_q, w_s)
    y_ref = torch_fp32_matmul(x_dq, w_dq)
    atol = 0.02 * y_ref.abs().mean().item()

    y = triton_tile_block_fp8_gemm(x_q, w_q, x_s, w_s)
    output_check(y_ref.to(dtype), y, name='y', atol=atol, rtol=2e-2)
    
    y = triton_split_tile_block_fp8_gemm(x_q, w_q, x_s, w_s)
    output_check(y_ref.to(dtype), y, name='split', atol=atol, rtol=2e-2)
    
    if bench:
        ref_bytes = M * K + N * K + M * N * 2
        ref_flops = 2 * M * N * K
        ref_time = benchmark_func(torch_fp16_matmul, x, w,
                                  ref_bytes=ref_bytes,
                                  ref_flops=ref_flops,
                                  n_profile=10)
        benchmark_func(triton_tile_block_fp8_gemm, x_q, w_q, x_s, w_s,
                       ref_bytes=ref_bytes,
                       ref_flops=ref_flops,
                       ref_time=ref_time,
                       n_profile=10)
        benchmark_func(triton_split_tile_block_fp8_gemm, x_q, w_q, x_s, w_s,
                       ref_bytes=ref_bytes,
                       ref_flops=ref_flops,
                       ref_time=ref_time,
                       n_profile=10)


if __name__ == '__main__':
    test_fp32_matmul(M=4096, N=256, K=8192, bench=False)
    test_fp32_matmul(M=16384, N=256, K=2048, bench=False)
    test_fp32_matmul(M=1235, N=256, K=8192, bench=False)

    test_fp32_matmul(M=2048, N=157184, K=2048, bench=False)
    test_fp32_matmul(M=2048-32, N=157184, K=4096, bench=False)
    test_fp32_matmul(M=2048-1, N=157184, K=8192, bench=False)
    test_fp32_matmul(M=4, N=157184, K=8192, bench=False)

    test_fp32_matmul(M=128, N=256, K=8192, bench=False)
    test_fp32_matmul(M=4, N=256, K=4096, bench=False)
    test_fp32_matmul(M=0, N=256, K=8192, bench=False)

    test_fp8_matmul(M=4096, N=1024, K=4096, transpose_scale=True, bench=False)
    test_fp8_matmul(M=1, N=1024, K=4096, transpose_scale=True, bench=False)
    test_fp8_matmul(M=1, N=1024, K=4096, transpose_scale=False, bench=False)
    test_fp8_matmul(M=1, N=4096, K=256, transpose_scale=True, bench=False)
    test_fp8_matmul(M=35, N=4096, K=256, transpose_scale=True, bench=False)
    test_fp8_matmul(M=87, N=1024, K=4096, transpose_scale=True, bench=False)