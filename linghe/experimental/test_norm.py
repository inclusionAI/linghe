# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.experimental.norm import triton_rms_norm_forward, \
    triton_parallel_rms_norm_and_block_quant_forward
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.utils.norm import triton_rms_norm_and_block_quant_forward


def torch_rms_forward(x, weight):
    dtype = x.dtype
    x = x.float()
    weight = weight.float()
    N = x.shape[-1]
    rmsnorm = torch.nn.RMSNorm(
        normalized_shape=N,
        eps=1e-6,
        dtype=torch.float32,
        device=x.device
    )
    with torch.no_grad():
        rmsnorm.weight.copy_(weight)
    rms = torch.rsqrt(torch.sum(x ** 2, 1) / N + 1e-6)
    return rmsnorm(x).to(dtype), rms


def test_norm(M=4096, N=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.ones(M, N, dtype=dtype, requires_grad=False, device=device)
    weight = torch.ones(N, dtype=dtype, requires_grad=False, device=device)

    output_ref, rms_ref = torch_rms_forward(x, weight)
    output, rms = triton_rms_norm_forward(x, weight)

    output_check(rms_ref, rms, name="rms", rtol=0.001)
    output_check(output_ref, output, name="output", rtol=0.001)


def test_parallel_rmsnorm_and_block_quant(M=4096, N=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.randn(M, N, dtype=dtype, requires_grad=True, device=device)
    weight = torch.randn(N, dtype=dtype, requires_grad=True, device=device)

    q_ref, scale_ref, rms_ref, qt_ref, scale_t_ref = triton_rms_norm_and_block_quant_forward(
        x, weight,
        round_scale=False,
        output_mode=2)

    q, scale, rms, qt, scale_t = triton_parallel_rms_norm_and_block_quant_forward(
        x, weight,
        round_scale=False,
        output_mode=2)
    output_check(q_ref, q, name='parallel.block.data', rtol=-0.125)
    output_check(scale_ref, scale, name="parallel.block.scale", rtol=-0.125)
    output_check(rms_ref, rms, name="parallel.block.rms", rtol=-0.125)
    output_check(qt_ref, qt, name='parallel.block.t_data', rtol=-0.125)
    output_check(scale_t_ref, scale_t, name="parallel.block.t_scale",
                 rtol=-0.125)

    if bench:
        benchmark_func(triton_rms_norm_and_block_quant_forward, x, weight,
                       round_scale=False,
                       output_mode=2,
                       ref_bytes=M * N * 4,
                       n_profile=2)

        benchmark_func(triton_parallel_rms_norm_and_block_quant_forward, x,
                       weight,
                       round_scale=False,
                       output_mode=2,
                       ref_bytes=M * N * 4,
                       n_profile=2)


if __name__ == '__main__':
    # /usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/bin/ptxas -lineinfo -v --gpu-name=sm_90a /tmp/tmp3l_m5rfp.ptx -o /tmp/tmp3l_m5rfp.ptx.o
    test_norm(M=1024, N=4096, bench=False)
    # test_parallel_rmsnorm_and_block_quant(M=4096, N=4096, bench=False)
