# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import pytest
import torch

from linghe.gemm.blockwise_fp8_gemm import triton_blockwise_fp8_gemm
from linghe.tools.check import output_check


@pytest.mark.parametrize(
    "M,N,K",
    [
        (4096, 8192, 2048),
    ],
)
def test_blockwise_gemm(M, N, K, benchmark):
    dtype = torch.bfloat16
    device = "cuda:0"
    B = 64

    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)

    x_scales = torch.rand((M // B, K // B), dtype=torch.float32, device=device)
    w_scales = torch.rand((N // B, K // B), dtype=torch.float32, device=device)

    x_q = x.to(torch.float8_e4m3fn)
    w_q = w.to(torch.float8_e4m3fn)

    x_dq = (x_q.float().view(M // B, B, K // B, B) * x_scales[:, None, :, None]).view(
        M, K
    )
    w_dq = (w_q.float().view(N // B, B, K // B, B) * w_scales[:, None, :, None]).view(
        N, K
    )

    y_ref = x_dq @ w_dq.t()
    y = triton_blockwise_fp8_gemm(
        x_q, w_q, x_scales, w_scales, out_dtype=dtype, block_size=B
    )
    output_check(y_ref.to(dtype), y, name="y", rtol=0.05, atol=1.0)

    n_repeat = 100
    ref_flops = M * N * K * 2

    benchmark(
        triton_blockwise_fp8_gemm,
        x_q,
        w_q,
        x_scales,
        w_scales,
        out_dtype=dtype,
        block_size=B,
        n_repeat=n_repeat,
        ref_flops=ref_flops,
    )
