# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.quant.channel import (
    triton_deprecated_tokenwise_row_quant,
    triton_row_quant,
    triton_tokenwise_row_quant,
)
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.tools.util import torch_row_quant


def test_row_quant(M=4096, N=4096, round_scale=True, bench=False):
    device = "cuda:0"
    dtype = torch.bfloat16
    x = torch.randn((M, N), dtype=dtype, device=device) ** 3

    x_q_ref, x_scale_ref = torch_row_quant(x, round_scale=round_scale)

    x_q, x_scale = triton_row_quant(x, round_scale=round_scale)
    output_check(x_q_ref, x_q, name="data")
    output_check(x_scale_ref, x_scale, name="scale")

    x_q, x_scale = triton_tokenwise_row_quant(x, round_scale=round_scale)
    output_check(x_q_ref, x_q, name="data")
    output_check(x_scale_ref, x_scale, name="scale")

    if bench:
        ref_time = benchmark_func(torch_row_quant, x, n_repeat=100, ref_bytes=M * N * 3)
        benchmark_func(
            triton_row_quant, x, n_repeat=100, ref_bytes=M * N * 3, ref_time=ref_time
        )
        benchmark_func(
            triton_deprecated_tokenwise_row_quant,
            x,
            n_repeat=100,
            ref_bytes=M * N * 3,
            ref_time=ref_time,
        )
        benchmark_func(
            triton_tokenwise_row_quant,
            x,
            n_repeat=100,
            ref_bytes=M * N * 3,
            ref_time=ref_time,
        )


if __name__ == "__main__":
    test_row_quant(M=4096, N=4096, round_scale=False)
    test_row_quant(M=4090, N=4096, round_scale=True)
    test_row_quant(M=4096, N=8192, round_scale=True)
    test_row_quant(M=3456, N=2048, round_scale=True)
    test_row_quant(M=1, N=2048, round_scale=True)
