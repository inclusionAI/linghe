# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import pytest
import torch

from linghe.quant.channel import (
    triton_deprecated_tokenwise_row_quant,
    triton_row_quant,
    triton_tokenwise_row_quant,
)
from linghe.tools.check import output_check
from linghe.tools.util import torch_row_quant


@pytest.mark.parametrize(
    "M,N",
    [
        (4096, 4096),
        (4090, 4096),
        (4096, 8192),
        (3456, 2048),
        (1, 2048),
    ],
)
@pytest.mark.parametrize("round_scale", [False, True])
def test_row_quant(M, N, round_scale, benchmark):
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

    ref_time = benchmark(torch_row_quant, x, n_repeat=100, ref_bytes=M * N * 3)
    benchmark(triton_row_quant, x, n_repeat=100, ref_bytes=M * N * 3, ref_time=ref_time)
    benchmark(
        triton_deprecated_tokenwise_row_quant,
        x,
        n_repeat=100,
        ref_bytes=M * N * 3,
        ref_time=ref_time,
    )
    benchmark(
        triton_tokenwise_row_quant,
        x,
        n_repeat=100,
        ref_bytes=M * N * 3,
        ref_time=ref_time,
    )
