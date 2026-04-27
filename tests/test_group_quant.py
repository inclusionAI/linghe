# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import pytest
import torch

from linghe.quant.group import triton_group_quant
from linghe.tools.check import output_check
from linghe.tools.util import torch_group_quant


@pytest.mark.parametrize(
    "M,N",
    [
        (4096, 4096),
        (4096, 8192),
        (2049, 8192),
        (2049, 1536),
    ],
)
def test_group_quant(M, N, benchmark, B=128, round_scale=False):
    x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda:0") ** 3
    xq_ref, x_scale_ref = torch_group_quant(x, B, round_scale=round_scale)
    xq, x_scale = triton_group_quant(x, group_size=B, round_scale=round_scale)
    output_check(xq_ref, xq, name="data")
    output_check(x_scale_ref, x_scale, name="scale")

    n_repeat = 100
    benchmark(
        triton_group_quant, x, group_size=B, n_repeat=n_repeat, ref_bytes=M * N * 3
    )
