# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import random

import pytest
import torch

from linghe.tools.check import output_check
from linghe.utils.mul import triton_dot, triton_batch_scale, triton_inplace_scale


def torch_fp16_dot(x, y):
    return (x * y).sum(1)


def torch_inplace_scale(x, scale):
    x *= scale
    return x


def torch_batch_scale(xs, scale):
    return [x * scale for x in xs]


@pytest.mark.parametrize(
    "M,N",
    [
        (4096, 4096),
    ],
)
def test_dot(M, N, benchmark):
    dtype = torch.bfloat16
    device = "cuda:0"

    n_repeat = 100

    x = torch.randn(M, N, dtype=dtype, device=device)
    y = torch.randn(M, N, dtype=dtype, device=device)
    q = torch.randn(M, N, dtype=dtype, device=device).to(torch.float8_e4m3fn)
    quant_scale = torch.randn(M, dtype=torch.float32, device=device).abs()
    smooth_scale = torch.randn(N, dtype=torch.float32, device=device).abs()

    sums_ref = torch_fp16_dot(x, q.float().to(dtype))
    sums = triton_dot(x, q)
    output_check(sums_ref, sums, "dot", atol=1.0)

    sums_ref = (
        x.float() * (q.to(torch.float32) * quant_scale[:, None] * smooth_scale[None, :])
    ).sum(dim=1)

    ref_time = benchmark(torch_fp16_dot, x, y, n_repeat=n_repeat)
    benchmark(triton_dot, x, q, n_repeat=n_repeat, ref_time=ref_time)


@pytest.mark.parametrize(
    "M",
    [
        2**28 + 1,
    ],
)
def test_inplace_scale(M, benchmark):
    x = torch.randn((M,), device="cuda:0", dtype=torch.float32)
    scale = 7.86
    sum_ref = torch_inplace_scale(x, scale)
    sums = triton_inplace_scale(x, scale)
    output_check(sum_ref, sums, "sum")

    ref_bytes = M * 8

    ref_time = benchmark(torch_inplace_scale, x, scale, ref_bytes=ref_bytes)
    benchmark(triton_inplace_scale, x, scale, ref_bytes=ref_bytes, ref_time=ref_time)


@pytest.mark.parametrize(
    "M,N,k,scale",
    [
        (2048, 1024, 128, 2.0),
        (2048, 1024, 128, 0.0),
    ],
)
def test_batch_scale(M, N, k, scale, benchmark):
    dtype = torch.float32
    xs = [
        torch.randn(
            random.randint(1, int(M**0.5)) ** 2,
            random.randint(1, int(N**0.5)) ** 2,
            dtype=dtype,
            device="cuda:0",
        )
        for i in range(k)
    ]
    xs1 = [x.clone().detach() for x in xs]
    xs2 = [x.clone().detach() for x in xs]
    if scale == 0.0:
        xs1[0][:10] = float("inf")
        xs1[0][10:20] = -float("inf")
        xs2[0][:10] = float("inf")
        xs2[0][10:20] = -float("inf")

    sum_ref = torch_batch_scale(xs1, scale)
    sums = triton_batch_scale(xs2, scale)
    output_check(
        torch.cat([x.view(-1) for x in sum_ref], 0),
        torch.cat([x.view(-1) for x in sums], 0),
        "batch_clip",
    )

    ref_bytes = sum([x.numel() for x in xs]) * 8

    ref_time = benchmark(torch_batch_scale, xs, scale, ref_bytes=ref_bytes)
    benchmark(triton_batch_scale, xs, scale, ref_bytes=ref_bytes, ref_time=ref_time)
