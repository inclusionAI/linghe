# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import random

import torch

from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.utils.unary import triton_calculate_smooth_scale, triton_batch_clip


def torch_calculate_smooth_scale(x, min_value=1.0, smooth_coef=0.5, round_scale=False):
    one = torch.ones([1], dtype=torch.float32, device=x.device)
    input_smooth_scales = torch.pow(torch.maximum(x, min_value * one), smooth_coef)
    weight_smooth_scales = 1 / input_smooth_scales
    if round_scale:
        weight_smooth_scales = torch.exp2(torch.ceil(torch.log2(weight_smooth_scales)))
    return weight_smooth_scales


def torch_batch_clip(xs, clip_value):
    torch._foreach_clamp_min_(xs, -clip_value)
    torch._foreach_clamp_max_(xs, clip_value)
    return xs


def test_calculate_smooth_scale(N=4096, bench=False):
    x = torch.randn(N, dtype=torch.float32, device="cuda:0").abs() ** 3 + 0.1

    min_value = 0.0
    smooth_coef = 0.5
    out_ref = torch_calculate_smooth_scale(
        x, min_value=min_value, smooth_coef=smooth_coef, round_scale=True
    )
    out = triton_calculate_smooth_scale(
        x, min_value=min_value, smooth_coef=smooth_coef, round_scale=True
    )
    output_check(out_ref, out, "torch_calculate_smooth_scale")

    n_repeat = 100

    if bench:
        ref_time = benchmark_func(torch_calculate_smooth_scale, x, n_repeat=n_repeat)
        benchmark_func(
            torch_calculate_smooth_scale,
            x,
            n_repeat=n_repeat,
            ref_time=ref_time,
            ref_bytes=N * 8,
        )


def test_batch_clip(M=2048, N=1024, k=1024, clip_value=1.0, inf=False, bench=False):
    shapes1 = [random.randint(1, int(M**0.5)) ** 2 for i in range(k)]
    shapes2 = [random.randint(1, int(N**0.5)) ** 2 for i in range(k)]
    xs = [
        torch.randn(shapes1[i], shapes2[i], dtype=torch.float32, device="cuda:0")
        for i in range(k)
    ]
    xs1 = [x.clone().detach() for x in xs]
    xs2 = [x.clone().detach() for x in xs]

    if inf:
        xs1[0][:100] = float("inf")
        xs2[0][:100] = float("inf")

    sum_ref = torch_batch_clip(xs1, clip_value)
    sums = triton_batch_clip(xs2, clip_value)
    output_check(
        torch.cat([x.view(-1) for x in sum_ref], 0),
        torch.cat([x.view(-1) for x in sums], 0),
        "batch_clip",
    )

    if bench:
        ref_bytes = sum([x.numel() for x in xs]) * 8
        xs3 = [x.clone().detach() for x in xs]
        n_repeat = 1  # inplace update will speedup our triton op
        ref_time = benchmark_func(
            torch_batch_clip,
            xs3,
            clip_value,
            ref_bytes=ref_bytes,
            n_repeat=n_repeat,
            n_warmup=0,
        )
        xs4 = [x.clone().detach() for x in xs]
        benchmark_func(
            triton_batch_clip,
            xs4,
            clip_value,
            ref_bytes=ref_bytes,
            ref_time=ref_time,
            n_repeat=n_repeat,
            n_warmup=0,
        )


if __name__ == "__main__":
    # test_calculate_smooth_scale(N=4096*32)
    # test_calculate_smooth_scale(N=4096*32-1897)
    # test_batch_clip(M=2048, N=8192, k=128, clip_value=0.1, bench=False)
    # test_batch_clip(M=2048, N=1024, k=128, clip_value=1.0, bench=False)
    test_batch_clip(M=2048, N=1024, k=128, clip_value=100.0, inf=True, bench=False)
