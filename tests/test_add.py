# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import pytest
import torch

from linghe.tools.check import output_check
from linghe.utils.add import triton_inplace_add, triton_batch_inplace_add


def torch_add(x, y, accum=True):
    if accum:
        x += y
        return x
    else:
        return x.copy_(y)


def torch_batch_add(xs, ys, accum=True):
    if accum:
        for i, x in enumerate(xs):
            x += ys[i]
        return xs
    else:
        for i, x in enumerate(xs):
            x.copy_(ys[i])
        return xs


@pytest.mark.parametrize(
    "M,N",
    [
        (4096, 4096),
        (4096, 3467),
        (3467, 3467),
    ],
)
@pytest.mark.parametrize("accum", [False, True])
def test_triton_inplace_add(M, N, accum, benchmark):
    x_dtype = torch.bfloat16
    y_dtype = torch.float32
    device = "cuda:0"

    x = torch.randn(M, N, dtype=x_dtype, device=device)
    y = torch.randn(M, N, dtype=y_dtype, device=device)

    out = x.clone()
    triton_inplace_add(out, y, accum=accum)
    out_ref = x.clone()
    out_ref = torch_add(out_ref, y, accum=accum)
    output_check(out_ref, out, "out")

    ref_bytes = M * N * (x_dtype.itemsize * (2 if accum else 1) + x_dtype.itemsize)
    ref_time = benchmark(torch_add, x, y, accum=accum, ref_bytes=ref_bytes)
    benchmark(
        triton_inplace_add, x, y, accum=accum, ref_time=ref_time, ref_bytes=ref_bytes
    )


@pytest.mark.parametrize(
    "B,N",
    [
        (8, 4096),
        (8, 4097),
    ],
)
@pytest.mark.parametrize("accum", [False, True])
def test_triton_batch_inplace_add(B, N, accum, benchmark, M=4096):
    x_dtype = torch.bfloat16
    y_dtype = torch.float32
    device = "cuda:0"

    xs = [torch.randn(M, N, dtype=x_dtype, device=device) for _ in range(B)]
    ys = [torch.randn(M, N, dtype=y_dtype, device=device) for _ in range(B)]

    out_ref = [x.clone() for x in xs]
    out_ref = torch_batch_add(out_ref, ys, accum=accum)
    out_ref = torch.cat([x.view(-1) for x in out_ref], 0)

    out = [x.clone() for x in xs]
    triton_batch_inplace_add(out, ys, accum=accum)
    out = torch.cat([x.view(-1) for x in out], 0)
    output_check(out_ref, out, "batch_add")

    ref_bytes = M * N * (x_dtype.itemsize * (2 if accum else 1) + x_dtype.itemsize)
    ref_time = benchmark(torch_batch_add, xs, ys, accum=accum, ref_bytes=ref_bytes)
    benchmark(
        triton_batch_inplace_add,
        xs,
        ys,
        accum=accum,
        ref_bytes=ref_bytes,
        ref_time=ref_time,
    )
