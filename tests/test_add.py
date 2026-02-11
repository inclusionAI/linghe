# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.tools.benchmark import benchmark_func
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

def test_triton_inplace_add(M=4096, N=4096, accum=True, bench=False):
    x_dtype = torch.bfloat16
    y_dtype = torch.float32
    device = 'cuda:0'

    x = torch.randn(M, N, dtype=x_dtype, device=device)
    y = torch.randn(M, N, dtype=y_dtype, device=device)

    out = x.clone()
    triton_inplace_add(out, y, accum=accum)
    out_ref = x.clone()
    out_ref = torch_add(out_ref, y, accum=accum)
    output_check(out_ref, out, 'out')

    if bench:
        ref_bytes = M * N * (x_dtype.itemsize * (2 if accum else 1) + x_dtype.itemsize)
        ref_time = benchmark_func(torch_add, x, y, accum=accum,
                                  ref_bytes=ref_bytes)
        benchmark_func(triton_inplace_add, x, y, accum=accum,
                       ref_time=ref_time, ref_bytes=ref_bytes)


def test_triton_batch_inplace_add(B=32, M=4096, N=4096, accum=True, bench=False):
    x_dtype = torch.bfloat16
    y_dtype = torch.float32
    device = 'cuda:0'

    xs = [torch.randn(M, N, dtype=x_dtype, device=device) for _ in range(B)]
    ys = [torch.randn(M, N, dtype=y_dtype, device=device) for _ in range(B)]

    out_ref = [x.clone() for x in xs]
    out_ref = torch_batch_add(out_ref, ys, accum=accum)
    out_ref = torch.cat([x.view(-1) for x in out_ref], 0)

    out = [x.clone() for x in xs]
    triton_batch_inplace_add(out, ys, accum=accum)
    out = torch.cat([x.view(-1) for x in out], 0)
    output_check(out_ref, out, 'batch_add')

    if bench:
        ref_bytes = M * N * (x_dtype.itemsize * (2 if accum else 1) + x_dtype.itemsize)
        ref_time = benchmark_func(torch_batch_add, xs, ys, accum=accum,
                                  ref_bytes=ref_bytes)
        benchmark_func(triton_batch_inplace_add, xs, ys, accum=accum,
                       ref_bytes=ref_bytes,
                       ref_time=ref_time)


if __name__ == '__main__':
    test_triton_inplace_add(M=4096, N=4096, accum=False, bench=True)
    test_triton_inplace_add(M=4096, N=3467, accum=False, bench=True)
    test_triton_inplace_add(M=3467, N=3467, accum=False, bench=True)
    test_triton_inplace_add(M=4096, N=4096, accum=True, bench=True)
    test_triton_inplace_add(M=4096, N=3467, accum=True, bench=True)
    test_triton_inplace_add(M=3467, N=3467, accum=True, bench=True)

    test_triton_batch_inplace_add(B=8, M=4096, N=4096, accum=False, bench=True)
    test_triton_batch_inplace_add(B=8, M=4096, N=4097, accum=False, bench=True)
    test_triton_batch_inplace_add(B=8, M=4096, N=4096, accum=True, bench=True)
    test_triton_batch_inplace_add(B=8, M=4096, N=4097, accum=True, bench=True)

