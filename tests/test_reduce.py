# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import random

import torch

from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.utils.reduce import (triton_abs_max,
                                 triton_batch_count_zero,
                                 triton_norm,
                                 triton_batch_norm)


def torch_sum(xs, ord=2, norm=True):
    if ord == 2:
        output = sum([x.square().sum() for x in xs])
        if norm:
            output = torch.sqrt(output)
        return output
    elif ord == 1:
        return sum([x.abs().sum() for x in xs])
    elif ord == -1:
        return max([x.abs().max() for x in xs])


def torch_count_zero(xs):
    count = torch.tensor([0], dtype=torch.int64, device='cuda')
    for x in xs:
        count += x.numel() - torch.count_nonzero(x)
    return count


def test_triton_abs_max(M=4096, N=4096, bench=False):
    x = 100 * torch.randn(M, 1, N, dtype=torch.bfloat16, device='cuda:0')

    # scales = 1.0/torch.sqrt(torch.maximum(x[:,0].abs().float().amax(0), torch.ones(M,N,dtype=dtype,device=device)) )
    # smooth_scale_ref = torch.exp2(torch.ceil(torch.log2(scales)))
    maxs_ref = x.abs().amax(0).float().view(N)

    maxs = triton_abs_max(x)
    output_check(maxs_ref, maxs, 'abs_max')
    if bench:
        benchmark_func(triton_abs_max, x, n_repeat=100, ref_bytes=M * N * 2)


def test_count_zero(M=4096, N=8192, k=32, bench=False):
    xs = [torch.randn(M, N, dtype=torch.float32, device='cuda:0').to(
        torch.float8_e4m3fn).to(torch.float32) for i in range(k)]

    ref_bytes = sum([x.numel() for x in xs]) * 4

    count_ref = torch_count_zero(xs)
    count = triton_batch_count_zero(xs)
    # print(f'{count_ref=} {count=}')
    assert count_ref.item() - count.item() == 0

    if bench:
        n_repeat = 100
        ref_time = benchmark_func(torch_count_zero, xs, n_repeat=n_repeat,
                                  ref_bytes=ref_bytes)
        benchmark_func(triton_batch_count_zero, xs, n_repeat=n_repeat,
                       ref_bytes=ref_bytes, ref_time=ref_time)


def test_norm(M=4096, N=8192, coef=1.0, bench=False):
    x = torch.randn(M, N, dtype=torch.float32, device='cuda:0') * 1.0

    sum_ref = x.norm(p=2)
    sums = triton_norm(x, ord=2, norm=True, scalar=True)
    output_check(sum_ref, sums, 'l2_norm')

    sum_ref = x.norm(p=1)
    sums = triton_norm(x, ord=1, norm=True, scalar=True)
    output_check(sum_ref, sums, 'l1_norm')

    if bench:
        ref_bytes = M * N * 4
        n_repeat = 100
        ref_time = benchmark_func(lambda x: x.norm(p=2), x, n_repeat=n_repeat,
                                  ref_bytes=ref_bytes)
        benchmark_func(triton_norm, x, ord=2, norm=True, scalar=True,
                       n_repeat=n_repeat,
                       ref_bytes=ref_bytes, ref_time=ref_time)


def test_batch_norm(M=4096, N=8192, k=32, coef=1.0, bench=False):
    bs = [random.randint(1, int(M ** 0.5)) ** 2 for i in range(k)]
    xs = [torch.randn(bs[i], N, dtype=torch.float32, device='cuda:0') * coef for
          i in range(k)]

    sum_ref = torch_sum(xs, ord=2, norm=False)
    sums = triton_batch_norm(xs, ord=2, norm=False)
    output_check(sum_ref, sums, 'l2_norm')

    sum_ref = torch_sum(xs, ord=1, norm=False)
    sums = triton_batch_norm(xs, ord=1, norm=False)
    output_check(sum_ref, sums, 'l1_norm')

    sum_ref = torch_sum(xs, ord=-1, norm=False)
    sums = triton_batch_norm(xs, ord=-1, norm=False)
    output_check(sum_ref, sums, 'inf_norm')

    if bench:
        ref_bytes = sum([x.numel() for x in xs]) * 4
        n_repeat = 100
        ref_time = benchmark_func(torch_sum, xs, n_repeat=n_repeat,
                                  ref_bytes=ref_bytes)
        benchmark_func(triton_batch_norm, xs, n_repeat=n_repeat,
                       ref_bytes=ref_bytes, ref_time=ref_time)


if __name__ == '__main__':
    test_triton_abs_max(M=4096, N=4096, bench=False)
    test_count_zero(M=4096, N=8192, k=32, bench=False)
    test_norm(M=100000, N=8192, bench=False)
    test_batch_norm(M=4096, N=1024, k=16, bench=False)
    test_batch_norm(M=4096, N=1024, k=64, bench=False)
    test_batch_norm(M=4096, N=2048, k=1024, coef=1e12, bench=True)
