# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import random

import torch

from linghe.tools.benchmark import benchmark_func
from linghe.tools.util import output_check
from linghe.utils.transpose import (round_up,
                                   triton_batch_transpose,
                                   triton_batch_transpose_and_pad,
                                   triton_transpose,
                                   triton_transpose_and_pad)


# from torch.profiler import profile, record_function, ProfilerActivity

def torch_nd_transpose(x, dim0, dim1):
    return x.transpose(dim0, dim1).contiguous()


def triton_sequence_transpose(xs):
    outputs = []
    for x in xs:
        output = triton_transpose(x)
        outputs.append(output)
    return outputs


def triton_split_transpose(xs, count_list):
    s = 0
    outputs = []
    for i, c in enumerate(count_list):
        x = xs[s:s + c]
        output = triton_transpose_and_pad(x, pad=True)
        outputs.append(output)
        s += c
    return outputs


def test_transpose(M=4096, N=4096, bench=False):
    # M, N, K = 8192, 4096, 13312
    # M, N, K = 4096, 4096, 6144
    # M, N, K = 4096, 4096, 4096

    dtype = torch.bfloat16
    device = 'cuda:0'

    n_repeat = 100

    x = torch.randn(M, N, dtype=dtype, device=device)

    x_q = x.to(torch.float8_e4m3fn)

    ref_output = x_q.t().contiguous()
    opt_output = triton_transpose(x_q)
    output_check(ref_output.float(), opt_output.float(), 'transpose')
    if bench:
        benchmark_func(triton_transpose, x_q, n_repeat=n_repeat,
                       ref_bytes=M * N * 2)


def test_nd_transpose(B=4096, M=4, N=4096, bench=False):
    # M, N, K = 8192, 4096, 13312
    # M, N, K = 4096, 4096, 6144
    # M, N, K = 4096, 4096, 4096

    dtype = torch.bfloat16
    device = 'cuda:0'

    n_repeat = 100

    x = torch.randn(B, M, N, dtype=dtype, device=device)
    t_ref = torch_nd_transpose(x, 0, 1)
    t = triton_transpose(x, inner=True)
    output_check(t_ref, t, '3d_transpose')

    x = torch.randn(B, M, N, dtype=dtype, device=device)[:, :M // 2]
    t_ref = torch_nd_transpose(x, 0, 1)
    t = triton_transpose(x, inner=True)
    output_check(t_ref, t, '3d_transpose_stride')

    x = torch.randn(B, M, N // 128, 128, dtype=dtype, device=device)[:, :M // 2]
    t_ref = torch_nd_transpose(x, 0, 1)
    t = triton_transpose(x, inner=True)
    output_check(t_ref, t, '4d_transpose')

    x = torch.randn(B, M, N, dtype=dtype, device=device)
    t_ref = torch_nd_transpose(x, 1, 2)
    t = triton_transpose(x, inner=False)
    output_check(t_ref, t, '3d_outer_transpose')


    if bench:
        x = torch.randn(B, M, N, dtype=dtype, device=device)
        ref_time = benchmark_func(torch_nd_transpose, x, 0, 1,
                                  n_repeat=n_repeat,
                                  ref_bytes=B * M * N * 4)
        benchmark_func(triton_transpose, x, inner=True, n_repeat=n_repeat,
                       ref_bytes=B * M * N * 4, ref_time=ref_time)
        x = torch.randn(M, B, N, dtype=dtype, device=device)
        ref_time = benchmark_func(torch_nd_transpose, x, 1, 2,
                                  n_repeat=n_repeat,
                                  ref_bytes=B * M * N * 4)
        benchmark_func(triton_transpose, x, inner=False, n_repeat=n_repeat,
                       ref_bytes=B * M * N * 4, ref_time=ref_time)


def test_transpose_and_pad(M=4095, N=4096, bench=False):
    # M, N, K = 8192, 4096, 13312
    # M, N, K = 4096, 4096, 6144
    # M, N, K = 4096, 4096, 4096

    dtype = torch.bfloat16
    device = 'cuda:0'

    n_repeat = 100

    if True:
        x = torch.randn(M, N, dtype=dtype, device=device)
    else:
        x = torch.load('/ossfs/workspace/tmp/vis/backward.bin')['w']
        M, N = x.shape
    P = round_up(M, b=32)
    tail = P - M

    x_q = x.to(torch.float8_e4m3fn)

    ref_output = x_q.t().contiguous()
    opt_output = torch.randn((N, P), dtype=dtype, device=device).to(
        torch.float8_e4m3fn)
    opt_output = triton_transpose_and_pad(x_q, out=opt_output, pad=True)
    output_check(ref_output.float(), opt_output[:, :M].float(),
                 'transpose_and_pad')
    if tail > 0:
        assert opt_output[:, -tail:].float().abs().sum().item() == 0

    if bench:
        benchmark_func(triton_transpose_and_pad, x_q, n_repeat=n_repeat,
                       ref_bytes=M * N * 2)


def test_batch_transpose(M=4096, N=4096, k=32, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    xs = [
        torch.randn((M, N), dtype=dtype, device=device).to(torch.float8_e4m3fn)
        for _ in range(k)]
    xts = triton_batch_transpose(xs)

    x_t_ref = triton_sequence_transpose(xs)
    for i in range(len(xs)):
        output_check(x_t_ref[i].float(), xts[i].float(), f'batch_transpose_{i}')

    if bench:
        n_repeat = 100
        ref_time = benchmark_func(triton_sequence_transpose, xs,
                                  n_repeat=n_repeat, ref_bytes=M * M * 2 * k)
        benchmark_func(triton_batch_transpose, xs, n_repeat=n_repeat,
                       ref_bytes=M * N * 2 * k, ref_time=ref_time)


def test_batch_transpose_and_pad(M=4096, N=4096, k=32, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'
    count_list = [random.randint(1500, 2600) for x in range(k)]
    xs = torch.randn((sum(count_list), N), dtype=dtype, device=device).to(
        torch.float8_e4m3fn)
    x_t = triton_batch_transpose_and_pad(xs, count_list, x_t=None, pad=True)

    x_t_ref = triton_split_transpose(xs, count_list)

    for i in range(len(count_list)):
        output_check(x_t_ref[i].float(), x_t[i].float(),
                     f'batch_transpose_and_pad_{i}')

    if bench:
        n_repeat = 100
        ref_time = benchmark_func(triton_split_transpose, xs, count_list,
                                  n_repeat=n_repeat)
        benchmark_func(triton_batch_transpose_and_pad, xs, count_list, x_t=None,
                       pad=True, n_repeat=n_repeat, ref_time=ref_time)


if __name__ == '__main__':
    # test_transpose(M=4096, N=4096)
    # test_transpose_and_pad(M=4095, N=4096)
    test_nd_transpose(B=4096, M=4, N=2048, bench=True)
    # test_batch_transpose(M=4096,N=4096,k=32)
    # test_batch_transpose_and_pad(M=4096,N=4096,k=32)
