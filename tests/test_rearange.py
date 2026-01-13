# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.utils.rearange import triton_sort_chunks_by_index


def torch_sort_chunks_by_index(x, scales, counts, indices):
    n = len(counts)
    chunks = torch.split(x, counts)
    chunks = [chunks[indices[i]] for i in range(n)]
    output_data = torch.cat(chunks, dim=0)

    chunks = torch.split(scales, counts)
    chunks = [chunks[indices[i]] for i in range(n)]
    output_scale = torch.cat(chunks, dim=0)
    return output_data, output_scale


def test_sort_chunks_by_index(M=4096, N=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'
    n_repeat = 100

    x = torch.randn(M, N, dtype=dtype, device=device)

    x_q = x.to(torch.float8_e4m3fn)
    x_scales = x.amax(1).float()

    split_size_list = [M // 256] * 256
    sorted_indices_list = [i // 32 + i % 8 * 32 for i in range(256)]

    counts = torch.tensor(split_size_list, device=device)
    indices = torch.tensor(sorted_indices_list, device=device)
    chunks = torch.split(x_q.view(torch.float8_e4m3fn), split_size_list)
    scale_chunks = torch.split(x_scales, split_size_list)

    data_ref, scale_ref = torch_sort_chunks_by_index(
        x_q.view(torch.float8_e4m3fn),
        x_scales, split_size_list,
        sorted_indices_list)

    data, scale = triton_sort_chunks_by_index(x_q, counts, indices,
                                              scales=x_scales)

    output_check(data_ref.view(torch.float8_e4m3fn), data,
                 name='data')
    output_check(scale_ref, scale, name='scale')

    if bench:
        benchmark_func(torch.split, x_q.view(torch.uint8), split_size_list,
                       n_repeat=n_repeat)
        benchmark_func(torch.cat, chunks, dim=0, n_repeat=n_repeat)
        benchmark_func(torch.split, x_scales, split_size_list,
                       n_repeat=n_repeat)
        benchmark_func(torch.cat, scale_chunks, dim=0, n_repeat=n_repeat)
        benchmark_func(torch_sort_chunks_by_index,
                       x_q.view(torch.float8_e4m3fn),
                       x_scales, split_size_list, sorted_indices_list,
                       n_repeat=n_repeat)
        benchmark_func(triton_sort_chunks_by_index, x_q, counts, indices,
                       scales=x_scales, n_repeat=n_repeat)


if __name__ == '__main__':
    test_sort_chunks_by_index(M=4096, N=4096)
