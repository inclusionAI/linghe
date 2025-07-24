


import torch

import time
import os
import random
from flops.utils.benchmark import benchmark_func
from flops.utils.util import *   # noqa: F403
from flops.utils.rearange import *   # noqa: F403



def torch_split_and_cat(x,scales,counts,indices):
    n = len(counts)
    chunks = torch.split(x,counts)
    chunks = [chunks[indices[i]] for i in range(n)]
    output_data = torch.cat(chunks,dim=0)

    chunks = torch.split(scales,counts)
    chunks = [chunks[indices[i]] for i in range(n)]
    output_scale = torch.cat(chunks,dim=0)
    return output_data, output_scale


def test_triton_split_and_cat(M=4096,N=4096):
    # M, N, K = 8192, 10240, 8192  # max qkv
    # M, N, K = 8192, 8192, 8192  # max out
    # M, N, K = 2048, 4096, 8192  # max gate_up
    # M, N, K = 2048, 8192, 2048  # max down
    # M, N, K = 8192, 8192, 8192
    # M, N, K = 2048, 8192, 8192

    # M, N, K = M-1, N-1, K-1

    dtype = torch.bfloat16
    device = 'cuda:0'
    n_repeat = 100

    x = torch.randn(M, N, dtype=dtype, device=device)

    x_q = x.to(torch.float8_e4m3fn)
    x_scales = x.amax(1).float()


    split_size_list = [M//256]*256
    sorted_indices_list = [i//32+i%8*32 for i in range(256)]

    counts = torch.tensor(split_size_list,device=device)
    indices = torch.tensor(sorted_indices_list,device=device)
    chunks = torch.split(x_q.view(torch.float8_e4m3fn), split_size_list)
    scale_chunks = torch.split(x_scales, split_size_list)

    data_ref, scale_ref = torch_split_and_cat(x_q.view(torch.float8_e4m3fn),x_scales,split_size_list,sorted_indices_list)

    data, scale = triton_split_and_cat(x_q,counts,indices,scales=x_scales)

    output_check(data_ref.view(torch.float8_e4m3fn).float(), data.float(), mode='data')
    output_check(scale_ref, scale, mode='scale')

    benchmark_func(torch.split,x_q.view(torch.uint8),split_size_list, n_repeat=n_repeat)
    benchmark_func(torch.cat,chunks,dim=0, n_repeat=n_repeat)
    benchmark_func(torch.split,x_scales,split_size_list, n_repeat=n_repeat)
    benchmark_func(torch.cat,scale_chunks,dim=0, n_repeat=n_repeat)
    benchmark_func(torch_split_and_cat,x_q.view(torch.float8_e4m3fn),x_scales,split_size_list,sorted_indices_list, n_repeat=n_repeat)
    benchmark_func(triton_split_and_cat,x_q,counts,indices,scales=x_scales, n_repeat=n_repeat)


if __name__ == '__main__':
    test_triton_split_and_cat(M=4096,N=4096)