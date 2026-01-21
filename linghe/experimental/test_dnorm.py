# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from linghe.utils.norm import triton_rms_norm_forward
from linghe.experimental.dnorm import triton_sp_rms_norm_forward
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check


def torch_rms_gather_forward(x, weight, group):
    M, N = x.shape
    out, _ = triton_rms_norm_forward(x, weight)
    torch_tensor_gather = torch.empty(group.size()*M, N, dtype=x.dtype,device=x.device)
    dist.all_gather_into_tensor(torch_tensor_gather, out)
    return torch_tensor_gather


def test_norm_gather(M=4096, N=2048, group=None, bench=False):
    dtype = torch.bfloat16
    group_size = group.size()
    group_rank = group.rank()

    device_module = torch.get_device_module("cuda")
    device_module.set_device(torch.device(f'cuda:{group_rank}'))

    device = 'cuda'
    dtype = torch.bfloat16

    buffers = symm_mem.empty((M, N), dtype=torch.bfloat16, device=device)
    hdl = symm_mem.rendezvous(buffers, dist.group.WORLD)

    x = torch.randn(M // group_size, N, dtype=dtype, requires_grad=True, device=device) ** 3
    weight = torch.randn(N, dtype=dtype, requires_grad=True, device=device)

    tri_out, _ = triton_sp_rms_norm_forward(x, weight, hdl)
    
    # ### linghe impl + torch all gather ###
    torch_x = x.clone()
    torch_out = torch_rms_gather_forward(torch_x, weight, group)
    
    output_check(torch_out, tri_out, 'norm out')

    if bench:
        benchmark_func(triton_sp_rms_norm_forward, x, weight, hdl)
        benchmark_func(torch_rms_gather_forward, x, weight, group)


if __name__ == "__main__":
    # torchrun --nproc_per_node=4 test_norm_allgather.py
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ['TORCH_NCCL_AVOID_RECORD_STREAMS'] = '1'
    print(f'{world_size=} {local_rank=}')
    dist.init_process_group(backend='nccl', init_method= "env://",
                            world_size=world_size, rank=local_rank,
                            timeout=timedelta(seconds=10))
    group = dist.distributed_c10d._get_default_group()
    # torch.distributed.distributed_c10d._set_pg_timeout(timedelta(seconds=10), dist.group.WORLD)
    test_norm_gather(M=8192, N=8192, group=group, bench=True)
    dist.destroy_process_group()