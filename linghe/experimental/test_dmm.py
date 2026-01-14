# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import torch.nn.functional as F

from linghe.experimental.dmm import triton_split_tp_gemm
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check


def torch_dist_mm(x, w, group):
    dtype = x.dtype
    output = x @ w.t()
    output = output.float()
    dist.all_reduce(output, op=dist.ReduceOp.SUM)
    return output.to(dtype)


def test_dist_mm(M=4096, N=2048, K=4096, coef=1.0, grad_coef=1.0,
                 group=None, bench=False):
    group_size = group.size()
    group_rank = group.rank()

    device_module = torch.get_device_module("cuda")
    device_module.set_device(torch.device(f'cuda:{group_rank}'))

    device = 'cuda'
    dtype = torch.bfloat16

    buffers = symm_mem.empty((M, N), dtype=torch.float32, device=device)
    hdl = symm_mem.rendezvous(buffers, dist.group.WORLD)

    # hdl = symm_mem.get_symm_mem_workspace(group.group_name, min_size=M * N * 4)
    # buf_list = [
    #     hdl.get_buffer(i, [M, N], torch.float32, 0)
    #     for i in range(hdl.world_size)
    # ]
    # buffer_tuple = tuple(buf_list)

    local_weights = torch.randn((N, K), dtype=dtype, device=device,
                                requires_grad=False)
    local_weights = (local_weights * coef).detach().clone().requires_grad_()
    global_weights = torch.empty((group_size, N, K), dtype=dtype, device=device)
    dist.all_gather_into_tensor(global_weights, local_weights.detach(),
                                group=group)
    global_weights = torch.reshape(torch.permute(global_weights, (1, 0, 2)), (
    N, group_size * K)).contiguous().requires_grad_()

    local_states = torch.randn((M, K), dtype=dtype, device=device)
    global_states = torch.empty((group_size, M, K), dtype=dtype, device=device)
    dist.all_gather_into_tensor(global_states, local_states, group=group)
    global_states = torch.reshape(torch.permute(global_states, (1, 0, 2)),
                                  (M, group_size * K)).contiguous()

    output_ref = global_states @ global_weights.t()

    dist_output = torch_dist_mm(local_states, local_weights, group)
    output_check(output_ref, dist_output, name=f'output:{group_rank}',
                 atol=10.0)

    output = triton_split_tp_gemm(local_states,
                                  local_weights,
                                  hdl,
                                  group,
                                  )
    output_check(output_ref, output, name=f'output:{group_rank}', atol=10.0)

    if bench:
        ref_flops = M * N * K * 2 * group_size
        benchmark_func(F.linear, global_states, global_weights,
                       ref_flops=ref_flops)
        benchmark_func(torch_dist_mm, local_states, local_weights, group,
                       ref_flops=ref_flops)
        benchmark_func(triton_split_tp_gemm, local_states, local_weights, hdl,
                       group,
                       ref_flops=ref_flops)


if __name__ == '__main__':
    # torchrun --nproc_per_node=2 test_dmm.py
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ['TORCH_NCCL_AVOID_RECORD_STREAMS'] = '1'
    print(f'{world_size=} {local_rank=}')
    dist.init_process_group(backend='nccl', init_method="env://",
                            world_size=world_size, rank=local_rank,
                            timeout=timedelta(seconds=10))
    group = dist.distributed_c10d._get_default_group()
    torch.distributed.distributed_c10d._set_pg_timeout(timedelta(seconds=10),
                                                       dist.group.WORLD)
    # test_dist_mm(M=1024, N=8192, K=8192, group=group, bench=True)
    # test_dist_mm(M=8192, N=1024, K=8192, group=group, bench=True)
    # test_dist_mm(M=8192, N=8192, K=1024, group=group, bench=True)
    test_dist_mm(M=8192, N=8192, K=8192, group=group, bench=True)
