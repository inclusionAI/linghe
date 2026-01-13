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

from linghe.experimental.demb import (triton_tp_embedding_lookup_forward,
                                      triton_tp_embedding_lookup_backward,
                                      triton_sp_embedding_lookup_forward,
                                      triton_sp_embedding_lookup_backward)
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check


def torch_tp_emb(input_ids, weights, group):
    V, D = weights.shape
    group_rank = group.rank()
    ids = input_ids % V
    output = weights[ids]
    mask = torch.logical_and(input_ids >= group_rank * V,
                             input_ids < (group_rank + 1) * V)
    output = torch.where(mask[:, :, None], output, 0.0 * output)
    dist.all_reduce(output, op=dist.ReduceOp.SUM)
    return output


def test_tp_emb(B=1, M=4096, N=157184, D=4096, coef=1.0, grad_coef=1.0,
                group=None, bench=False):
    group_size = group.size()
    group_rank = group.rank()

    device_module = torch.get_device_module("cuda")
    device_module.set_device(torch.device(f'cuda:{group_rank}'))

    device = 'cuda'
    dtype = torch.bfloat16
    buffers = symm_mem.empty(
        (B * M, D),
        dtype=torch.bfloat16,
        device=device,
    )
    hdl = symm_mem.rendezvous(buffers, dist.group.WORLD)

    local_weights = torch.randn((N, D), dtype=dtype, device=device,
                                requires_grad=False)

    local_weights = (local_weights * coef).detach().clone().requires_grad_()

    global_weights = torch.empty((group_size, N, D), dtype=dtype, device=device)
    dist.all_gather_into_tensor(global_weights, local_weights.detach(),
                                group=group)
    global_weights = torch.reshape(global_weights, (
    group_size * N, D)).contiguous().requires_grad_()

    local_ids = torch.randint(0, N * group_size, (B, M), dtype=torch.long,
                              device=device)
    global_ids = torch.empty((group_size, B, M), dtype=torch.long,
                             device=device)
    dist.all_gather_into_tensor(global_ids, local_ids, group=group)
    global_ids = global_ids[0]

    local_grad = torch.randn((B, M, D), dtype=dtype, device=device,
                             requires_grad=False)
    global_grad = torch.empty((group_size, B, M, D), dtype=dtype, device=device)
    dist.all_gather_into_tensor(global_grad, local_grad, group=group)
    global_grad = global_grad.sum(0)

    output_ref = global_weights[global_ids]
    output_ref.backward(global_grad)
    global_weight_grad_ref = global_weights.grad
    local_weight_grad_ref = global_weight_grad_ref[
                            group_rank * N:(group_rank + 1) * N]
    global_weights.grad = None
    # dist_output = torch_tp_emb(global_ids, local_weights, group)
    # output_check(output_ref, dist_output, name=f'output:{group_rank}', atol=1e-4, rtol=1e-5)

    output = triton_tp_embedding_lookup_forward(global_ids,
                                                local_weights,
                                                hdl,
                                                group,
                                                )
    output_check(output_ref, output, name=f'output:{group_rank}', atol=1e-4,
                 rtol=1e-5)

    weight_grad = torch.zeros((N, D), dtype=torch.float32, device=device)
    triton_tp_embedding_lookup_backward(local_grad, global_ids,
                                        weight_grad.data_ptr(), N, hdl, group,
                                        dtype=weight_grad.dtype)
    output_check(local_weight_grad_ref, weight_grad.to(dtype),
                 name=f'grad:{group_rank}', atol=-1e-4, rtol=1e-2)

    if bench:
        benchmark_func(F.embedding, global_ids, global_weights,
                       ref_bytes=M * D * group_size * 2)
        benchmark_func(torch_tp_emb, global_ids, local_weights, group,
                       ref_bytes=M * D * group_size * 2)
        benchmark_func(triton_tp_embedding_lookup_forward, global_ids,
                       local_weights, hdl, group,
                       ref_bytes=M * D * group_size * 2)
        benchmark_func(triton_tp_embedding_lookup_backward, local_grad,
                       global_ids,
                       weight_grad.data_ptr(), N, hdl, group,
                       dtype=weight_grad.dtype,
                       ref_bytes=M * D * group_size * 2)


def test_sp_emb(B=1, M=4096, N=157184, D=4096, coef=1.0, grad_coef=1.0,
                group=None, bench=False):
    group_size = group.size()
    group_rank = group.rank()

    device_module = torch.get_device_module("cuda")
    device_module.set_device(torch.device(f'cuda:{group_rank}'))

    device = 'cuda'
    dtype = torch.bfloat16
    buffers = symm_mem.empty(
        (B * M, D),
        dtype=torch.float32,
        device=device,
    )
    hdl = symm_mem.rendezvous(buffers, dist.group.WORLD)

    local_weights = torch.randn((N, D), dtype=dtype, device=device,
                                requires_grad=False)

    local_weights = (local_weights * coef).detach().clone().requires_grad_()

    global_weights = torch.empty((group_size, N, D), dtype=dtype, device=device)
    dist.all_gather_into_tensor(global_weights, local_weights.detach(),
                                group=group)
    global_weights = torch.reshape(global_weights, (
    group_size * N, D)).contiguous().requires_grad_()

    local_ids = torch.randint(0, N * group_size, (B, M), dtype=torch.long,
                              device=device)
    global_ids = torch.empty((group_size, B, M), dtype=torch.long,
                             device=device)
    dist.all_gather_into_tensor(global_ids, local_ids, group=group)
    global_ids = torch.reshape(global_ids, (group_size * B, M))

    local_grad = torch.randn((B, M, D), dtype=dtype, device=device,
                             requires_grad=False)
    global_grad = torch.empty((group_size, B, M, D), dtype=dtype, device=device)
    dist.all_gather_into_tensor(global_grad, local_grad, group=group)
    global_grad = torch.reshape(global_grad, (group_size * B, M, D))

    output_ref = global_weights[global_ids]
    output_ref.backward(global_grad)
    global_weight_grad_ref = global_weights.grad
    local_weight_grad_ref = global_weight_grad_ref[
                            group_rank * N:(group_rank + 1) * N]
    global_weights.grad = None
    output_ref = output_ref[group_rank * B:(group_rank + 1) * B]
    # dist_output = torch_sp_emb(global_ids, local_weights, group)
    # output_check(output_ref, dist_output, name=f'output:{group_rank}', atol=1e-4, rtol=1e-5)

    output = triton_sp_embedding_lookup_forward(local_ids,
                                                local_weights,
                                                hdl,
                                                group,
                                                )
    output_check(output_ref, output, name=f'output:{group_rank}', atol=1e-4,
                 rtol=1e-5)

    weight_grad = torch.zeros((N, D), dtype=torch.float32, device=device)
    triton_sp_embedding_lookup_backward(local_grad, local_ids,
                                        weight_grad.data_ptr(), N, hdl, group,
                                        dtype=weight_grad.dtype)
    output_check(local_weight_grad_ref, weight_grad.to(dtype),
                 name=f'grad:{group_rank}', atol=0.02, rtol=0.03)

    if bench:
        benchmark_func(F.embedding, global_ids, global_weights,
                       ref_bytes=M * D * group_size * 2)
        # benchmark_func(torch_sp_emb, global_ids, local_weights, group,
        #                ref_bytes=M * D * group_size * 2)
        benchmark_func(triton_sp_embedding_lookup_forward, local_ids,
                       local_weights, hdl, group,
                       ref_bytes=M * D * group_size * 2)
        benchmark_func(triton_sp_embedding_lookup_backward, local_grad,
                       local_ids,
                       weight_grad.data_ptr(), N, hdl, group,
                       dtype=weight_grad.dtype,
                       ref_bytes=M * D * group_size * 2)


if __name__ == '__main__':
    # torchrun --nproc_per_node=2 test_demb.py
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
    # test_tp_emb(M=8192, N=157184, D=4096, coef=1.0, grad_coef=1.0, group=group, bench=True)
    test_sp_emb(M=8192, N=157184, D=4096, coef=1.0, grad_coef=1.0, group=group,
                bench=True)
