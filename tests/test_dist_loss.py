# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import os
import random
from datetime import timedelta

import torch
import torch.distributed as dist

from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.utils.loss import (
    triton_parallel_softmax_cross_entropy_forward,
    triton_parallel_softmax_cross_entropy_backward,
    triton_softmax_cross_entropy_forward,
)

# from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy


def torch_cross_entropy(logits, targets, ignore_index=-100, reduction="none"):
    float_logits = logits.to(torch.float32)
    losses = torch.nn.functional.cross_entropy(
        float_logits.view(-1, logits.size()[-1]),
        targets.view(-1),
        reduction=reduction,
        ignore_index=ignore_index,
    )
    return losses


def test_triton_softmax_cross_entropy(
    M=4096,
    N=157184,
    coef=1.0,
    grad_coef=1.0,
    ignore_index=None,
    fill=False,
    inplace=False,
    group=None,
    bench=False,
):
    group_size = group.size()
    group_rank = group.rank()

    device_module = torch.get_device_module("cuda")
    device_module.set_device(torch.device(f"cuda:{group_rank}"))

    device = "cuda"
    dtype = torch.bfloat16
    local_logits = torch.randn((M, N), dtype=dtype, device=device, requires_grad=False)

    select = True
    if select:
        top_indices = torch.topk(local_logits, 1)[1].tolist()
        targets = []
        for i, idx in enumerate(top_indices):
            targets.append(random.choice(idx) * group_size)

        targets = torch.tensor(targets, dtype=torch.long, device=device)
    else:
        targets = torch.randint(
            0, N * group_size, (M,), dtype=torch.long, device=device
        )

    if ignore_index is not None:
        targets[:10] = ignore_index

    if fill:
        local_logits[:, :8192] = -10000

    ignore_index = -100 if ignore_index is None else ignore_index
    local_logits = (local_logits * coef).detach().clone().requires_grad_()

    global_logits = torch.empty((group_size, M, N), dtype=dtype, device=device)
    dist.all_gather_into_tensor(global_logits, local_logits.detach(), group=group)
    global_logits = (
        torch.reshape(torch.permute(global_logits, (1, 0, 2)), (M, group_size * N))
        .contiguous()
        .requires_grad_()
    )

    global_targets = torch.empty((group_size, M), dtype=torch.long, device=device)
    dist.all_gather_into_tensor(global_targets, targets, group=group)
    global_targets = global_targets[0]

    local_output_grad = (
        torch.randn((M,), dtype=torch.float32, device=device) * grad_coef
    )
    global_output_grad = torch.empty(
        (group_size, M), dtype=torch.float32, device=device
    )
    dist.all_gather_into_tensor(global_output_grad, local_output_grad, group=group)
    global_output_grad = global_output_grad[0]

    loss_ref = torch_cross_entropy(
        global_logits, global_targets, ignore_index=ignore_index, reduction="none"
    )
    loss_ref.backward(global_output_grad, retain_graph=True)
    grad_ref = global_logits.grad
    global_logits.grad = None

    loss_sa, sum_exp_sa, max_logit_sa = triton_softmax_cross_entropy_forward(
        global_logits.detach().clone(), global_targets, ignore_index=ignore_index
    )

    loss, sum_exp, max_logit = triton_parallel_softmax_cross_entropy_forward(
        local_logits.detach().clone(), global_targets, group, ignore_index=ignore_index
    )
    output_check(loss_ref, loss, name=f"ref_loss:{group_rank}", atol=1e-4, rtol=1e-5)

    grad = triton_parallel_softmax_cross_entropy_backward(
        local_logits.detach().clone(),
        global_targets,
        sum_exp,
        max_logit,
        global_output_grad,
        group,
        ignore_index=ignore_index,
        inplace=inplace,
    )

    output_check(
        grad_ref[:, group_rank * N : (group_rank + 1) * N],
        grad,
        name=f"grad:{group_rank}",
        digest=10,
    )

    # loss_native = fused_vocab_parallel_cross_entropy(local_logits, global_targets, group)
    # grad_native = loss_native.backward(global_output_grad)
    # grad_native = local_logits.grad
    # local_logits.grad = None
    # output_check(loss_ref, loss_native, name=f'native_loss:{group_rank}', atol=1e-4, rtol=1e-5)
    # output_check(grad_ref[:, group_rank*N:(group_rank+1)*N], grad_native, name=f'native_grad:{group_rank}', atol=1e-4, rtol=1e-5)

    if bench:
        benchmark_func(
            torch_cross_entropy,
            global_logits.requires_grad_(),
            global_targets,
            ref_bytes=M * N * 2,
        )
        benchmark_func(
            triton_parallel_softmax_cross_entropy_forward,
            local_logits,
            global_targets,
            group,
            ignore_index=ignore_index,
            ref_bytes=M * N * 2,
        )
        benchmark_func(
            loss_ref.backward,
            global_output_grad,
            retain_graph=True,
            ref_bytes=M * N * 4,
        )
        benchmark_func(
            triton_parallel_softmax_cross_entropy_backward,
            local_logits.detach().clone(),
            global_targets,
            sum_exp,
            max_logit,
            global_output_grad,
            group,
            ignore_index=ignore_index,
            inplace=True,
            ref_bytes=M * N * 4,
        )


if __name__ == "__main__":
    # torchrun --nproc_per_node=2 test_dist_loss.py
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"{world_size=} {local_rank=}")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=local_rank,
        timeout=timedelta(seconds=30),
    )
    pg = dist.distributed_c10d._get_default_group()
    test_triton_softmax_cross_entropy(
        M=8192, N=157184, coef=1.0, grad_coef=1.0, inplace=False, group=pg, bench=False
    )
    test_triton_softmax_cross_entropy(
        M=8192,
        N=157184,
        coef=1.0,
        grad_coef=1.0,
        ignore_index=-100,
        inplace=False,
        group=pg,
        bench=False,
    )
    test_triton_softmax_cross_entropy(
        M=8192,
        N=157184,
        coef=1.0,
        grad_coef=1.0,
        fill=True,
        inplace=False,
        group=pg,
        bench=False,
    )
    test_triton_softmax_cross_entropy(
        M=8192,
        N=157184,
        coef=1.0,
        grad_coef=1.0,
        fill=True,
        inplace=True,
        group=pg,
        bench=False,
    )
