# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from datetime import timedelta

import torch
import torch.distributed as dist
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

from linghe.facade.loss import softmax_cross_entropy
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.utils.loss import (
    triton_softmax_cross_entropy_backward,
    triton_softmax_cross_entropy_forward,
)


def fused_cross_entropy_forward_backward(logits, targets, input_grad, pg):
    losses = fused_vocab_parallel_cross_entropy(logits[None], targets[None], pg)[0]
    losses.backward(input_grad)
    return losses, logits.grad


def te_cross_entropy_forward_backward(logits, targets, input_grad):
    losses = parallel_cross_entropy(logits[None], targets[None])
    losses.backward(input_grad[None])
    return losses, logits.grad


def triton_cross_entropy_forward_backward(logits, targets, input_grad, inplace=True):
    # losses, sum_exp, max_logits = triton_softmax_cross_entropy_forward(logits,
    #                                                                    targets)
    # output_grad = triton_softmax_cross_entropy_backward(logits, targets,
    #                                                     sum_exp, max_logits,
    #                                                     input_grad,
    #                                                     inplace=inplace)
    # return losses, output_grad
    losses = softmax_cross_entropy(logits, targets, inplace=inplace)
    losses.backward(input_grad)
    return losses, logits.grad


def bench_triton_softmax_cross_entropy(M=4096, N=157184):
    device = "cuda:0"
    logits = torch.randn((M, N), dtype=torch.bfloat16, device=device)
    logits = logits.detach().clone().requires_grad_()
    targets = (torch.rand((M,), dtype=torch.float32, device=device) * N).to(torch.int64)
    # targets = torch.topk(logits, 4)[1][:, 3].contiguous()
    input_grad = 1 / M * torch.randn((M,), dtype=torch.float32, device=device)

    sum_exp = torch.rand((M,), dtype=torch.float32, device=device)
    max_logits = torch.rand((M,), dtype=torch.float32, device=device)

    pg = dist.new_group(ranks=[0], backend="nccl")
    fused_losses, fused_grad = fused_cross_entropy_forward_backward(
        logits.detach().clone().requires_grad_(), targets, input_grad, pg
    )
    te_losses, te_grad = te_cross_entropy_forward_backward(
        logits.detach().clone().requires_grad_(), targets, input_grad
    )
    triton_losses, triton_grad = triton_cross_entropy_forward_backward(
        logits.detach().clone().requires_grad_(), targets, input_grad, inplace=False
    )
    output_check(fused_losses, triton_losses)
    output_check(fused_grad, triton_grad)

    ref_time = benchmark_func(
        fused_cross_entropy_forward_backward,
        logits.detach().clone().requires_grad_(),
        targets,
        input_grad,
        pg,
        n_repeat=1,
        ref_bytes=M * N * 6,
    )
    benchmark_func(
        te_cross_entropy_forward_backward,
        logits.detach().clone().requires_grad_(),
        targets,
        input_grad,
        n_repeat=1,
        ref_bytes=M * N * 6,
        ref_time=ref_time,
    )
    benchmark_func(
        triton_cross_entropy_forward_backward,
        logits.detach().clone().requires_grad_(),
        targets,
        input_grad,
        inplace=False,
        n_repeat=1,
        ref_bytes=M * N * 6,
        ref_time=ref_time,
    )
    benchmark_func(
        triton_softmax_cross_entropy_forward,
        logits,
        targets,
        ref_bytes=M * N * 2,
        ref_time=ref_time,
    )
    benchmark_func(
        triton_softmax_cross_entropy_backward,
        logits,
        targets,
        sum_exp,
        max_logits,
        input_grad,
        ref_bytes=M * N * 4,
        ref_time=ref_time,
    )


if __name__ == "__main__":
    # torchrun bench_loss.py
    init_method = "env://"
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        world_size=1,
        rank=0,
        timeout=timedelta(seconds=30),
    )
    bench_triton_softmax_cross_entropy(M=4096, N=157184)
    bench_triton_softmax_cross_entropy(M=8192, N=157184)
    bench_triton_softmax_cross_entropy(M=8192, N=128)
