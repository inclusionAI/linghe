# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from datetime import timedelta
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt

from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy
from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

from linghe.facade.loss import softmax_cross_entropy
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.utils.loss import (
    triton_softmax_cross_entropy_backward,
    triton_softmax_cross_entropy_forward,
)


def fused_cross_entropy_forward_backward(logits, targets, input_grad, pg):
    logits.grad = None
    losses = fused_vocab_parallel_cross_entropy(logits[None], targets[None], pg)[0]
    losses.backward(input_grad)
    return losses, logits.grad


def te_cross_entropy_forward_backward(logits, targets, input_grad):
    logits.grad = None
    losses = parallel_cross_entropy(logits[None], targets[None])
    losses.backward(input_grad[None])
    return losses, logits.grad


def triton_cross_entropy_forward_backward(logits, targets, input_grad, inplace=True, tp_group=None):
    logits.grad = None
    losses = softmax_cross_entropy(logits, targets, inplace=inplace, tp_group=tp_group)
    losses.backward(input_grad)
    return losses, logits.grad

def fa_cross_entropy_forward_backward(logits, targets, input_grad):
    logits.grad = None
    losses = cross_entropy_loss(logits, targets)[0]
    losses.backward(input_grad)
    return losses, logits.grad

def bench_triton_softmax_cross_entropy(M=4096, N=157184):
    device = 'cuda:0'
    logits = torch.randn((M, N), dtype=torch.bfloat16, device=device)
    logits = logits.detach().clone().requires_grad_()
    targets = (torch.rand((M,), dtype=torch.float32, device=device) * N).to(torch.int64)
    input_grad = 1 / M * torch.randn((M,), dtype=torch.float32, device=device)

    sum_exp = torch.rand((M,), dtype=torch.float32, device=device)
    max_logits = torch.rand((M,), dtype=torch.float32, device=device)

    pg = dist.new_group(ranks=[0], backend='nccl')

    fused_losses, fused_grad = fused_cross_entropy_forward_backward(
        logits.detach().clone().requires_grad_(), targets, input_grad, pg)
    triton_losses, triton_grad = triton_cross_entropy_forward_backward(
        logits.detach().clone().requires_grad_(), targets, input_grad, inplace=False)

    output_check(fused_losses, triton_losses, name='loss')
    output_check(fused_grad, triton_grad, name='grad')

    # 测量时间
    ref_time = benchmark_func(
        fused_cross_entropy_forward_backward,
        logits.detach().clone().requires_grad_(),
        targets, input_grad, pg, n_repeat=100, ref_bytes=M * N * 6
    )
    te_time = benchmark_func(
        te_cross_entropy_forward_backward,
        logits.detach().clone().requires_grad_(), targets, input_grad,
        n_repeat=100, ref_bytes=M * N * 6, ref_time=ref_time
    )
    fa_time = benchmark_func(
        fa_cross_entropy_forward_backward,
        logits.detach().clone().requires_grad_(), targets, input_grad,
        n_repeat=100, ref_bytes=M * N * 6, ref_time=ref_time
    )
    
    triton_time = benchmark_func(
        triton_cross_entropy_forward_backward,
        logits.detach().clone().requires_grad_(),
        targets, input_grad, inplace=False,
        n_repeat=100, ref_bytes=M * N * 6, ref_time=ref_time
    )

    return ref_time, te_time, triton_time, fa_time


if __name__ == '__main__':
    init_method = "env://"
    dist.init_process_group(
        backend='nccl', init_method=init_method,
        world_size=1, rank=0,
        timeout=timedelta(seconds=30)
    )

    test_cases = [
        (4096, 157184),
        (8192, 157184),
        # (8192, 128),
        # (2048, 4096)
    ]

    results = []
    for M, N in test_cases:
        print(f"Running benchmark for M={M}, N={N} ...")
        fused_t, te_t, triton_t, fa_t = bench_triton_softmax_cross_entropy(M=M, N=N)
        results.append((M, N, fused_t, te_t, triton_t, fa_t))

    labels = [f"M={M}\nN={N}" for M, N, *_ in results]
    fused_times = [r[2] for r in results]
    te_times = [r[3] for r in results]
    triton_times = [r[4] for r in results]
    fa_times = [r[5] for r in results]

    x = torch.arange(len(test_cases))
    width = 0.20 

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5*width, fused_times, width, label='Fused CE', color='#1f77b4')
    ax.bar(x - 0.5*width, te_times, width, label='TE CE', color='#ff7f0e')
    ax.bar(x + 0.5*width, triton_times, width, label='Triton CE', color='#2ca02c')
    ax.bar(x + 1.5*width, fa_times, width, label='FlashAttention CE', color='#d62728')

    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Test Cases')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('Cross Entropy Benchmark')
    ax.legend()
    plt.tight_layout()
    plt.savefig('cross_entropy_benchmark.png', dpi=300)
    plt.show()