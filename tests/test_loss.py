# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import random

import torch

from linghe.tools.benchmark import benchmark_func
from linghe.tools.util import output_check
from linghe.utils.loss import triton_softmax_cross_entropy_forward, \
    triton_softmax_cross_entropy_backward, triton_moe_z_loss_forward, triton_moe_z_loss_backward


def torch_cross_entropy(logits, targets, grad):
    float_logits = logits.float()
    losses = torch.nn.functional.cross_entropy(
        float_logits.view(-1, logits.size()[-1]),
        targets.view(-1),
        reduction='none')
    loss = (losses*grad).sum()
    loss.backward()
    return losses, logits.grad


def torch_z_loss(logits, coef=1e-6):
    float_logits = logits.float()
    loss = torch.mean(torch.square(torch.logsumexp(float_logits, dim=-1))) * coef
    loss.backward()
    return loss, logits.grad


def test_triton_softmax_cross_entropy(M=4096, N=157184, coef=1.0, inplace=False, bench=False):
    device = 'cuda:0'
    logits = torch.randn((M, N), dtype=torch.bfloat16, device=device,
                         requires_grad=False)
    logits = (logits * coef).detach().clone().requires_grad_()
    top_indices = torch.topk(logits, 16)[1].tolist()
    targets = []
    for i, idx in enumerate(top_indices):
        targets.append(random.choice(idx))
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    output_grad = torch.randn((M,), dtype=torch.bfloat16, device=device)
    loss_ref, grad_ref = torch_cross_entropy(logits, targets, output_grad)

    loss, sum_exp, max_logit = triton_softmax_cross_entropy_forward(logits,
                                                                    targets)
    output_check(loss_ref, loss, mode='loss')

    grad = triton_softmax_cross_entropy_backward(logits, targets, sum_exp,
                                                 max_logit,
                                                 output_grad,
                                                 inplace=inplace)
    output_check(grad_ref.float(), grad.float(), mode='grad')
    if bench:
        benchmark_func(torch_cross_entropy, logits, targets, output_grad,
                       ref_bytes=M * N * 6)
        benchmark_func(triton_softmax_cross_entropy_forward, logits, targets,
                       ref_bytes=M * N * 2)
        benchmark_func(triton_softmax_cross_entropy_backward, logits, targets,
                       sum_exp, max_logit, output_grad, ref_bytes=M * N * 4)



def test_z_loss(L=4096, B=2, N=256, coef=0.001, bench=False):
    device = 'cuda:0'
    logits = torch.randn((L, B, N), dtype=torch.float32, device=device,
                         requires_grad=False)
    logits = (logits * 1).detach().clone().requires_grad_()
    input_grad = torch.ones((1,), dtype=torch.float32, device=device)
    loss_ref, grad_ref = torch_z_loss(logits, coef=coef)

    loss = triton_moe_z_loss_forward(logits, coef=coef)
    output_check(loss_ref, loss, mode='loss')

    grad = triton_moe_z_loss_backward(input_grad, logits, coef=coef)
    output_check(grad_ref.float(), grad.float(), mode='grad')
    if bench:
        benchmark_func(torch_z_loss, logits, coef=coef,
                       ref_bytes=L * B * N * 4)
        benchmark_func(triton_moe_z_loss_forward, logits, coef=coef,
                       ref_bytes=L * B * N * 4)
        benchmark_func(triton_moe_z_loss_backward, input_grad, logits,
                       coef=coef, ref_bytes=L * B * N * 8)


if __name__ == '__main__':
    test_triton_softmax_cross_entropy(M=8192, N=157184, coef=1.0, inplace=False, bench=False)
    test_triton_softmax_cross_entropy(M=8192, N=157184, coef=1e-6, inplace=True, bench=False)
    test_triton_softmax_cross_entropy(M=8192, N=157175, coef=100.0, inplace=False, bench=False)
    test_triton_softmax_cross_entropy(M=4096, N=157175, coef=10.0, inplace=True, bench=False)
    test_z_loss(L=4096, B=2, N=256, coef=1e-6, bench=False)
