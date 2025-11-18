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


def torch_cross_entropy(logits, targets):
    float_logits = logits.float()
    losses = torch.nn.functional.cross_entropy(
        float_logits.view(-1, logits.size()[-1]),
        targets.view(-1),
        reduction='none')
    loss = losses.sum()
    loss.backward()
    return losses, logits.grad


def torch_z_loss(logits, coef=1e-6):
    float_logits = logits.float()
    loss = torch.mean(torch.square(torch.logsumexp(float_logits, dim=-1))) * coef
    loss.backward()
    return loss, logits.grad


def test_triton_softmax_cross_entropy(M=4096, N=157184, coef=1.0, bench=False):
    device = 'cuda:0'
    logits = torch.randn((M, N), dtype=torch.bfloat16, device=device,
                         requires_grad=False)
    logits = (logits * coef).detach().clone().requires_grad_()
    top_indices = torch.topk(logits, 16)[1].tolist()
    targets = []
    for i, idx in enumerate(top_indices):
        targets.append(random.choice(idx))
    targets = torch.tensor(targets, dtype=torch.long, device=device)
    input_grad = torch.ones((M,), dtype=torch.bfloat16, device=device)
    loss_ref, grad_ref = torch_cross_entropy(logits, targets)
    sum_exp_ref = torch.sum(torch.exp(logits.float()), dim=-1)
    max_logit_ref = 0.0 * torch.amax(logits, dim=-1).float()

    loss, sum_exp, max_logit = triton_softmax_cross_entropy_forward(logits,
                                                                    targets)
    output_check(loss_ref, loss, mode='loss')
    # output_check(sum_exp_ref, sum_exp, mode='sum_exp')
    # output_check(max_logit_ref, max_logit, mode='max_logit')

    grad = triton_softmax_cross_entropy_backward(logits, targets, sum_exp,
                                                 max_logit,
                                                 input_grad, output_grad=logits)
    output_check(grad_ref.float(), grad.float(), mode='grad')
    if bench:
        benchmark_func(torch_cross_entropy, logits, targets,
                       ref_bytes=M * N * 6)
        benchmark_func(triton_softmax_cross_entropy_forward, logits, targets,
                       ref_bytes=M * N * 2)
        benchmark_func(triton_softmax_cross_entropy_backward, logits, targets,
                       sum_exp, max_logit, input_grad, ref_bytes=M * N * 4)



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
    test_triton_softmax_cross_entropy(M=8192, N=157184, coef=1.0, bench=False)
    test_triton_softmax_cross_entropy(M=8192, N=157184, coef=10.0, bench=False)
    test_triton_softmax_cross_entropy(M=4096, N=157175, coef=10.0, bench=False)
    test_z_loss(L=4096, B=2, N=256, coef=1e-6, bench=False)
