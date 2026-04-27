# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import random

import pytest
import torch

from linghe.facade.loss import moe_z_loss, softmax_cross_entropy
from linghe.tools.check import output_check
from linghe.utils.loss import (
    triton_softmax_cross_entropy_forward,
    triton_softmax_cross_entropy_backward,
    triton_moe_z_loss_forward,
    triton_moe_z_loss_backward,
)


def torch_cross_entropy(logits, targets, grad, ignore_index=-100):
    float_logits = logits.to(torch.float32)
    losses = torch.nn.functional.cross_entropy(
        float_logits.view(-1, logits.size()[-1]),
        targets.view(-1),
        reduction="none",
        ignore_index=ignore_index,
    )
    loss = (losses * grad).sum()
    loss.backward()
    return losses.to(torch.float32), logits.grad


def torch_z_loss(logits, coef=1e-6):
    float_logits = logits.float()
    loss = torch.mean(torch.square(torch.logsumexp(float_logits, dim=-1))) * coef
    loss.backward()
    return loss, logits.grad


@pytest.mark.parametrize(
    "M,N,coef,grad_coef,fill,ignore_index",
    [
        (8192, 157184, 1.0, 1.0, False, None),
        (8192, 157184, 1.0, 1e-6, False, None),
        (8192, 157184, 10000.0, 100.0, True, None),
        (8192, 157184, 1.0, 1.0, True, -100),
        (8192, 157184, 1.0, 1.0, True, 0),
        (8192, 157184 - 16, 10000.0, 100.0, True, None),
        (8192, 175175, 1.0, 1.0, False, None),
        (8192, 157184, 0.0, 0.0, False, None),
        (8192, 157184, 0.0, 100.0, False, None),
        (8192, 157184, 1000.0, 0.0, False, None),
        (8192, 157184, 100.0, 100.0, True, None),
        (4096, 157184, 0.1, 1.0, False, None),
    ],
)
def test_triton_softmax_cross_entropy(
    M, N, coef, grad_coef, ignore_index, fill, benchmark, inplace=True
):
    device = "cuda:0"
    dtype = torch.bfloat16
    logits = torch.randn((M, N), dtype=dtype, device=device, requires_grad=False)

    select = True
    if select:
        top_indices = torch.topk(logits, 1)[1].tolist()
        targets = []
        for i, idx in enumerate(top_indices):
            targets.append(random.choice(idx))

        targets = torch.tensor(targets, dtype=torch.long, device=device)
    else:
        targets = torch.randint(0, N, (M,), dtype=torch.long, device=device)

    if ignore_index is not None:
        targets[:10000] = ignore_index

    if fill:
        logits[:, :8192] = -10000
        # logits[:,0] = -100000000
        # logits[:,1:] = 1000
        # logits[:,100:] = 1000
        # targets = 0 * torch.ones((M,), dtype=torch.long, device=device)

    ignore_index = -100 if ignore_index is None else ignore_index
    logits = (logits * coef).detach().clone().requires_grad_()
    output_grad = torch.randn((M,), dtype=torch.float32, device=device) * grad_coef
    loss_ref, grad_ref = torch_cross_entropy(
        logits, targets, output_grad, ignore_index=ignore_index
    )

    loss, sum_exp, max_logit = triton_softmax_cross_entropy_forward(
        logits.detach().clone(), targets, ignore_index=ignore_index
    )
    output_check(loss_ref, loss, name="loss", atol=1e-4, rtol=1e-5)

    grad = triton_softmax_cross_entropy_backward(
        logits.detach().clone(),
        targets,
        sum_exp,
        max_logit,
        output_grad,
        ignore_index=ignore_index,
        inplace=inplace,
    )
    output_check(grad_ref, grad, name="grad", digest=10)

    logits_ = logits.detach().clone().requires_grad_()
    loss = softmax_cross_entropy(
        logits_, targets, ignore_index=ignore_index, inplace=True
    )
    loss.backward(output_grad)
    grad = logits_.grad
    output_check(loss_ref, loss, name="loss", atol=1e-4, rtol=1e-5)
    output_check(grad_ref, grad, name="grad", digest=10)

    logits_clone = logits.detach().clone()
    benchmark(
        torch_cross_entropy,
        logits.requires_grad_(),
        targets,
        output_grad,
        ref_bytes=M * N * 2,
    )
    benchmark(
        triton_softmax_cross_entropy_forward,
        logits,
        targets,
        ignore_index=ignore_index,
        ref_bytes=M * N * 2,
    )
    benchmark(
        triton_softmax_cross_entropy_backward,
        logits_clone,
        targets,
        sum_exp,
        max_logit,
        output_grad,
        ignore_index=ignore_index,
        ref_bytes=M * N * 4,
    )


@pytest.mark.parametrize(
    "L,B,N,coef",
    [
        (4096, 2, 256, 1e-6),
    ],
)
def test_z_loss(L, B, N, coef, benchmark):
    device = "cuda:0"
    logits = torch.randn(
        (L, B, N), dtype=torch.float32, device=device, requires_grad=False
    )
    logits = (logits * 1).detach().clone().requires_grad_()
    input_grad = torch.ones((1,), dtype=torch.float32, device=device)
    loss_ref, grad_ref = torch_z_loss(logits, coef=coef)

    loss = triton_moe_z_loss_forward(logits, coef=coef)
    grad = triton_moe_z_loss_backward(input_grad, logits, coef=coef)
    output_check(loss_ref, loss, name="loss")
    output_check(grad_ref.float(), grad.float(), name="grad")

    loss = moe_z_loss(logits, coef=coef)
    loss.backward(gradient=input_grad[0])
    grad = logits.grad
    output_check(loss_ref, loss, name="loss")
    output_check(grad_ref.float(), grad.float(), name="grad")

    benchmark(torch_z_loss, logits, coef=coef, ref_bytes=L * B * N * 4)
    benchmark(triton_moe_z_loss_forward, logits, coef=coef, ref_bytes=L * B * N * 4)
    benchmark(
        triton_moe_z_loss_backward,
        input_grad,
        logits,
        coef=coef,
        ref_bytes=L * B * N * 8,
    )
