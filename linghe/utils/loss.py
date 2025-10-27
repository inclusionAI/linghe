# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl

@triton.jit
def softmax_cross_entropy_forward_kernel(logit_ptr, label_ptr, loss_ptr,
                                         sum_exp_ptr, max_logit_ptr, N,
                                         B: tl.constexpr):
    pid = tl.program_id(axis=0).to(tl.int64)
    label = tl.load(label_ptr + pid)
    sum_exp = 0.0
    T = tl.cdiv(N, B)
    max_logit = -1e30
    for i in range(T):
        logit = tl.load(logit_ptr + pid * N + i * B + tl.arange(0, B),
                        mask=i * B + tl.arange(0, B) < N, other=-1e30).to(
            tl.float32)
        max_logit = tl.maximum(max_logit, tl.max(logit))
        sum_exp += tl.sum(tl.exp(logit))

    retry = sum_exp > 3.389e38
    # triton 3.2.0 will raise pass error
    # max_logit = tl.where(retry, max_logit, 0.0)
    retry_sum_exp = 0.0
    if retry:
        for i in range(T):
            logit = tl.load(logit_ptr + pid * N + i * B + tl.arange(0, B),
                            mask=i * B + tl.arange(0, B) < N, other=-1e30).to(
                tl.float32)
            retry_sum_exp += tl.sum(tl.exp(logit - max_logit))
    else:
        max_logit = 0.0
    sum_exp = tl.where(retry, retry_sum_exp, sum_exp)
    tl.store(sum_exp_ptr + pid, sum_exp)
    target_logit = tl.load(logit_ptr + pid * N + label)
    loss = tl.log(sum_exp) - (target_logit - max_logit)
    tl.store(loss_ptr + pid, loss)
    tl.store(max_logit_ptr + pid, max_logit)


"""
TODO: support distributed loss with pytorch ongoing nvshmem feature
"""
def triton_softmax_cross_entropy_forward(logits, labels):
    """
    compute token-wise softmax cross entropy loss
    Args:
        logits: logits tensor
        labels: labels tensor

    Returns:
        loss of each token
    """
    M, N = logits.shape
    device = logits.device
    loss = torch.empty((M,), device=device, dtype=torch.float32)
    sum_exp = torch.empty((M,), device=device, dtype=torch.float32)
    max_logit = torch.empty((M,), device=device, dtype=torch.float32)
    B = 4096
    grid = (M,)
    softmax_cross_entropy_forward_kernel[grid](
        logits,
        labels,
        loss,
        sum_exp,
        max_logit,
        N,
        B,
        num_stages=3,
        num_warps=8
    )
    return loss, sum_exp, max_logit


@triton.jit
def softmax_cross_entropy_backward_kernel(logit_ptr, label_ptr, sum_exp_ptr,
                                          max_logit_ptr,
                                          input_grad_ptr, output_grad_ptr,
                                          N, B: tl.constexpr):
    pid = tl.program_id(axis=0).to(tl.int64)
    label = tl.load(label_ptr + pid)
    input_grad = tl.load(input_grad_ptr + pid).to(tl.float32)
    sum_exp = tl.load(sum_exp_ptr + pid)
    max_logit = tl.load(max_logit_ptr + pid)
    coef = input_grad / sum_exp
    T = tl.cdiv(N, B)
    for i in range(T):
        logit = tl.load(logit_ptr + pid * N + i * B + tl.arange(0, B),
                        mask=i * B + tl.arange(0, B) < N, other=-1e30).to(
            tl.float32)
        grad = tl.exp(logit - max_logit) * coef
        tl.store(output_grad_ptr + pid * N + i * B + tl.arange(0, B), grad,
                 mask=i * B + tl.arange(0, B) < N)
    tl.debug_barrier()
    target_grad = tl.load(output_grad_ptr + pid * N + label)
    target_grad -= input_grad
    tl.store(output_grad_ptr + pid * N + label, target_grad)


def triton_softmax_cross_entropy_backward(logits, labels, sum_exp, max_logit,
                                          input_grad,
                                          output_grad=None):
    """
    backward of softmax cross entropy loss
    Args:
        logits: logit tensor, [bs, dim]
        labels: label tensor, [bs]
        sum_exp:  [bs]
        max_logit: [bs]
        input_grad: gradient, [bs, dim]

    Returns:
        output_grad: [bs, dim]
    """
    M, N = logits.shape
    device = logits.device
    if output_grad is None:
        output_grad = torch.empty((M, N), device=device, dtype=logits.dtype)
    B = 4096
    grid = (M,)
    softmax_cross_entropy_backward_kernel[grid](
        logits,
        labels,
        sum_exp,
        max_logit,
        input_grad,
        output_grad,
        N,
        B,
        num_stages=3,
        num_warps=8
    )
    return output_grad
