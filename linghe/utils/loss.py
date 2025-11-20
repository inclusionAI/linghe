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
                                          N, B: tl.constexpr,
                                          INPLACE: tl.constexpr):
    pid = tl.program_id(axis=0).to(tl.int64)
    N = N.to(tl.int64)
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
        if INPLACE:
            tl.debug_barrier()
            tl.store(logit_ptr + pid * N + i * B + tl.arange(0, B), grad,
                    mask=i * B + tl.arange(0, B) < N)
        else:
            tl.store(output_grad_ptr + pid * N + i * B + tl.arange(0, B), grad,
                    mask=i * B + tl.arange(0, B) < N)
    tl.debug_barrier()
    if INPLACE:
        target_grad = tl.load(logit_ptr + pid * N + label)
    else:
        target_grad = tl.load(output_grad_ptr + pid * N + label)
    target_grad -= input_grad
    tl.debug_barrier()
    if INPLACE:
        tl.store(logit_ptr + pid * N + label, target_grad)
    else:
        tl.store(output_grad_ptr + pid * N + label, target_grad)


def triton_softmax_cross_entropy_backward(logits, labels, sum_exp, max_logit,
                                          output_grad,
                                          inplace=False):
    """
    backward of softmax cross entropy loss
    Args:
        logits: logit tensor, [bs, dim]
        labels: label tensor, [bs]
        sum_exp:  [bs]
        max_logit: [bs]
        output_grad: gradient, [bs, dim]

    Returns:
        grad of input: [bs, dim]
    """
    M, N = logits.shape
    device = logits.device
    if not inplace:
        dx = torch.empty((M, N), device=device, dtype=logits.dtype)
    else:
        dx = None
    B = 4096
    grid = (M,)
    softmax_cross_entropy_backward_kernel[grid](
        logits,
        labels,
        sum_exp,
        max_logit,
        output_grad,
        dx,
        N,
        B,
        inplace,
        num_stages=3,
        num_warps=8
    )
    if inplace:
        dx = logits
    return dx




@triton.jit
def moe_z_loss_forward_kernel(logit_ptr, loss_ptr, coef,
                             T: tl.constexpr,
                              D: tl.constexpr):
    pid = tl.program_id(axis=0)

    logit = tl.load(logit_ptr + pid * T * D + tl.arange(0, T)[:, None]*D + tl.arange(0, D)).to(
        tl.float32)
    max_logit = tl.max(logit, 1)
    lse = tl.log(tl.sum(tl.exp(logit - max_logit[:, None]), 1)) + max_logit
    loss = coef / T * tl.sum(lse * lse)

    tl.store(loss_ptr + pid, loss)


def triton_moe_z_loss_forward(logits, coef=1e-6):
    """
    compute moe z loss,
    z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1))) * coef
    Args:
        logits: logits tensor
        coef: z loss coef
    Returns:
        z loss
    """
    L, B, D = logits.shape
    device = logits.device
    M = L*B
    T = 4
    assert M % T == 0
    loss = torch.empty((M//T,), device=device, dtype=torch.float32)
    grid = (M//T,)
    moe_z_loss_forward_kernel[grid](
        logits,
        loss,
        coef,
        T,
        D,
        num_stages=3,
        num_warps=1
    )
    return loss.mean()


@triton.jit
def moe_z_loss_backward_kernel(input_grad_ptr, logit_ptr, output_grad_ptr, coef,
                                          T: tl.constexpr, 
                                          D: tl.constexpr):
    pid = tl.program_id(axis=0)
    n_tokens = tl.num_programs(axis=0) * T
    grad = tl.load(input_grad_ptr).to(tl.float32)

    logit = tl.load(logit_ptr + pid * T * D + tl.arange(0, T)[:, None]*D + tl.arange(0, D)[None, :]).to(
        tl.float32)
    max_logit = tl.max(logit, 1, keep_dims=True)
    e = tl.exp(logit - max_logit)
    se = tl.sum(e, 1, keep_dims=True)
    lse = tl.log(se) + max_logit

    grads = 2 * coef / n_tokens * grad * lse * e/se

    tl.store(output_grad_ptr + pid * T * D + tl.arange(0, T)[:, None]*D + tl.arange(0, D), grads)


def triton_moe_z_loss_backward(grads, logits, coef=1e-6):
    """
    backward of moe z loss
    Args:
        grads: grad scalar tensor
        logits: logit tensor, [L, B, dim]
        coef: python scalar
    Returns:
        output_grad: [L, B, dim]
    """
    L, B, D = logits.shape
    device = logits.device
    M = L*B
    T = 4
    assert M % T == 0
    output_grad = torch.empty((L,B,D), device=device, dtype=logits.dtype)
    grid = (M//T,)
    moe_z_loss_backward_kernel[grid](
        grads,
        logits,
        output_grad,
        coef,
        T,
        D,
        num_stages=3,
        num_warps=1
    )
    return output_grad




