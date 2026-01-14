# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_cross_entropy_forward_kernel(logit_ptr,
                                         label_ptr,
                                         loss_ptr,
                                         sum_exp_ptr,
                                         max_logit_ptr,
                                         N,
                                         ignore_index,
                                         B: tl.constexpr):
    pid = tl.program_id(axis=0).to(tl.int64)
    label = tl.load(label_ptr + pid)
    if label == ignore_index:
        tl.store(sum_exp_ptr + pid, 0.0)
        tl.store(max_logit_ptr + pid, 0.0)
        tl.store(loss_ptr + pid, 0.0)
        return

    sum_exp = 0.0
    sum_exp = sum_exp.to(tl.float64)
    T = tl.cdiv(N, B)
    max_logit = -1e9
    for i in range(T):
        logit = tl.load(logit_ptr + pid * N + i * B + tl.arange(0, B),
                        mask=i * B + tl.arange(0, B) < N, other=-1e10).to(
            tl.float32)
        latest_max_logit = tl.maximum(max_logit, tl.max(logit))

        sum_exp = sum_exp * tl.exp(max_logit - latest_max_logit) + tl.sum(
            tl.exp(logit - latest_max_logit))
        max_logit = latest_max_logit

    tl.store(sum_exp_ptr + pid, sum_exp)
    tl.store(max_logit_ptr + pid, max_logit)
    target_logit = tl.load(logit_ptr + pid * N + label)
    loss = tl.log(sum_exp) - (target_logit - max_logit)
    tl.store(loss_ptr + pid, loss)


def triton_softmax_cross_entropy_forward(logits, labels, ignore_index=-100):
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
    assert logits.is_contiguous() and labels.is_contiguous()
    loss = torch.empty((M,), device=device, dtype=torch.float32)
    sum_exp = torch.empty((M,), device=device, dtype=torch.float32)
    max_logit = torch.empty((M,), device=device, dtype=torch.float32)
    B = 2048
    grid = (M,)
    softmax_cross_entropy_forward_kernel[grid](
        logits,
        labels,
        loss,
        sum_exp,
        max_logit,
        N,
        ignore_index,
        B,
        num_stages=3,
        num_warps=2
    )
    return loss, sum_exp, max_logit


@triton.jit
def softmax_cross_entropy_backward_kernel(logit_ptr, label_ptr, sum_exp_ptr,
                                          max_logit_ptr,
                                          output_grad_ptr,
                                          input_grad_ptr,
                                          N,
                                          ignore_index,
                                          B: tl.constexpr,
                                          INPLACE: tl.constexpr):
    pid = tl.program_id(axis=0).to(tl.int64)
    T = tl.cdiv(N, B)
    label = tl.load(label_ptr + pid)
    if label == ignore_index:
        for i in range(T):
            grad = tl.zeros((B,), dtype=tl.float32)
            if INPLACE:
                tl.store(logit_ptr + pid * N + i * B + tl.arange(0, B), grad,
                         mask=i * B + tl.arange(0, B) < N)
            else:
                tl.store(input_grad_ptr + pid * N + i * B + tl.arange(0, B),
                         grad,
                         mask=i * B + tl.arange(0, B) < N)
        return

    output_grad = tl.load(output_grad_ptr + pid).to(tl.float32)
    sum_exp = tl.load(sum_exp_ptr + pid)
    max_logit = tl.load(max_logit_ptr + pid)
    coef = output_grad / sum_exp
    target_logit = tl.load(logit_ptr + pid * N + label).to(tl.float32)
    target_grad = (tl.exp(target_logit - max_logit) / sum_exp - 1) * output_grad
    tl.debug_barrier()  # must add barrier here, or it may read stored values
    for i in range(T):
        logit = tl.load(logit_ptr + pid * N + i * B + tl.arange(0, B),
                        mask=i * B + tl.arange(0, B) < N, other=-1e10).to(
            tl.float32)
        grad = tl.exp(logit - max_logit) * coef
        if INPLACE:
            tl.store(logit_ptr + pid * N + i * B + tl.arange(0, B), grad,
                     mask=i * B + tl.arange(0, B) < N)
        else:
            tl.store(input_grad_ptr + pid * N + i * B + tl.arange(0, B), grad,
                     mask=i * B + tl.arange(0, B) < N)
    tl.debug_barrier()  # must add barrier here, or it may execute before loop
    if INPLACE:
        tl.store(logit_ptr + pid * N + label, target_grad)
    else:
        tl.store(input_grad_ptr + pid * N + label, target_grad)


def triton_softmax_cross_entropy_backward(logits, labels, sum_exp, max_logit,
                                          output_grad,
                                          ignore_index=-100,
                                          inplace=False):
    """
    backward of softmax cross entropy loss
    Args:
        logits: logit tensor, [bs, dim]
        labels: label tensor, [bs]
        sum_exp:  [bs]
        max_logit: [bs]
        output_grad: gradient, [bs, dim]
        inplace: whether to reuse logits as gradient

    Returns:
        grad of input: [bs, dim]
    """
    assert output_grad.is_contiguous()
    M, N = logits.shape
    device = logits.device
    if not inplace:
        dx = torch.empty((M, N), device=device, dtype=logits.dtype)
    else:
        dx = None
    B = 2048
    grid = (M,)
    softmax_cross_entropy_backward_kernel[grid](
        logits,
        labels,
        sum_exp,
        max_logit,
        output_grad,
        dx,
        N,
        ignore_index,
        B,
        inplace,
        num_stages=3,
        num_warps=8
    )
    if inplace:
        dx = logits
    return dx


@triton.jit
def parallel_logit_stat_kernel(logit_ptr,
                               label_ptr,
                               sum_exp_ptr,
                               max_logit_ptr,
                               target_logit_ptr,
                               N,
                               ignore_index,
                               group_rank,
                               group_size,
                               B: tl.constexpr):
    pid = tl.program_id(axis=0).to(tl.int64)
    label = tl.load(label_ptr + pid)

    if label == ignore_index:
        tl.store(sum_exp_ptr + pid, 0.0)
        tl.store(max_logit_ptr + pid, 0.0)
        tl.store(target_logit_ptr + pid, 0.0)
        return

    sum_exp = 0.0
    sum_exp = sum_exp.to(tl.float64)
    T = tl.cdiv(N, B)
    max_logit = -1e9
    for i in range(T):
        logit = tl.load(logit_ptr + pid * N + i * B + tl.arange(0, B),
                        mask=i * B + tl.arange(0, B) < N, other=-1e10).to(
            tl.float32)
        latest_max_logit = tl.maximum(max_logit, tl.max(logit))

        sum_exp = sum_exp * tl.exp(max_logit - latest_max_logit) + tl.sum(
            tl.exp(logit - latest_max_logit))
        max_logit = latest_max_logit

    tl.store(sum_exp_ptr + pid, sum_exp)
    tl.store(max_logit_ptr + pid, max_logit)

    if label // N == group_rank:
        target_logit = tl.load(logit_ptr + pid * N + label % N).to(tl.float32)
    else:
        target_logit = float('-inf')
    tl.store(target_logit_ptr + pid, target_logit)


@triton.jit
def parallel_calc_loss_kernel(label_ptr, stats, sum_exp_ptr, max_logit_ptr,
                              loss_ptr,
                              M,
                              N,
                              ignore_index,
                              group_size):
    pid = tl.program_id(axis=0).to(tl.int64)
    label = tl.load(label_ptr + pid)
    if label == ignore_index:
        tl.store(loss_ptr + pid, 0.0)
        tl.store(sum_exp_ptr + pid, 0.0)
        tl.store(max_logit_ptr + pid, 0.0)
        return

    sum_exp = 0.0
    sum_exp = sum_exp.to(tl.float64)
    max_logit = -1e9
    tg = float('-inf')  # target logit
    for i in range(group_size):
        se = tl.load(stats + i * M * 3 + pid)
        ml = tl.load(stats + i * M * 3 + M + pid)
        tg = tl.maximum(tl.load(stats + i * M * 3 + 2 * M + pid), tg)
        latest_max_logit = tl.maximum(max_logit, ml)
        sum_exp = sum_exp * tl.exp(max_logit - latest_max_logit) + se * tl.exp(
            ml - latest_max_logit)
        max_logit = latest_max_logit

    loss = tl.log(sum_exp) - (tg - max_logit)
    tl.store(loss_ptr + pid, loss)
    tl.store(sum_exp_ptr + pid, sum_exp)
    tl.store(max_logit_ptr + pid, max_logit)


"""
TODO1: support distributed loss with pytorch ongoing nvshmem feature
TODO2: optimize performance when vocab size is not multiple of 16
"""


def triton_parallel_softmax_cross_entropy_forward(logits, labels, group,
                                                  ignore_index=-100):
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
    assert logits.is_contiguous() and labels.is_contiguous()
    loss = torch.empty((M,), device=device, dtype=torch.float32)

    group_size = group.size()
    group_rank = group.rank()
    stats = torch.empty((3, M), device=device, dtype=torch.float32)
    sum_exp = stats[0]
    max_logit = stats[1]
    target_logit = stats[2]
    statistic = torch.empty((3 * group_size, M), device=device,
                            dtype=torch.float32)
    B = 2048
    grid = (M,)
    parallel_logit_stat_kernel[grid](
        logits,
        labels,
        sum_exp,
        max_logit,
        target_logit,
        N,
        ignore_index,
        group_rank,
        group_size,
        B,
        num_stages=3,
        num_warps=2
    )
    torch.distributed.all_gather_into_tensor(statistic, stats, group=group)
    parallel_calc_loss_kernel[grid](labels, statistic, sum_exp, max_logit, loss,
                                    M,
                                    N,
                                    ignore_index,
                                    group_size,
                                    num_stages=3,
                                    num_warps=2)

    return loss, sum_exp, max_logit


@triton.jit
def parallel_softmax_cross_entropy_backward_kernel(logit_ptr, label_ptr,
                                                   sum_exp_ptr,
                                                   max_logit_ptr,
                                                   output_grad_ptr,
                                                   input_grad_ptr,
                                                   N,
                                                   ignore_index,
                                                   group_rank,
                                                   group_size,
                                                   B: tl.constexpr,
                                                   INPLACE: tl.constexpr):
    pid = tl.program_id(axis=0).to(tl.int64)
    label = tl.load(label_ptr + pid)
    T = tl.cdiv(N, B)

    if label == ignore_index:
        for i in range(T):
            grad = tl.zeros((B,), dtype=tl.float32)
            if INPLACE:
                tl.store(logit_ptr + pid * N + i * B + tl.arange(0, B), grad,
                         mask=i * B + tl.arange(0, B) < N)
            else:
                tl.store(input_grad_ptr + pid * N + i * B + tl.arange(0, B),
                         grad,
                         mask=i * B + tl.arange(0, B) < N)
        return

    output_grad = tl.load(output_grad_ptr + pid).to(tl.float32)
    sum_exp = tl.load(sum_exp_ptr + pid)
    max_logit = tl.load(max_logit_ptr + pid)
    coef = output_grad / sum_exp
    tl.debug_barrier()  # must add barrier here, or it may read stored values
    for i in range(T):
        logit = tl.load(logit_ptr + pid * N + i * B + tl.arange(0, B),
                        mask=i * B + tl.arange(0, B) < N, other=-1e10).to(
            tl.float32)
        grad = tl.exp(logit - max_logit) * coef
        if INPLACE:
            tl.store(logit_ptr + pid * N + i * B + tl.arange(0, B), grad,
                     mask=i * B + tl.arange(0, B) < N)
        else:
            tl.store(input_grad_ptr + pid * N + i * B + tl.arange(0, B), grad,
                     mask=i * B + tl.arange(0, B) < N)
    tl.debug_barrier()  # must add barrier here, or it may execute before loop

    if label // N == group_rank:
        target_logit = tl.load(logit_ptr + pid * N + label % N).to(tl.float32)
        target_grad = (tl.exp(
            target_logit - max_logit) / sum_exp - 1) * output_grad
        if INPLACE:
            tl.store(logit_ptr + pid * N + label % N, target_grad)
        else:
            tl.store(input_grad_ptr + pid * N + label % N, target_grad)


def triton_parallel_softmax_cross_entropy_backward(logits, labels, sum_exp,
                                                   max_logit,
                                                   output_grad,
                                                   group,
                                                   ignore_index=-100,
                                                   inplace=False):
    """
    backward of softmax cross entropy loss
    Args:
        logits: logit tensor, [bs, dim]
        labels: label tensor, [bs]
        sum_exp:  [bs]
        max_logit: [bs]
        output_grad: gradient, [bs, dim]
        inplace: whether to reuse logits as gradient

    Returns:
        grad of input: [bs, dim]
    """
    assert output_grad.is_contiguous()
    M, N = logits.shape
    device = logits.device
    if not inplace:
        dx = torch.empty((M, N), device=device, dtype=logits.dtype)
    else:
        dx = None

    group_size = group.size()
    group_rank = group.rank()

    B = 2048
    grid = (M,)
    parallel_softmax_cross_entropy_backward_kernel[grid](
        logits,
        labels,
        sum_exp,
        max_logit,
        output_grad,
        dx,
        N,
        ignore_index,
        group_rank,
        group_size,
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

    logit = tl.load(
        logit_ptr + pid * T * D + tl.arange(0, T)[:, None] * D + tl.arange(0,
                                                                           D)).to(
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
    assert logits.is_contiguous()
    shape = logits.shape
    if len(shape) == 3:
        L, B, D = logits.shape
        M = L * B
    else:
        M, D = logits.shape
    device = logits.device
    T = 4
    assert M % T == 0
    loss = torch.empty((M // T,), device=device, dtype=torch.float32)
    grid = (M // T,)
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

    logit = tl.load(
        logit_ptr + pid * T * D + tl.arange(0, T)[:, None] * D + tl.arange(0,
                                                                           D)[
                                                                 None, :]).to(
        tl.float32)
    max_logit = tl.max(logit, 1, keep_dims=True)
    e = tl.exp(logit - max_logit)
    se = tl.sum(e, 1, keep_dims=True)
    lse = tl.log(se) + max_logit

    grads = 2 * coef / n_tokens * grad * lse * e / se

    tl.store(output_grad_ptr + pid * T * D + tl.arange(0, T)[:,
                                             None] * D + tl.arange(0, D), grads)


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
    assert grads.is_contiguous()
    device = logits.device
    shape = logits.shape
    if len(shape) == 3:
        L, B, D = logits.shape
        M = L * B
        output_grad = torch.empty((L, B, D), device=device, dtype=logits.dtype)
    else:
        M, D = logits.shape
        output_grad = torch.empty((M, D), device=device, dtype=logits.dtype)

    T = 4
    assert M % T == 0
    grid = (M // T,)
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
