# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.utils.loss import (
    triton_softmax_cross_entropy_forward,
    triton_softmax_cross_entropy_backward,
    triton_parallel_softmax_cross_entropy_forward,
    triton_parallel_softmax_cross_entropy_backward,
    triton_moe_z_loss_forward,
    triton_moe_z_loss_backward,
)


class SoftmaxCrossEntropyFunction(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, logits, labels, ignore_index=-100, inplace=False, tp_group=None):
        shape = logits.shape
        logits_view = logits.view(-1, shape[-1]) if len(shape) == 3 else logits
        parallel = tp_group is not None and tp_group.size() > 1
        if parallel:
            loss, sum_exp, max_logit = triton_parallel_softmax_cross_entropy_forward(
                logits, labels, tp_group, ignore_index=ignore_index
            )
        else:
            loss, sum_exp, max_logit = triton_softmax_cross_entropy_forward(
                logits_view, labels, ignore_index=ignore_index
            )
        ctx.save_for_backward(logits, labels, sum_exp, max_logit)
        ctx.ignore_index = ignore_index
        ctx.inplace = inplace
        ctx.shape = shape
        ctx.parallel = parallel
        if len(shape) == 3:
            loss = loss.view(shape[0], shape[1])
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        logits, labels, sum_exp, max_logit = ctx.saved_tensors
        shape = ctx.shape
        if len(shape) == 3:
            logits = logits.view(-1, shape[-1])
            grad_output = torch.reshape(grad_output, (-1,))
        if ctx.parallel:
            grad = triton_parallel_softmax_cross_entropy_backward(
                logits,
                labels,
                sum_exp,
                max_logit,
                grad_output,
                ctx.tp_group,
                ignore_index=ctx.ignore_index,
                inplace=ctx.inplace,
            )
        else:

            grad = triton_softmax_cross_entropy_backward(
                logits,
                labels,
                sum_exp,
                max_logit,
                grad_output,
                ignore_index=ctx.ignore_index,
                inplace=ctx.inplace,
            )
        if len(shape) == 3:
            grad = grad.view(shape)
        return grad, None, None, None, None


def softmax_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    inplace: bool = False,
    tp_group=None,
):
    """
    softmax cross entropy
    Args:
        logits: logits tensor, shape [...,dim]
        labels: labels tensor, shape [...]
        inplace: update gradient in the `logits` tensor if True
    Returns:
        a tensor of per token loss
    """
    assert logits.is_contiguous()
    assert labels.is_contiguous()
    return SoftmaxCrossEntropyFunction.apply(
        logits, labels, ignore_index, inplace, tp_group
    )


class GradScalingFunction(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, x, coef=0.2):
        ctx.coef = coef
        return x

    @staticmethod
    def backward(ctx, grad_output):
        shape = grad_output.shape
        assert len(shape) == 2
        bs, length = grad_output.shape
        array = length - torch.arange(0, length, device=grad_output.device)
        scale = 1 / torch.pow(array.float(), ctx.coef)
        grad = grad_output * scale
        return grad, None


class MoeZLossFunction(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, logits, coef):
        loss = triton_moe_z_loss_forward(logits, coef=coef)
        ctx.save_for_backward(
            logits,
        )
        ctx.coef = coef
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (logits,) = ctx.saved_tensors
        grad = triton_moe_z_loss_backward(grad_output, logits, coef=ctx.coef)
        return grad, None


def moe_z_loss(logits: torch.Tensor, coef: float = 1e-3):
    """
    softmax cross entropy
    Args:
        logits: logits tensor, shape [...,dim]
        coef: z loss coef
    Returns:
        z loss
    """
    assert logits.is_contiguous()
    return MoeZLossFunction.apply(logits, coef)
