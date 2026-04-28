# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.gemm.fp32_gemm import (
    triton_fp32_gemm,
    triton_fp32_gemm_for_backward,
    triton_fp32_gemm_for_update,
    triton_split_fp32_gemm,
    triton_split_fp32_gemm_for_backward,
    triton_split_fp32_gemm_for_update,
)
from linghe.experimental.gemm import triton_tma_persistent_matmul
from linghe.utils.add import triton_inplace_add
from linghe.utils.transpose import triton_pad_transpose


# the function is used in transformer_engine/pytorch/cpp_extensions/gemm.py
def smooth_gemm(A, B, layout="TN", out=None, accumulate=True, online_transpose=False):
    if layout == "TN":  # forward, y=x@w
        x_q = B._rowwise_data
        x_scale = B._rowwise_scale_inv
        w_q = A._rowwise_data
        w_scale = A._rowwise_scale_inv
        out = torch._scaled_mm(
            x_q,
            w_q.t(),
            scale_a=x_scale.view(-1, 1),
            scale_b=w_scale.view(1, -1),
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        out = out.view(*B.shape[:-1], A.shape[0])
    elif layout == "NN":  # backward, dx=dy@wT
        y_q = B._rowwise_data
        y_scale = B._rowwise_scale_inv
        w_q = A._columnwise_data
        if w_q is None:
            w_q = triton_pad_transpose(
                A._rowwise_data, out=A._columnwise_data, multiple=32
            )
            if not online_transpose:
                A._columnwise_data = w_q
        w_scale = A._columnwise_scale_inv
        out = torch._scaled_mm(
            y_q,
            w_q.t(),
            scale_a=y_scale.view(-1, 1),
            scale_b=w_scale.view(1, -1),
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        out = out.view(*B.shape[:-1], A.shape[1])
    elif layout == "NT":  # update, dw=dyT@dx
        y_q = B._columnwise_data
        y_scale = B._columnwise_scale_inv
        if A._columnwise_data is None:
            x_q = triton_pad_transpose(A._rowwise_data, multiple=32)
        else:
            x_q = A._columnwise_data
        x_scale = A._columnwise_scale_inv
        o = torch._scaled_mm(
            y_q,
            x_q.t(),
            scale_a=y_scale.view(-1, 1),
            scale_b=x_scale.view(1, -1),
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        triton_inplace_add(out, o, accum=accumulate)
        A._columnwise_data = None
    else:
        raise ValueError(f"layout {layout} is not supported")
    return out


def smooth_groued_gemm(
    A, B, out, m_splits, layout="TN", accumulate=True, online_transpose=False
):
    s = 0
    if layout == "TN":  # forward, y=x@w
        for i, m in enumerate(m_splits):
            if m == 0:
                continue

            x_q = B[i]._rowwise_data
            x_scale = B[i]._rowwise_scale_inv
            w_q = A[i]._rowwise_data
            w_scale = A[i]._rowwise_scale_inv
            torch._scaled_mm(
                x_q,
                w_q.t(),
                scale_a=x_scale.view(-1, 1),
                scale_b=w_scale.view(1, -1),
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
                out=out[0][s : s + m],
            )
            s += m
    elif layout == "NN":  # backward, dx=dy@wT
        for i, m in enumerate(m_splits):
            if m == 0:
                continue

            y_q = B[i]._rowwise_data
            y_scale = B[i]._rowwise_scale_inv
            w_q = A[i]._columnwise_data
            if w_q is None:
                w_q = triton_pad_transpose(
                    A[i]._rowwise_data, out=A[i]._columnwise_data, multiple=32
                )
                if not online_transpose:
                    A[i]._columnwise_data = w_q

            w_scale = A[i]._columnwise_scale_inv
            torch._scaled_mm(
                y_q,
                w_q.t(),
                scale_a=y_scale.view(-1, 1),
                scale_b=w_scale.view(1, -1),
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
                out=out[0][s : s + m],
            )
            s += m
    elif layout == "NT":  # update, dw=dyT@dx
        for i, m in enumerate(m_splits):
            if m == 0:
                continue

            y_q = B[i]._columnwise_data
            y_scale = B[i]._columnwise_scale_inv
            if A[i]._columnwise_data is None:
                x_q = triton_pad_transpose(A[i]._rowwise_data, multiple=32)
            else:
                x_q = A[i]._columnwise_data
            x_scale = A[i]._columnwise_scale_inv
            # out is float32
            o = torch._scaled_mm(
                y_q,
                x_q.t(),
                scale_a=y_scale.view(-1, 1),
                scale_b=x_scale.view(1, -1),
                out_dtype=torch.bfloat16,
                use_fast_accum=True,
            )

            triton_inplace_add(out[i], o, accum=accumulate)
            A[i]._columnwise_data = None
            s += m
    else:
        raise ValueError(f"layout {layout} is not supported")
    return out


class Fp32GEMM(torch.autograd.Function):
    """"""

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor, impl: str):
        shape = input.shape
        if len(shape) == 3:
            input = input.view(shape[0] * shape[1], shape[2])
        if impl == "native":
            logits = triton_fp32_gemm(input, weight)
        elif impl == "tma":
            logits = triton_tma_persistent_matmul(input, weight)
        elif impl == "split":
            logits = triton_split_fp32_gemm(input, weight)

        ctx.input_requires_grad = input.requires_grad
        ctx.weight_requires_grad = weight.requires_grad
        ctx.shape = shape
        ctx.impl = impl
        ctx.save_for_backward(input, weight)
        if len(shape) == 3:
            logits = logits.view(shape[0], shape[1], weight.shape[0])
        return logits

    @staticmethod
    def backward(ctx, grad_output):
        grad_shape = grad_output.shape
        if len(grad_shape) == 3:
            grad_output = grad_output.view(grad_shape[0] * grad_shape[1], grad_shape[2])

        input, weight = ctx.saved_tensors

        if ctx.impl == "split":
            dx = triton_split_fp32_gemm_for_backward(grad_output, weight)
        else:
            dx = triton_fp32_gemm_for_backward(grad_output, weight)
        if len(grad_shape) == 3:
            dx = dx.view(*ctx.shape)

        if ctx.impl == "split":
            dw = triton_split_fp32_gemm_for_update(grad_output, input)
        else:
            dw = triton_fp32_gemm_for_update(grad_output, input)

        return dx, dw, None


def fp32_gemm(input: torch.Tensor, weight: torch.Tensor, impl="native"):
    """
    gemm with bf16/fp16 inputs and float32 output,
    currently used in MoE router gemm.
    Args:
        input: bf16/fp16 activation tensor
        weight: bf16/fp16 weight tensor
    Returns:
        output of gemm
    """
    assert impl in ("native", "split", "tma")
    assert input.dtype == weight.dtype, f"{input.dtype=} {weight.dtype=}"
    return Fp32GEMM.apply(input, weight, impl)
