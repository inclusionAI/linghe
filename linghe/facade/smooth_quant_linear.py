# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import Optional

import torch

from linghe.quant.smooth import triton_smooth_quant, triton_transpose_smooth_quant
from linghe.utils.reduce import triton_abs_max
from linghe.utils.transpose import triton_transpose_and_pad


class _SmoothQuantLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        smooth_scale: torch.Tensor,
    ):
        ctx.input_requires_grad = input.requires_grad
        ctx.weight_requires_grad = weight.requires_grad
        ctx.bias_requires_grad = bias is not None and bias.requires_grad
        ctx.out_dtype = input.dtype
        ctx.input_shape = input.shape

        round_scale = True
        ctx.round_scale = round_scale

        input = input.view(-1, input.shape[-1])

        x_q, x_scale, x_maxs = triton_smooth_quant(
            input, 1 / smooth_scale, round_scale=round_scale
        )
        w_q, w_scale, w_maxs = triton_smooth_quant(
            weight, smooth_scale, round_scale=round_scale
        )

        output = torch._scaled_mm(
            x_q,
            w_q.t(),
            scale_a=x_scale.view(-1, 1),
            scale_b=w_scale.view(1, -1),
            out_dtype=ctx.out_dtype,
            use_fast_accum=True,
        )

        if bias is not None:
            output += bias

        saved_tensors = [
            x_q if ctx.weight_requires_grad else None,
            x_scale if ctx.weight_requires_grad else None,
            w_q if ctx.input_requires_grad else None,
            w_scale if ctx.input_requires_grad else None,
            (
                smooth_scale
                if ctx.weight_requires_grad or ctx.weight_requires_grad
                else None
            ),
        ]

        ctx.save_for_backward(*saved_tensors)
        out_shape = (*ctx.input_shape[0:-1], -1)
        return output.view(out_shape)

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor):

        x_q, x_s, w_q, w_s, smooth_scale = ctx.saved_tensors

        output_grad = output_grad.view(-1, output_grad.shape[-1])
        round_scale = ctx.round_scale
        y_q, y_scale, y_maxs = triton_smooth_quant(
            output_grad, w_s, reverse=True, round_scale=round_scale
        )

        wt_q = triton_transpose_and_pad(w_q, pad=True)
        dx = torch._scaled_mm(
            y_q,
            wt_q.t(),
            scale_a=y_scale.view(-1, 1),
            scale_b=smooth_scale.view(1, -1),
            out_dtype=ctx.out_dtype,
            use_fast_accum=True,
        )

        yt_q, yt_scale = triton_transpose_smooth_quant(
            output_grad, x_s, reverse=True, round_scale=round_scale
        )

        xt_q = triton_transpose_and_pad(x_q, pad=True)
        dw = torch._scaled_mm(
            yt_q,
            xt_q.t(),
            scale_a=yt_scale.view(-1, 1),
            scale_b=1 / smooth_scale.view(1, -1),
            out_dtype=ctx.out_dtype,
            use_fast_accum=True,
        )

        db = None
        if ctx.bias_requires_grad:
            db = torch.sum(output_grad, dim=0)

        return dx, dw, db, None


class SmoothQuantLinear(torch.nn.Module):
    """
    a naive implementation of smooth quantization linear
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Args:
            in_features: in feature number
            out_features: out feature number
            bias: whether use bias
            device: weight device
            dtype: weight dtype
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.parameter.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        if bias:
            self.bias = torch.nn.parameter.Parameter(
                torch.empty(out_features, device=device, dtype=dtype)
            )
        else:
            self.bias = None

        self.gap_step = 16
        self.smooth_scale = None
        self.smooth_update_step = 0

        self.reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        if self.training:

            if self.smooth_update_step % self.gap_step == 0:
                input_maxs = triton_abs_max(input)
                weight_maxs = triton_abs_max(self.weight)
                self.smooth_scale = torch.sqrt(input_maxs * weight_maxs)

            output = _SmoothQuantLinear.apply(
                input, self.weight, self.bias, self.smooth_scale
            )
            self.smooth_update_step += 1
        else:
            output = input @ self.weight.t()
            if self.bias is not None:
                output = output + self.bias
        return output

    def extra_repr(self) -> str:
        """"""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def reset_parameters(self):
        """"""
        self.weight.data.normal_(mean=0.0, std=0.02)
        if self.bias is not None:
            self.bias.data.zero_()
