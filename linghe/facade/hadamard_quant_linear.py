# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math
from typing import Optional

import torch

from linghe.quant.hadamard import triton_hadamard_quant


class _HadamardQuantLinear(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        hadamard_matrix: torch.Tensor,
    ):
        ctx.input_requires_grad = input.requires_grad
        ctx.weight_requires_grad = weight.requires_grad
        ctx.bias_requires_grad = bias is not None and bias.requires_grad

        ctx.out_dtype = input.dtype
        ctx.input_shape = input.shape
        input = input.view(-1, input.shape[-1])

        x_q, x_scale, xt_q, xt_scale = triton_hadamard_quant(input, hadamard_matrix)
        w_q, w_scale, wt_q, wt_scale = triton_hadamard_quant(weight, hadamard_matrix)

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
            xt_q if ctx.weight_requires_grad else None,
            xt_scale if ctx.weight_requires_grad else None,
            wt_q if ctx.input_requires_grad else None,
            wt_scale if ctx.input_requires_grad else None,
            (
                hadamard_matrix
                if ctx.weight_requires_grad or ctx.weight_requires_grad
                else None
            ),
        ]

        ctx.save_for_backward(*saved_tensors)
        out_shape = (*ctx.input_shape[0:-1], -1)
        return output.view(out_shape)

    @staticmethod
    def backward(
        ctx,
        output_grad: torch.Tensor,
    ):
        xt_q, xt_scale, wt_q, wt_scale, hadamard_matrix = ctx.saved_tensors

        output_grad = output_grad.view(-1, output_grad.shape[-1])

        y_q, y_scale, yt_q, yt_scale = triton_hadamard_quant(
            output_grad, hadamard_matrix
        )

        dx = torch._scaled_mm(
            y_q,
            wt_q.t(),
            scale_a=y_scale.view(-1, 1),
            scale_b=wt_scale.view(1, -1),
            out_dtype=ctx.out_dtype,
            use_fast_accum=True,
        )

        dx = dx.view(ctx.input_shape)

        dw = torch._scaled_mm(
            yt_q,
            xt_q.t(),
            scale_a=yt_scale.view(-1, 1),
            scale_b=xt_scale.view(1, -1),
            out_dtype=ctx.out_dtype,
            use_fast_accum=True,
        )

        db = None
        if ctx.bias_requires_grad:
            db = torch.sum(output_grad, dim=0)

        return dx, dw, db, None


class HadamardQuantLinear(torch.nn.Module):
    """
    a naive implementation of hadamard transformation and quantization
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

        size = 32 if "H20" in torch.cuda.get_device_properties(0).name else 64
        data = self._hadamard_matrix(size, device=device, dtype=dtype, norm=True)
        self.hadamard_matrix = torch.nn.parameter.Parameter(data, requires_grad=False)
        self.reset_parameters()

    def _hadamard_matrix(self, size, device=None, dtype=None, norm=False):
        assert 2 ** int(math.log2(size)) == size
        m2 = torch.tensor([[1, 1], [1, -1]], device=device, dtype=torch.float32)
        m = m2
        for _ in range(int(math.log2(size)) - 1):
            m = torch.kron(m, m2)
        if norm:
            m = m / size**0.5
        if dtype is not None:
            m = m.to(dtype)
        return m

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        if self.training:
            return _HadamardQuantLinear.apply(
                input, self.weight, self.bias, self.hadamard_matrix
            )
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
