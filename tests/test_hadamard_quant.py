# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.facade.hadamard_quant_linear import HadamardQuantLinear
from linghe.quant.hadamard import triton_hadamard_quant
from linghe.tools.check import output_check
from linghe.tools.util import (
    make_hadamard_matrix,
    torch_hadamard_transform,
    torch_row_quant,
)


# apply hadamard transformation and quantization for x
def torch_hadamard_quant(x, hm, round_scale=False):
    dtype = x.dtype
    x = x.float()
    hm = hm.float()
    xh = torch_hadamard_transform(x, hm, side="right")
    q, s = torch_row_quant(xh, round_scale=round_scale)
    xht = torch_hadamard_transform(x.t().contiguous(), hm, side="right")
    qt, st = torch_row_quant(xht, round_scale=round_scale)

    return xh.to(dtype), xht.to(dtype), q, s, qt, st


def test_hadamard_quant(M=8192, N=1024, K=2048, B=64, bench=False):
    dtype = torch.bfloat16
    device = "cuda:0"
    x = torch.randn((M, K), dtype=dtype, device=device)
    w = torch.randn((N, K), dtype=dtype, device=device)
    dy = torch.randn((M, N), dtype=dtype, device=device)

    hm = make_hadamard_matrix(B, dtype=dtype, device=device, norm=True)

    y_ref = x @ w.t()
    dx_ref = dy @ w
    dw_ref = dy.t() @ x

    xh, xht, xq, xs, xqt, xst = torch_hadamard_quant(x, hm, round_scale=False)
    wh, wht, wq, ws, wqt, wst = torch_hadamard_quant(w, hm, round_scale=False)
    dyh, dyht, dyq, dys, dyqt, dyst = torch_hadamard_quant(dy, hm, round_scale=False)

    y = xh @ wh.t()
    dx = dyh @ wht.t()
    dw = dyht @ xht.t()

    output_check(y_ref, y, "bf16.y", atol=2)
    output_check(dx_ref, dx, "bf16.dx", atol=2)
    output_check(dw_ref, dw, "bf16.dw", atol=2)

    x_q, x_scale, xt_q, xt_scale = triton_hadamard_quant(x, hm)
    output_check(xq, x_q, "x.data", rtol=0.125)
    output_check(xs, x_scale, "x.scale")
    output_check(xqt, xt_q, "xt.data", rtol=0.125)
    output_check(xst, xt_scale, "xt.scale")

    w_q, w_scale, wt_q, wt_scale = triton_hadamard_quant(w, hm)
    output_check(wq, w_q, "w.data", rtol=0.125)
    output_check(ws, w_scale, "w.scale")
    output_check(wqt, wt_q, "wt.data", rtol=0.125)
    output_check(wst, wt_scale, "wt.scale")

    dy_q, dy_scale, dyt_q, dyt_scale = triton_hadamard_quant(dy, hm)
    output_check(dyq, dy_q, "dy.data", rtol=0.125)
    output_check(dys, dy_scale, "dy.scale")
    output_check(dyqt, dyt_q, "dyt.data", rtol=0.125)
    output_check(dyst, dyt_scale, "dyt.scale")


def test_hadamard_quant_linear(M=8192, N=1024, K=2048, B=64):
    dtype = torch.bfloat16
    device = "cuda:0"
    linear = HadamardQuantLinear(K, N, bias=False, dtype=dtype, device=device)
    x = torch.randn((M, K), dtype=dtype, device=device).requires_grad_()
    w = torch.randn((N, K), dtype=dtype, device=device)
    dy = torch.randn((M, N), dtype=dtype, device=device)
    linear.weight.data.copy_(w)

    y_ref = x @ w.t()
    y = linear(x)
    output_check(y_ref, y, name="y", rtol=-1)

    dx_ref = dy @ w
    dw_ref = dy.t() @ x
    y.backward(dy)
    dw = linear.weight.grad
    dx = x.grad
    output_check(dx_ref, dx, name="dx", rtol=-1)
    output_check(dw_ref, dw, name="dw", rtol=-1)


if __name__ == "__main__":
    test_hadamard_quant(M=8192, N=1024, K=2048, B=64, bench=False)
    test_hadamard_quant_linear(M=8192, N=1024, K=2048, B=64)
