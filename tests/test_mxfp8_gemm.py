
# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import random
from typing import Optional, List


import torch

from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.tools.util import torch_mxfp8_quant, torch_batch_mxfp8_quant
from linghe.gemm.mxfp8_gemm import (triton_mxfp8_gemm,
                                    triton_mxfp8_gemm_forward,
                                    triton_mxfp8_gemm_backward,
                                    triton_mxfp8_gemm_update,
                                    triton_mxfp8_grouped_gemm,
                                    triton_mxfp8_grouped_gemm_forward,
                                    triton_mxfp8_grouped_gemm_backward,
                                    triton_mxfp8_grouped_gemm_update,
                                    triton_native_mxfp8_grouped_gemm
                                    )

def test_mxfp8_gemm(M=4096, N=4096, K=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)
    y = torch.randn(M, N, dtype=dtype, device=device)

    x_q, x_scale, xt_q, xt_scale = torch_mxfp8_quant(x)
    w_q, w_scale, wt_q, wt_scale = torch_mxfp8_quant(w)
    y_q, y_scale, yt_q, yt_scale = torch_mxfp8_quant(y)

    out_ref = x@w.t()
    out = triton_mxfp8_gemm_forward(x_q, w_q, x_scale, w_scale, out_dtype=dtype)
    output_check(out_ref, out, 'gemm.forward', atol=-1)

    out = triton_mxfp8_gemm(x_q, w_q, x_scale, w_scale, out_dtype=dtype, layout='TN')
    output_check(out_ref, out, 'gemm.forward', atol=-1)

    out_ref = y@w
    out = triton_mxfp8_gemm_backward(y_q, wt_q, y_scale, wt_scale, out_dtype=dtype)
    output_check(out_ref, out, 'gemm.backward', atol=-1)

    out = triton_mxfp8_gemm(y_q, wt_q, y_scale, wt_scale, out_dtype=dtype, layout='NN')
    output_check(out_ref, out, 'gemm.backward', atol=-1)


    out_ref = y.t()@x
    out = triton_mxfp8_gemm_update(yt_q, xt_q, yt_scale, xt_scale, out_dtype=dtype)
    output_check(out_ref, out, 'gemm.update', atol=-1)

    out = triton_mxfp8_gemm(yt_q, xt_q, yt_scale, xt_scale, out_dtype=dtype, layout='NT')
    output_check(out_ref, out, 'gemm.update', atol=-1)


    if bench:
        ref_bytes = M * N * 4
        ref_flops = M * N * N * 2
        benchmark_func(triton_mxfp8_gemm_forward, x_q, w_q, x_scale, w_scale,
                      out_dtype=dtype,
                      ref_bytes=ref_bytes,
                      ref_flops=ref_flops,
                      n_repeat=1)
        benchmark_func(triton_mxfp8_gemm, x_q, w_q, x_scale, w_scale,
                      out_dtype=dtype,
                      layout='TN',
                      ref_bytes=ref_bytes,
                      ref_flops=ref_flops,
                      n_repeat=1)

        benchmark_func(triton_mxfp8_gemm_backward, y_q, wt_q, y_scale, wt_scale,
                      out_dtype=dtype,
                      ref_bytes=ref_bytes,
                      ref_flops=ref_flops,
                      n_repeat=1)
        benchmark_func(triton_mxfp8_gemm, y_q, wt_q, y_scale, wt_scale,
                      out_dtype=dtype,
                      layout='NN',
                      ref_bytes=ref_bytes,
                      ref_flops=ref_flops,
                      n_repeat=1)

        benchmark_func(triton_mxfp8_gemm_update, yt_q, xt_q, yt_scale, xt_scale,
                      out_dtype=dtype,
                      ref_bytes=ref_bytes,
                      ref_flops=ref_flops,
                      n_repeat=1)
        benchmark_func(triton_mxfp8_gemm, yt_q, xt_q, yt_scale, xt_scale,
                      out_dtype=dtype,
                      layout='NT',
                      ref_bytes=ref_bytes,
                      ref_flops=ref_flops,
                      n_repeat=1)


def test_mxfp8_grouped_gemm(M=1024, N=4096, K=4096, n_experts=32, native=True, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    splits = [M] * n_experts
    # splits = [max(random.randint(M - 256, M + 256), 0) for x in
    #           range(n_experts)]
    # splits = [(x + 31) // 32 * 32 for x in splits]

    xs = [torch.randn((M, K), dtype=dtype, device=device) for M in splits]
    ws = [torch.randn((N, K), dtype=dtype, device=device) for _ in splits]
    ys = [torch.randn((M, N), dtype=dtype, device=device) for M in splits]

    forward_out_ref = [x@ws[i].t() for i, x in enumerate(xs)]
    forward_out_ref = torch.cat(forward_out_ref, 0)

    x_ims = [torch_mxfp8_quant(x) for x in xs]
    xqs = [x[0] for x in x_ims]
    xss = [x[1] for x in x_ims]

    w_ims = [torch_mxfp8_quant(w) for w in ws]
    wqs = [x[0] for x in w_ims]
    wss = [x[1] for x in w_ims]

    forward_out = triton_mxfp8_grouped_gemm_forward(xqs,
                                    wqs,
                                    xss,
                                    wss,
                                    splits,
                                    out_dtype=dtype)
    output_check(forward_out_ref, forward_out, 'grouped_gemm.forward', atol=-1)

    forward_out = triton_mxfp8_grouped_gemm(xqs,
                                    wqs,
                                    xss,
                                    wss,
                                    splits,
                                    out_dtype=dtype,
                                    layout='TN')
    output_check(forward_out_ref, forward_out, 'grouped_gemm.forward', atol=-1)

    forward_out = triton_native_mxfp8_grouped_gemm(xqs,
                                    wqs,
                                    xss,
                                    wss,
                                    forward_out,
                                    splits,
                                    layout='TN')
    output_check(forward_out_ref, forward_out, 'grouped_gemm.forward', atol=-1)


    y_ims = [torch_mxfp8_quant(y) for y in ys]
    yqs = [x[0] for x in y_ims]
    yss = [x[1] for x in y_ims]

    backward_out_ref = [y@ws[i] for i, y in enumerate(ys)]
    backward_out_ref = torch.cat(backward_out_ref, 0)

    wtqs = [x[2] for x in w_ims]
    wtss = [x[3] for x in w_ims]

    backward_out = triton_mxfp8_grouped_gemm_backward(yqs,
                                    wtqs,
                                    yss,
                                    wtss,
                                    splits,
                                    out_dtype=dtype)
    output_check(backward_out_ref, backward_out, 'grouped_gemm.backward', atol=-1)

    backward_out = triton_mxfp8_grouped_gemm(yqs,
                                    wtqs,
                                    yss,
                                    wtss,
                                    splits,
                                    out_dtype=dtype,
                                    layout='NN')
    output_check(backward_out_ref, backward_out, 'grouped_gemm.backward', atol=-1)

    backward_out = triton_native_mxfp8_grouped_gemm(yqs,
                                    wtqs,
                                    yss,
                                    wtss,
                                    backward_out,
                                    splits,
                                    layout='NN')
    output_check(backward_out_ref, backward_out, 'grouped_gemm.backward', atol=-1)


    update_out_ref = [y.float().t()@xs[i].float() for i, y in enumerate(ys)]
    update_out_ref = torch.cat(update_out_ref, 0)

    ytqs = [x[2] for x in y_ims]
    ytss = [x[3] for x in y_ims]
    xtqs = [x[2] for x in x_ims]
    xtss = [x[3] for x in x_ims]

    update_out = [torch.randn((N, K), dtype=torch.float32, device=device) for _ in splits]
    update_out = triton_mxfp8_grouped_gemm_update(ytqs,
                                    xtqs,
                                    ytss,
                                    xtss,
                                    splits,
                                    out_dtype=dtype,
                                    out=update_out)
    output_check(update_out_ref, torch.cat(update_out, 0), 'grouped_gemm.update', atol=-1)

    update_out = [torch.randn((N, K), dtype=torch.float32, device=device) for _ in splits]
    update_out = triton_mxfp8_grouped_gemm(ytqs,
                                    xtqs,
                                    ytss,
                                    xtss,
                                    splits,
                                    out_dtype=dtype,
                                    out=update_out,
                                    layout='NT')
    output_check(update_out_ref, torch.cat(update_out, 0), 'grouped_gemm.update', atol=-1)

    update_out = [torch.randn((N, K), dtype=torch.float32, device=device) for _ in splits]
    update_out = triton_native_mxfp8_grouped_gemm(ytqs,
                                    xtqs,
                                    ytss,
                                    xtss,
                                    update_out,
                                    splits,
                                    layout='NT')
    output_check(update_out_ref, torch.cat(update_out, 0), 'grouped_gemm.update', atol=-1)


    if bench:
        ref_flops = M * N * K * n_experts * 2
        benchmark_func(triton_mxfp8_grouped_gemm_forward,
                       xqs, 
                       wqs,
                       xss,
                       wss,
                       splits,
                       out_dtype=dtype,
                       ref_flops=ref_flops,
                       n_repeat=10)

        benchmark_func(triton_mxfp8_grouped_gemm,
                       xqs, 
                       wqs,
                       xss,
                       wss,
                       splits,
                       out_dtype=dtype,
                       layout='TN',
                       ref_flops=ref_flops,
                       n_repeat=10)

        benchmark_func(triton_native_mxfp8_grouped_gemm,
                       xqs, 
                       wqs,
                       xss,
                       wss,
                       forward_out,
                       splits,
                       layout='TN',
                       ref_flops=ref_flops,
                       n_repeat=10)

        benchmark_func(triton_mxfp8_grouped_gemm_backward,
                       yqs,
                       wtqs,
                       yss,
                       wtss,
                       splits,
                       out_dtype=dtype,
                       ref_flops=ref_flops,
                       n_repeat=10)
        benchmark_func(triton_mxfp8_grouped_gemm,
                       yqs,
                       wtqs,
                       yss,
                       wtss,
                       splits,
                       out_dtype=dtype,
                       layout='NN',
                       ref_flops=ref_flops,
                       n_repeat=10)

        benchmark_func(triton_native_mxfp8_grouped_gemm,
                       yqs,
                       wtqs,
                       yss,
                       wtss,
                       backward_out,
                       splits,
                       layout='NN',
                       ref_flops=ref_flops,
                       n_repeat=10)

        benchmark_func(triton_mxfp8_grouped_gemm_update,
                       ytqs,
                       xtqs,
                       ytss,
                       xtss,
                       splits,
                       out_dtype=dtype,
                       out=update_out,
                       ref_flops=ref_flops,
                       n_repeat=10)
        benchmark_func(triton_mxfp8_grouped_gemm,
                       ytqs,
                       xtqs,
                       ytss,
                       xtss,
                       splits,
                       out_dtype=dtype,
                       out=update_out,
                       layout='NT',
                       ref_flops=ref_flops,
                       n_repeat=10)
        benchmark_func(triton_native_mxfp8_grouped_gemm,
                       ytqs,
                       xtqs,
                       ytss,
                       xtss,
                       update_out,
                       splits,
                       layout='NT',
                       ref_flops=ref_flops,
                       n_repeat=10)


if __name__ == '__main__':
    # test_mxfp8_gemm(M=4096, N=8192, K=4096, bench=True)
    # test_mxfp8_grouped_gemm(M=8192, N=8192, K=4096, n_experts=1, bench=True)
    # test_mxfp8_grouped_gemm(M=4096, N=8192, K=4096, n_experts=2, bench=True)
    # test_mxfp8_grouped_gemm(M=1024, N=8192, K=4096, n_experts=32, bench=True)
    test_mxfp8_grouped_gemm(M=4096, N=8192, K=4096, n_experts=32, bench=True)