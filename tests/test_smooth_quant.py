# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.facade.smooth_quant_linear import SmoothQuantLinear
from linghe.quant.smooth import (triton_batch_smooth_quant,
                                 triton_subrow_smooth_quant,
                                 triton_transpose_rescale_smooth_quant,
                                 triton_smooth_quant,
                                 triton_transpose_smooth_quant)
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.tools.util import (torch_make_indices,
                               torch_smooth_quant,
                               round_up)


def torch_split_smooth_quant(x_split, smooth_scales, round_scale=False):
    x_qs = []
    x_scales = []
    x_maxs = []
    for i, x_ in enumerate(x_split):
        x_maxs.append(x_.abs().amax(0))
        x_smooth = x_ / smooth_scales[i]
        x_scale_ = x_smooth.float().abs().amax(1) / 448
        if round_scale:
            x_scale_ = torch.exp2(torch.ceil(torch.log2(x_scale_)))
        x_q_ = (x_smooth / x_scale_[:, None]).to(torch.float8_e4m3fn)
        x_qs.append(x_q_)
        x_scales.append(x_scale_)
    x_maxs = torch.stack(x_maxs, 0)
    return x_qs, x_scales, x_maxs


def torch_subrow_smooth_quant(x, smooth_scale, x_q, x_scale, subrow_scales,
                              offset, size,
                              reverse=False, round_scale=False):
    limit = 448 * torch.ones((1,), dtype=smooth_scale.dtype,
                             device=smooth_scale.device)
    # subrow_scales is saved as 448/max

    M, N = x_q.shape
    if offset % N > 0:
        si = offset % N
        k = N - si
        x_slice = x.view(-1)[0:k]
        smooth_scale_slice = smooth_scale[si: N]
        if not reverse:
            smooth_scale_slice = 1 / smooth_scale_slice
        x_smooth = x_slice * smooth_scale_slice

        scale = subrow_scales[0:1]
        if round_scale:
            scale = torch.exp2(torch.floor(torch.log2(scale)))

        x_q_slice = torch.minimum(torch.maximum(x_smooth / scale, -limit),
                                  limit).to(torch.float8_e4m3fn)
        x_q.view(-1)[offset:offset + k] = x_q_slice

    if (offset + size) % N > 0:
        k = (offset + size) % N
        x_slice = x.view(-1)[-k:]
        smooth_scale_slice = smooth_scale[0: k]
        if not reverse:
            smooth_scale_slice = 1 / smooth_scale_slice
        x_smooth = x_slice * smooth_scale_slice
        scale = subrow_scales[1:2]
        if round_scale:
            scale = torch.exp2(torch.floor(torch.log2(scale)))
        x_q_slice = torch.minimum(torch.maximum(x_smooth / scale, -limit),
                                  limit).to(torch.float8_e4m3fn)
        x_q.view(-1)[(offset + size - k):(offset + size)] = x_q_slice
        x_scale[(offset + size) // N] = scale


def torch_rescale_quant(y_q, org_smooth_scale, y_scale, transpose_smooth_scale,
                        reverse=True, round_scale=True):
    assert reverse
    y = y_q.float() / org_smooth_scale * y_scale[:, None]
    y_q, y_scale, _ = torch_smooth_quant(y.t(), transpose_smooth_scale,
                                         reverse=True, round_scale=round_scale)
    return y_q, y_scale


def triton_split_smooth_quant(x_split, smooth_scales):
    x_qs = []
    x_scales = []
    for i, x_ in enumerate(x_split):
        x_q_, x_scale_, _ = triton_smooth_quant(x_, smooth_scales[i])
        x_qs.append(x_q_)
        x_scales.append(x_scale_)
    return x_qs, x_scales


def test_triton_smooth_quant(M=4096, N=4096, bench=False):
    device = 'cuda:0'
    x = torch.randn((M, N), dtype=torch.bfloat16, device=device)
    smooth_scale = torch.randn((N,), device=device,
                               dtype=torch.float32).abs() + 1.0
    round_scale = False
    rtol = 2 if round_scale else 0.125
    x_q_ref, scales_ref, x_maxs_ref = torch_smooth_quant(x, smooth_scale,
                                                         reverse=False,
                                                         round_scale=round_scale)

    x_q, x_scale, x_maxs = triton_smooth_quant(x, smooth_scale,
                                               reverse=False,
                                               round_scale=round_scale,
                                               calibrate=True)
    output_check(x_q_ref, x_q,
                 'triton_smooth_quant.data', rtol=rtol)
    output_check(scales_ref, x_scale, 'triton_smooth_quant.scale')
    output_check(x_maxs_ref, x_maxs, 'triton_smooth_quant.x_maxs')

    if bench:
        benchmark_func(triton_smooth_quant, x,
                       smooth_scale,
                       reverse=False,
                       round_scale=True,
                       calibrate=False,
                       ref_bytes=M * N * 3)


def test_triton_subrow_smooth_quant(M=4096, N=5120, offset=4096,
                                    size=16384):
    device = 'cuda:0'
    x = torch.randn((size,), dtype=torch.float32, device=device)
    x_q = torch.zeros((M, N), dtype=torch.bfloat16, device=device).to(
        torch.float8_e4m3fn)
    x_scale = torch.zeros((M,), dtype=torch.float32, device=device).abs()
    smooth_scale = torch.randn((N,), device=device,
                               dtype=torch.float32).abs() + 1
    subrow_scales = torch.randn((2,), device=device,
                                dtype=torch.float32).abs() + 1

    x_ref = x.clone()
    x_q_ref = x_q.clone()
    x_scale_ref = x_scale.clone()
    subrow_scales_ref = subrow_scales.clone()
    torch_subrow_smooth_quant(x_ref, smooth_scale, x_q_ref, x_scale_ref,
                              subrow_scales_ref, offset, size,
                              reverse=False, round_scale=False)

    triton_subrow_smooth_quant(x, smooth_scale, x_q, x_scale,
                               subrow_scales, offset, size,
                               reverse=False, round_scale=False)

    output_check(x_q_ref.float(), x_q.float(), 'subrow.data')
    output_check(x_scale_ref, x_scale, 'subrow.scale')

    if offset % N > 0:
        k = N - offset % N
        output_check(x_q_ref.float().view(-1)[offset:offset + k],
                     x_q.float().view(-1)[offset:offset + k],
                     'subrow.data.tail')

    if (offset + size) % N > 0:
        k = (offset + size) % N
        output_check(x_q_ref.float().view(-1)[offset + size - k:offset + size],
                     x_q.float().view(-1)[offset + size - k:offset + size],
                     'subrow.data.head')
        row_id = (offset + size) // N
        output_check(x_scale_ref[row_id], x_scale[row_id], 'subrow.scale.slice')


def test_triton_transpose_smooth_quant(M=4096, N=4096, bench=False):
    device = 'cuda:0'
    P = round_up(M, b=32)
    y = torch.randn((M, N), dtype=torch.bfloat16, device=device) ** 3 * 1e-10
    transpose_smooth_scale = torch.randn((M,), device=device,
                                         dtype=torch.float32).abs() * 10 + 1
    yt_q, yt_scale = triton_transpose_smooth_quant(y,
                                                   transpose_smooth_scale,
                                                   reverse=True,
                                                   pad=True,
                                                   round_scale=True)
    q_ref, scale_ref, maxs_ref = torch_smooth_quant(y.T.contiguous(),
                                                    transpose_smooth_scale,
                                                    reverse=True,
                                                    round_scale=True)

    assert yt_q.shape[1] == P
    if P > M:
        assert yt_q.float()[:, M:].abs().sum().item() == 0
    output_check(q_ref, yt_q[:, :M],
                 'triton_transpose_smooth_quant.data')
    output_check(scale_ref, yt_scale,
                 'triton_transpose_smooth_quant.scale')

    if bench:
        benchmark_func(triton_transpose_smooth_quant, y,
                       transpose_smooth_scale,
                       reverse=True,
                       pad=True,
                       round_scale=True,
                       ref_bytes=M * N * 3)


def test_triton_transpose_rescale_smooth_quant(M=4096, N=4096,
                                               round_scale=False):
    device = 'cuda:0'
    P = round_up(M, b=32)
    y = torch.randn((M, N), dtype=torch.bfloat16, device=device) ** 3
    org_smooth_scale = torch.randn((N,), device=device,
                                   dtype=torch.float32).abs() * 10 + 1
    if round_scale:
        org_smooth_scale = torch.exp2(torch.ceil(torch.log2(org_smooth_scale)))
    transpose_smooth_scale = torch.randn((M,), device=device,
                                         dtype=torch.float32).abs() + 0.1
    if round_scale:
        transpose_smooth_scale = torch.exp2(
            torch.ceil(torch.log2(transpose_smooth_scale)))

    y_q, y_scale, y_maxs = triton_smooth_quant(y, org_smooth_scale,
                                               reverse=True,
                                               round_scale=round_scale)

    yt_gt, yt_scale_gt, yt_maxs_gt = torch_smooth_quant(y.t(),
                                                        transpose_smooth_scale,
                                                        reverse=True,
                                                        round_scale=round_scale)

    yt_q_ref, yt_scale_ref = torch_rescale_quant(y_q, org_smooth_scale, y_scale,
                                                 transpose_smooth_scale,
                                                 reverse=True,
                                                 round_scale=round_scale)

    yt_q, yt_scale = triton_transpose_rescale_smooth_quant(y_q,
                                                           org_smooth_scale,
                                                           y_scale,
                                                           transpose_smooth_scale,
                                                           reverse=True,
                                                           pad=True,
                                                           round_scale=round_scale)

    if P > M:
        assert yt_q.shape[1] == P
        yt_q.float()[:, M:].abs().sum().item() == 0

    output_check(yt_q_ref, yt_q[:, :M],
                 'triton_transpose_rescale_smooth_quant.data')
    output_check(yt_scale_ref, yt_scale,
                 'triton_transpose_rescale_smooth_quant.scale')

    # should dequant and compare with gt
    # output_check(yt_gt, yt_q[:, :M],
    #              'triton_transpose_rescale_smooth_quant.data.gt')
    # output_check(yt_scale_gt, yt_scale,
    #              'triton_transpose_rescale_smooth_quant.scale.gt')


def test_triton_batch_smooth_quant(M=4096, N=4096, n_experts=32, topk=8,
                                   round_scale=False, bench=False):
    device = 'cuda:0'

    smooth_scales = 1 + 10 * torch.rand((n_experts, N), device=device,
                                        dtype=torch.float32)

    logits = torch.randn((M, n_experts), dtype=torch.float32, device=device)
    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
        logits, topk=topk, bias=0.0)
    token_count_per_expert_list = token_count_per_expert.tolist()
    x = torch.randn((sum(token_count_per_expert_list), N), dtype=torch.bfloat16,
                    device=device)

    x_q, x_scale, x_maxs = triton_batch_smooth_quant(x, smooth_scales,
                                                     token_count_per_expert,
                                                     reverse=False,
                                                     round_scale=round_scale,
                                                     calibrate=True)

    x_split = torch.split(x, token_count_per_expert_list)
    x_q_ref, x_scale_ref, x_maxs_ref = torch_split_smooth_quant(x_split,
                                                                smooth_scales)
    x_q_ref = torch.cat([x.view(torch.uint8) for x in x_q_ref], 0).view(
        torch.float8_e4m3fn)
    x_scale_ref = torch.cat(x_scale_ref, 0)
    rtol = 2 if round_scale else 0.125
    output_check(x_q_ref, x_q, 'triton_batch_smooth_quant.data', rtol=rtol)
    output_check(x_scale_ref.float(), x_scale.float(),
                 'triton_batch_smooth_quant.scale')
    output_check(x_maxs_ref.float(), x_maxs.float(),
                 'triton_batch_smooth_quant.maxs')

    if bench:
        n_repeat = 100
        ref_time = benchmark_func(triton_split_smooth_quant, x_split,
                                  smooth_scales, n_repeat=n_repeat)
        benchmark_func(triton_batch_smooth_quant, x, smooth_scales,
                       token_count_per_expert, reverse=False,
                       round_scale=round_scale, n_repeat=n_repeat,
                       ref_time=ref_time)
        benchmark_func(triton_batch_smooth_quant, x, smooth_scales,
                       token_count_per_expert, reverse=False,
                       round_scale=round_scale, calibrate=True,
                       n_repeat=n_repeat, ref_time=ref_time)


def test_smooth_quant_linear(M=8192, N=1024, K=2048):
    dtype = torch.bfloat16
    device = 'cuda:0'
    linear = SmoothQuantLinear(K, N, bias=False, dtype=dtype, device=device)
    x = (10 * torch.randn((M, K), dtype=dtype, device=device)).requires_grad_()
    w = 0.1 * torch.randn((N, K), dtype=dtype, device=device)
    dy = 1e-6 * torch.randn((M, N), dtype=dtype, device=device)
    linear.weight.data.copy_(w)

    y_ref = x @ w.t()
    y = linear(x)
    output_check(y_ref, y, name='y')

    dx_ref = dy @ w
    dw_ref = dy.t() @ x
    y.backward(dy)
    dw = linear.weight.grad
    dx = x.grad
    output_check(dx_ref, dx, name='dx')
    output_check(dw_ref, dw, name='dw')


if __name__ == '__main__':
    test_triton_smooth_quant(M=16384, N=2048, bench=False)
    test_triton_smooth_quant(M=8192, N=4096, bench=False)
    test_triton_smooth_quant(M=4096, N=8192, bench=False)
    test_triton_smooth_quant(M=8192, N=3072, bench=False)
    test_triton_smooth_quant(M=8192, N=6144, bench=False)
    test_triton_smooth_quant(M=16384, N=512, bench=False)
    test_triton_smooth_quant(M=3457, N=512, bench=False)

    test_triton_subrow_smooth_quant(M=4096, N=5120, offset=5120,
                                    size=2048)
    test_triton_subrow_smooth_quant(M=4096, N=5120, offset=4096,
                                    size=5120)
    test_triton_subrow_smooth_quant(M=4096, N=5120, offset=5120,
                                    size=5120 * 10 - 1024)

    test_triton_transpose_smooth_quant(M=16384, N=2048, bench=False)
    test_triton_transpose_smooth_quant(M=8192, N=4096, bench=False)
    test_triton_transpose_smooth_quant(M=4096, N=8192, bench=False)
    test_triton_transpose_smooth_quant(M=4096, N=3072, bench=False)

    test_triton_transpose_rescale_smooth_quant(M=4096, N=4096,
                                               round_scale=True)
    test_triton_transpose_rescale_smooth_quant(M=3895, N=4096,
                                               round_scale=True)
    test_triton_transpose_rescale_smooth_quant(M=4096, N=3072,
                                               round_scale=True)
    test_triton_transpose_rescale_smooth_quant(M=395, N=2048,
                                               round_scale=True)

    test_triton_batch_smooth_quant(M=4096, N=4096, n_experts=32, topk=8,
                                   round_scale=False)
    # test_smooth_quant_linear(M=8192, N=1024, K=2048)
