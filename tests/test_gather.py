# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.quant.block import triton_batch_blockwise_quant
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.tools.util import (torch_batch_smooth_quant,
                               torch_blockwise_quant,
                               torch_make_indices,
                               torch_smooth_quant)
from linghe.utils.gather import (triton_make_row_id_map,
                                 triton_make_row_id_map_and_index,
                                 triton_index_select,
                                 triton_permute_with_mask_map,
                                 triton_smooth_permute_with_indices,
                                 triton_smooth_permute_with_mask_map,
                                 triton_smooth_weighted_permute_with_indices,
                                 triton_batch_transpose_smooth_permute_with_indices,
                                 triton_batch_block_pad_permute_with_indices,
                                 )


def torch_index_select(y, indices):
    output = y.index_select(0, indices)
    return output


def torch_select_with_padded_map_mask(y, mask_map, out_tokens):
    E = mask_map.shape[1]
    if y.ndim > 1:
        output = torch.zeros((out_tokens, y.shape[1]), dtype=y.dtype,
                             device=y.device)
    else:
        output = torch.zeros((out_tokens,), dtype=y.dtype, device=y.device)
    for i in range(E):
        indices = mask_map[:, i]
        src_idx = torch.nonzero(indices > -1)
        dst_idx = indices[src_idx]
        output[dst_idx] = y[src_idx]
    return output


def torch_ravel_with_padded_map_mask(y, mask_map, out_tokens):
    E = mask_map.shape[1]
    output = torch.zeros((out_tokens,), dtype=y.dtype, device=y.device)
    for i in range(E):
        indices = mask_map[:, i]
        src_idx = torch.nonzero(indices > -1)
        dst_idx = indices[src_idx]
        output[dst_idx] = y[src_idx, i]
    return output


def torch_fp16_index_select(x, scales, indices):
    return x.index_select(0, indices), scales.index_select(0, indices)


def torch_scatter(logits, routing_map, weights):
    logits[routing_map] = weights


# optional dequant and smooth and quant
def torch_smooth_permute_with_indices(grad_data, grad_scale, indices,
                                      smooth_scales,
                                      token_count_per_expert_list,
                                      round_scale=True):
    M, N = grad_data.shape
    if grad_scale is not None:
        B = grad_data.shape[1] // (
            1 if grad_scale.ndim == 1 else grad_scale.shape[1])
    q_refs = []
    scale_refs = []
    s = 0
    for i, c in enumerate(token_count_per_expert_list):
        c = token_count_per_expert_list[i]
        data_slice = grad_data.view(torch.uint8)[indices[s:s + c]].view(
            torch.float8_e4m3fn)
        if grad_scale is not None:
            scale_slice = grad_scale[indices[s:s + c]]
            y_smooth = (data_slice.float().view(c, N // B, B) * scale_slice[:,
                                                                :,
                                                                None]).view(c,
                                                                            N) / \
                       smooth_scales[i]
        else:
            y_smooth = data_slice.float() / smooth_scales[i]
        scale = y_smooth.abs().amax(1) / 448
        if round_scale:
            scale = torch.exp2(torch.ceil(torch.log2(scale)))
        scale_refs.append(scale)
        q = (y_smooth / scale[:, None]).to(torch.float8_e4m3fn)
        q_refs.append(q.view(torch.uint8))
        s += c
    q_ref = torch.cat(q_refs, 0).view(torch.float8_e4m3fn)
    scale_ref = torch.cat(scale_refs, 0)

    return q_ref, scale_ref


# desmooth,dequant, gather, pad, transpose, smooth, quant
def torch_batch_transpose_smooth_permute_with_indices(x_q, x_scale,
                                                      org_smooth_scale,
                                                      smooth_scales,
                                                      indices,
                                                      token_count_per_expert_list,
                                                      round_scale=True):
    M, DIM = x_q.shape
    q_refs = []
    scale_refs = []
    s = 0
    for i, c in enumerate(token_count_per_expert_list):
        c = token_count_per_expert_list[i]
        if c == 0:
            y_scale = torch.zeros((DIM,), dtype=torch.float32,
                                  device=x_q.device)
            scale_refs.append(y_scale.view(-1))
            continue
        N = (c + 31) // 32 * 32
        data_slice = x_q[indices[s:s + c]]
        if x_scale is not None:
            scale_slice = x_scale[indices[s:s + c]]
            y = data_slice.float() * scale_slice[:, None] * org_smooth_scale
        else:
            y = data_slice.float()
        smooth_scale = smooth_scales[s:s + c]
        if N > c:
            y = torch.nn.functional.pad(y, (0, 0, 0, N - c))
            smooth_scale = torch.nn.functional.pad(smooth_scale, (0, N - c))
        y_q, y_scale, y_max = torch_smooth_quant(y.t().contiguous(),
                                                 smooth_scale, reverse=True,
                                                 round_scale=round_scale)
        scale_refs.append(y_scale.view(-1))
        q_refs.append(y_q.view(-1))
        s += c
    q_ref = torch.cat(q_refs, 0)
    scale_ref = torch.stack(scale_refs, 0)
    return q_ref, scale_ref


def torch_batch_block_pad_permute_with_indices(x,
                                               indices,
                                               probs,
                                               token_count_per_expert_list,
                                               round_scale=True):
    M, DIM = x.shape
    if M == 0:
        device = x.device
        q_ref = torch.empty((0,), device=device, dtype=torch.float8_e4m3fn)
        s_ref = torch.empty((0,), device=device, dtype=torch.float32)
        qt_ref = torch.empty((0,), device=device, dtype=torch.float8_e4m3fn)
        st_ref = torch.empty((0,), device=device, dtype=torch.float32)
        probs_refs = torch.empty((0,), device=device, dtype=torch.float32)
        return q_ref, s_ref, qt_ref, st_ref, probs_refs

    q_refs = []
    s_refs = []
    qt_refs = []
    st_refs = []
    probs_refs = []
    s = 0
    for i, c in enumerate(token_count_per_expert_list):
        c = token_count_per_expert_list[i]
        if c == 0:
            continue
        index = indices[s:s + c]
        assert len(index) == c
        y = x[index]
        y = y.float()
        p_slice = probs[:, i][index]

        padding_size = (c + 15) // 16 * 16 - c
        if padding_size > 0:
            p_slice = torch.nn.functional.pad(p_slice, (0, padding_size))

        y_q, y_scale, yt_q, yt_scale = torch_blockwise_quant(y,
                                                             round_scale=round_scale,
                                                             padding=True)
        q_refs.append(y_q.view(-1))
        s_refs.append(y_scale.view(-1))
        qt_refs.append(yt_q.view(-1))
        st_refs.append(yt_scale.view(-1))
        probs_refs.append(p_slice)
        s += c
    q_ref = torch.cat(q_refs, 0)
    s_ref = torch.cat(s_refs, 0)
    qt_ref = torch.cat(qt_refs, 0)
    st_ref = torch.cat(st_refs, 0)
    probs_refs = torch.cat(probs_refs, 0)
    return q_ref, s_ref, qt_ref, st_ref, probs_refs


def test_make_id_map(M=4098, n_experts=32, topk=2, bias=0.0, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    logits = torch.randn((M, n_experts), dtype=torch.float32, device=device)
    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
        logits, topk=topk, bias=bias)

    token_count_per_expert_list = token_count_per_expert.tolist()
    out_tokens = sum(token_count_per_expert_list)

    row_id_map_output = triton_make_row_id_map(mask_map)
    assert (row_id_map - row_id_map_output).abs().sum().item() == 0

    _, row_id_indices = triton_make_row_id_map_and_index(mask_map, out_tokens)
    assert (row_id_indices - indices).abs().sum().item() == 0


def test_triton_smooth_weighted_permute_with_indices(M=4096, N=4096,
                                                     n_experts=256,
                                                     topk=8,
                                                     round_scale=True,
                                                     bench=False):
    device = 'cuda:0'
    reverse = True
    y = torch.randn((M, N), dtype=torch.bfloat16, device=device)
    logits = torch.randn((M, n_experts), dtype=torch.float32, device=device)
    smooth_scales = 1 + 10 * torch.rand((n_experts, N), device=device,
                                        dtype=torch.float32)
    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
        logits, topk=topk, bias=0.0)

    tokens = torch.randn((indices.shape[0], N), dtype=torch.bfloat16,
                         device=device)
    y_q, y_scale, y_sum = triton_smooth_weighted_permute_with_indices(
        y, tokens, smooth_scales, token_count_per_expert, indices, x_q=None,
        x_scale=None, reverse=reverse, round_scale=round_scale)

    y_q_ref, y_scale_ref = torch_batch_smooth_quant(y, smooth_scales, indices,
                                                    token_count_per_expert,
                                                    reverse=reverse,
                                                    round_scale=round_scale)
    sum_ref = (tokens * y[indices]).sum(1)

    output_check(y_q_ref.float(), y_q.float(), 'data')
    output_check(y_scale_ref.float(), y_scale.float(), 'scale')
    output_check(sum_ref.float(), y_sum.float(), 'sum')

    if bench:
        n_repeat = 100
        benchmark_func(triton_smooth_weighted_permute_with_indices,
                       y, tokens, smooth_scales, token_count_per_expert,
                       indices, reverse=reverse, round_scale=round_scale,
                       n_repeat=n_repeat)


def test_triton_permute_with_mask_map(M=4096, N=4096, n_experts=256, topk=8,
                                      bench=False):
    device = 'cuda:0'
    dtype = torch.bfloat16
    x = torch.randn(M, N, dtype=dtype, device=device) ** 3
    x_q = x.to(torch.float8_e4m3fn)
    scales = torch.rand(M, dtype=dtype, device=device) * 10

    logits = torch.randn((M, n_experts), dtype=torch.float32, device=device)

    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
        logits, topk=topk, bias=-0.01)
    out_tokens = sum(token_count_per_expert.tolist())

    x_out, scale_out = triton_index_select(x, indices, scale=scales)
    x_out_ref, scale_out_ref = torch_fp16_index_select(x, scales, indices)
    output_check(x_out_ref, x_out, 'x_out')
    output_check(scale_out_ref, scale_out, 'scale_out')

    probs_out_ref = probs.T.contiguous().masked_select(
        (probs > 0).T.contiguous())
    x_out, scale_out, probs_out = triton_permute_with_mask_map(x, scales, probs,
                                                               row_id_map,
                                                               out_tokens,
                                                               contiguous=True)
    output_check(x_out_ref, x_out, 'x_out')
    output_check(scale_out_ref, scale_out, 'scale_out')
    output_check(probs_out_ref, probs_out, 'prob_out')

    nzs = torch.sum(row_id_map >= 0, 0)
    bias = torch.cumsum((nzs + 15) // 16 * 16 - nzs, 0)
    row_id_map_clone = row_id_map.clone().detach()
    row_id_map_clone[:, 1:] += bias[:-1]
    round_row_id_map = torch.where(row_id_map >= 0, row_id_map_clone, -1)
    padded_out_tokens = sum(
        [(x + 15) // 16 * 16 for x in token_count_per_expert.tolist()])
    x_out_ref = torch_select_with_padded_map_mask(x, round_row_id_map,
                                                  padded_out_tokens)
    scale_out_ref = torch_select_with_padded_map_mask(scales, round_row_id_map,
                                                      padded_out_tokens)
    prob_out_ref = torch_ravel_with_padded_map_mask(probs, round_row_id_map,
                                                    padded_out_tokens)
    x_out, scale_out, probs_out = triton_permute_with_mask_map(x, scales, probs,
                                                               round_row_id_map,
                                                               padded_out_tokens,
                                                               contiguous=False,
                                                               tokens_per_expert=token_count_per_expert)
    output_check(x_out_ref, x_out, 'noncontiguous.x_out')
    output_check(scale_out_ref, scale_out, 'noncontiguous.scale_out')
    output_check(prob_out_ref, probs_out, 'noncontiguous.prob')

    if bench:
        n_repeat = 100
        ref_bytes = out_tokens * N * 2
        ref_time = benchmark_func(torch_fp16_index_select, x, scales, indices,
                                  n_repeat=n_repeat, ref_bytes=ref_bytes)
        benchmark_func(triton_index_select, x, indices, scale=scales,
                       n_repeat=n_repeat, ref_time=ref_time,
                       ref_bytes=ref_bytes)
        benchmark_func(triton_permute_with_mask_map, x, scales, probs,
                       row_id_map, out_tokens, contiguous=True,
                       n_repeat=n_repeat,
                       ref_time=ref_time, ref_bytes=ref_bytes)
        benchmark_func(triton_permute_with_mask_map, x, scales, probs,
                       row_id_map, out_tokens, contiguous=False,
                       tokens_per_expert=token_count_per_expert,
                       n_repeat=n_repeat,
                       ref_time=ref_time, ref_bytes=ref_bytes)


def test_triton_smooth_permute_with_mask_map(M=4096, N=4096, n_experts=32,
                                             topk=8, round_scale=True,
                                             bench=False):
    device = 'cuda:0'
    dtype = torch.bfloat16
    smooth_scales = 1 + 10 * torch.rand((n_experts, N), device=device,
                                        dtype=torch.float32)
    logits = torch.randn((M, n_experts), dtype=torch.float32,
                         device=device) ** 3
    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
        logits, topk=topk, bias=-0.01)

    token_count_per_expert_list = token_count_per_expert.tolist()
    out_tokens = sum(token_count_per_expert_list)

    B = 128
    grad_data = torch.randn((M, N), dtype=dtype, device=device).to(
        torch.float8_e4m3fn)
    grad_scale = 1 + torch.rand((M, N // B), dtype=torch.float32, device=device)
    q_ref, scale_ref = torch_smooth_permute_with_indices(grad_data, grad_scale,
                                                         indices, smooth_scales,
                                                         token_count_per_expert_list,
                                                         round_scale=round_scale)
    y_q, y_scale = triton_smooth_permute_with_indices(grad_data,
                                                      grad_scale,
                                                      smooth_scales,
                                                      token_count_per_expert,
                                                      indices,
                                                      x_q=None,
                                                      x_scale=None,
                                                      reverse=False,
                                                      round_scale=round_scale)
    output_check(q_ref, y_q, name='data', rtol=0.125)
    output_check(scale_ref, y_scale, name='scale')

    # smooth_scale_ptrs = torch.tensor([x.data_ptr() for x in torch.split(smooth_scales,1)], device=device)
    permuted_data, permuted_scale = triton_smooth_permute_with_mask_map(
        grad_data, row_id_map, grad_scale, M, n_experts, out_tokens, N,
        smooth_scales, reverse=False, round_scale=round_scale)
    output_check(q_ref.float(), permuted_data.float(), name='smoothed.data',
                 rtol=0.125)
    output_check(scale_ref.float(), permuted_scale.float(), 'smoothed.scale')

    q_ref, scale_ref = torch_smooth_permute_with_indices(grad_data, None,
                                                         indices, smooth_scales,
                                                         token_count_per_expert_list,
                                                         round_scale=round_scale)
    permuted_data, permuted_scale = triton_smooth_permute_with_mask_map(
        grad_data, row_id_map, None, M, n_experts, out_tokens, N,
        smooth_scales, reverse=False, round_scale=round_scale)
    output_check(q_ref.float(), permuted_data.float(), name='smoothed.data',
                 rtol=0.125)
    output_check(scale_ref.float(), permuted_scale.float(), 'smoothed.scale')

    if bench:
        benchmark_func(triton_smooth_permute_with_indices, grad_data,
                       grad_scale, smooth_scales, token_count_per_expert,
                       indices, round_scale=round_scale, n_repeat=100,
                       ref_bytes=out_tokens * N * 2)
        benchmark_func(triton_smooth_permute_with_mask_map, grad_data,
                       row_id_map, grad_scale, M, n_experts, out_tokens, N,
                       smooth_scales, reverse=False, round_scale=round_scale,
                       n_repeat=100, ref_bytes=out_tokens * N * 2)


def test_triton_batch_transpose_smooth_permute_with_indices(M=1024, N=2048,
                                                            n_experts=32,
                                                            topk=8,
                                                            bench=False):
    device = 'cuda:0'
    if True:
        logits = torch.randn((M, n_experts), dtype=torch.float32,
                             device=device) ** 3
        logits[:, 0] -= 1000
        logits[:, 2] -= 100
        probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
            logits, topk=topk, bias=-0.01)

        token_count_per_expert_list = token_count_per_expert.tolist()
        out_tokens = sum(token_count_per_expert_list)

        x = torch.randn((M, N), dtype=torch.bfloat16, device=device).to(
            torch.float8_e4m3fn)
        scale = torch.rand((M,), dtype=torch.float32, device=device) + 0.1
        org_smooth_scale = torch.rand((N,), dtype=torch.float32,
                                      device=device) + 0.1
        smooth_scales = torch.rand((out_tokens,), dtype=torch.float32,
                                   device=device) + 0.1
    else:
        #         torch.save({"x":x, "scale":scale, "org_smooth_scale":org_smooth_scale,"smooth_scales":smooth_scales, "indices":indices, "token_count_per_expert":token_count_per_expert,"splits":splits}, '/tmp/debug.bin')
        state = torch.load('/tmp/debug.bin')
        x = state['x']
        scale = state['scale']
        org_smooth_scale = state['org_smooth_scale']
        smooth_scales = state['smooth_scales']
        indices = state['indices']
        token_count_per_expert = state['token_count_per_expert']
        token_count_per_expert_list = state['splits']
        out_tokens = sum(token_count_per_expert_list)

    x_q_ref, x_scale_ref = torch_batch_transpose_smooth_permute_with_indices(x,
                                                                             scale,
                                                                             org_smooth_scale,
                                                                             smooth_scales,
                                                                             indices,
                                                                             token_count_per_expert_list,
                                                                             round_scale=True)

    x_q, x_scale = triton_batch_transpose_smooth_permute_with_indices(x, scale,
                                                                      org_smooth_scale,
                                                                      smooth_scales,
                                                                      indices,
                                                                      token_count_per_expert,
                                                                      token_count_per_expert_list,
                                                                      round_scale=True)
    output_check(x_q_ref.float(), x_q.float(), name='smoothed.data', rtol=0.125)
    output_check(x_scale_ref.float(), x_scale.float(), 'smoothed.scale')

    x_q_ref, x_scale_ref = torch_batch_transpose_smooth_permute_with_indices(x,
                                                                             None,
                                                                             None,
                                                                             smooth_scales,
                                                                             indices,
                                                                             token_count_per_expert_list,
                                                                             round_scale=True)

    x_q, x_scale = triton_batch_transpose_smooth_permute_with_indices(x, None,
                                                                      None,
                                                                      smooth_scales,
                                                                      indices,
                                                                      token_count_per_expert,
                                                                      token_count_per_expert_list,
                                                                      round_scale=True)
    output_check(x_q_ref.float(), x_q.float(), 'bf16.data')
    output_check(x_scale_ref.float(), x_scale.float(), 'bf16.scale')

    if bench:
        benchmark_func(torch_batch_transpose_smooth_permute_with_indices, x,
                       scale, org_smooth_scale, smooth_scales,
                       indices,
                       token_count_per_expert_list,
                       round_scale=True,
                       ref_bytes=out_tokens * N * 2)
        benchmark_func(triton_batch_transpose_smooth_permute_with_indices, x,
                       scale, org_smooth_scale, smooth_scales,
                       indices,
                       token_count_per_expert, token_count_per_expert_list,
                       round_scale=True,
                       ref_bytes=out_tokens * N * 2)


def test_batch_block_pad_permute_with_indices(M=16384, N=2048, n_experts=32,
                                              topk=2, bench=False):
    device = 'cuda:0'
    logits = torch.randn((M, n_experts), dtype=torch.float32,
                         device=device) ** 3
    logits[:, 0] -= 1000
    logits[:, 2] -= 100
    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
        logits, topk=topk, bias=-0.01)
    token_count_per_expert_list = token_count_per_expert.tolist()

    num_out_tokens = sum(
        [(x + 15) // 16 * 16 for x in token_count_per_expert_list])
    row_id_map, pad_indices = triton_make_row_id_map_and_index(mask_map,
                                                               num_out_tokens,
                                                               multiple_of=16)

    x = torch.randn((M, N), dtype=torch.bfloat16, device=device)

    x_q_ref, x_s_ref, xt_q_ref, xt_s_ref, p_ref = torch_batch_block_pad_permute_with_indices(
        x,
        indices,
        probs,
        token_count_per_expert_list,
        round_scale=True)

    x_q, x_s, xt_q, xt_s, p = triton_batch_block_pad_permute_with_indices(x,
                                                                          token_count_per_expert,
                                                                          pad_indices,
                                                                          token_count_per_expert_list,
                                                                          probs=probs,
                                                                          round_scale=True)
    output_check(x_q_ref.float(), x_q.view(-1).float(), 'data')
    output_check(x_s_ref.float(), x_s.view(-1).float(), 'scale')
    output_check(xt_q_ref.float(), xt_q.view(-1).float(), 't.data')
    output_check(xt_s_ref.float(), xt_s.view(-1).float(), 't.scale')
    output_check(p_ref.float(), p.view(-1).float(), 'prob')

    if bench:
        benchmark_func(triton_batch_block_pad_permute_with_indices, x,
                       token_count_per_expert,
                       pad_indices,
                       token_count_per_expert_list,
                       probs=probs,
                       round_scale=True,
                       ref_bytes=num_out_tokens * N * 4)

        benchmark_func(triton_permute_with_mask_map, x, None, probs,
                       row_id_map, num_out_tokens, contiguous=False,
                       tokens_per_expert=token_count_per_expert,
                       ref_bytes=num_out_tokens * N * 4)
        xs = x[indices]
        benchmark_func(triton_batch_blockwise_quant, xs, token_count_per_expert,
                       token_count_per_expert_list,
                       round_scale=True,
                       ref_bytes=num_out_tokens * N * 4)



if __name__ == '__main__':
    test_make_id_map(M=4098, n_experts=32, topk=2, bias=0.0, bench=False)

    test_triton_permute_with_mask_map(M=16384, N=2048, n_experts=32, topk=8,
                                      bench=False)
    test_triton_permute_with_mask_map(M=8192, N=4096, n_experts=32, topk=8,
                                      bench=False)
    test_triton_permute_with_mask_map(M=7628, N=2048, n_experts=32, topk=8,
                                      bench=False)

    test_triton_smooth_permute_with_mask_map(M=4096, N=4096, n_experts=32,
                                             topk=8)
    test_triton_smooth_permute_with_mask_map(M=7628, N=2048, n_experts=32,
                                             topk=8)

    test_triton_batch_transpose_smooth_permute_with_indices(M=16384, N=2048,
                                                            n_experts=32,
                                                            topk=2, bench=False)
    test_triton_batch_transpose_smooth_permute_with_indices(M=8192, N=4096,
                                                            n_experts=32,
                                                            topk=2, bench=False)

    test_batch_block_pad_permute_with_indices(M=8192 * 2, N=2048, n_experts=32,
                                              topk=2, bench=False)
    test_batch_block_pad_permute_with_indices(M=0, N=2048, n_experts=32, topk=2,
                                              bench=False)
    test_batch_block_pad_permute_with_indices(M=8192, N=1536, n_experts=32,
                                              topk=2, bench=False)
