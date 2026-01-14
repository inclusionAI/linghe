# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def sort_chunks_by_index_kernel(x_ptr, y_ptr, scale_ptr, scale_output_ptr,
                                count_ptr,
                                accum_ptr, rev_accum_ptr, index_ptr, M,
                                N: tl.constexpr, SCALE: tl.constexpr,
                                K: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    index = tl.load(index_ptr + pid)
    count = tl.load(count_ptr + index)
    ei = tl.load(accum_ptr + index)
    si = ei - count
    rev_ei = tl.load(rev_accum_ptr + pid)
    rev_si = rev_ei - count
    for i in range(count):
        x = tl.load(x_ptr + si * N + i * N + tl.arange(0, N))
        tl.store(y_ptr + rev_si * N + i * N + tl.arange(0, N), x)

    if SCALE:
        for i in range(tl.cdiv(count, K)):
            scale = tl.load(scale_ptr + si + i * K + tl.arange(0, K),
                            mask=i * K + tl.arange(0, K) < count)
            tl.store(scale_output_ptr + rev_si + i * K + tl.arange(0, K), scale,
                     mask=i * K + tl.arange(0, K) < count)


def triton_sort_chunks_by_index(x, counts, indices, scales=None):
    """
    split x to multiple tensors and cat with indices,
    it is used for permutation in moe with all2all communication
    Args:
        x: [bs, dim]
        counts: [n_split]
        indices: [n_split]
        scales: [bs]

    Returns:
        - y: output tensor
        - output_scales: output scales if scales is not None
    """
    assert x.is_contiguous()
    M, N = x.shape
    n_split = counts.shape[0]
    device = x.device
    y = torch.empty((M, N), device=device, dtype=x.dtype)
    if scales is not None:
        output_scales = torch.empty((M,), device=device, dtype=torch.float32)
        S = True
    else:
        output_scales = None
        S = False
    accums = torch.cumsum(counts, 0)
    reverse_accums = torch.cumsum(counts[indices], 0)
    # TODO: adapt for n_expert <= 64
    K = 256
    grid = (n_split,)
    sort_chunks_by_index_kernel[grid](
        x,
        y,
        scales,
        output_scales,
        counts,
        accums,
        reverse_accums,
        indices,
        M,
        N,
        S,
        K,
        num_stages=3,
        num_warps=8
    )
    return y, output_scales
