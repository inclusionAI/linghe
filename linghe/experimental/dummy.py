# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import time
from typing import Optional

import torch
import triton
import triton.language as tl
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check

"""
the code is used to profile triton cpu overhead
"""


@triton.jit
def dummy_kernel(x_ptr,
                 y_ptr,
                 y1_ptr,
                 y2_ptr,
                 y3_ptr,
                 y4_ptr,
                 M,
                 D: tl.constexpr):
    pid = tl.program_id(axis=0)
    x = tl.load(x_ptr + pid * D + tl.arange(0, D))
    tl.store(y_ptr + pid * D + tl.arange(0, D), x)


def triton_dummy(x: torch.Tensor, y: torch.Tensor):
    M, D = x.shape
    grid = (M, )
    dummy_kernel[grid](
        x,
        y,
        y,
        y,
        y,
        y,
        M,
        D,
        num_stages=3,
        num_warps=2
    )
    return x


def triton_fast_dummy(x: torch.Tensor, y: torch.Tensor):
    M, D = x.shape
    grid = (M, )
    dummy_kernel[grid](
        x,
        y,
        y,
        y,
        y,
        y,
        M,
        D,
        specialize=False,
        num_stages=3,
        num_warps=2
    )
    return x


def test_dummy(M=4096, D=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.ones(M, D, dtype=dtype, requires_grad=False, device=device)
    y = torch.ones(M, D, dtype=dtype, requires_grad=False, device=device)

    triton_dummy(x, y)

    if bench:
        benchmark_func(triton_dummy, x, y,
                       ref_bytes=M * D * 4,
                       n_profile=0,
                       trace_dir='/tmp/org.json')

    time.sleep(1)

    triton_fast_dummy(x, y)
    if bench:
        benchmark_func(triton_fast_dummy, x, y,
                       ref_bytes=M * D * 4,
                       n_profile=0,
                       trace_dir='/tmp/opt.json')


if __name__ == '__main__':
    # torchrun --nproc_per_node=2 dummy.py
    test_dummy(M=128, D=128, bench=True)