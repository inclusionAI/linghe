# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.utils.unary import triton_calculate_smooth_scale
from linghe.tools.benchmark import benchmark_func
from linghe.tools.util import output_check


def torch_calculate_smooth_scale(x, min_value=1.0, smooth_coef=0.5):
    one = torch.ones([1], dtype=torch.float32, device=x.device)
    input_smooth_scales = torch.pow(torch.maximum(x, min_value*one), smooth_coef)
    weight_smooth_scales = 1/input_smooth_scales
    weight_smooth_scales = torch.exp2(torch.ceil(torch.log2(weight_smooth_scales)))
    return weight_smooth_scales


def test_calculate_smooth_scale(N=4096, bench=False):

    x = torch.randn(N, dtype=torch.float32, device='cuda:0').abs()**3+0.1

    min_value = 0.0
    smooth_coef = 0.7
    out_ref = torch_calculate_smooth_scale(x, min_value=min_value, smooth_coef=smooth_coef)
    out = triton_calculate_smooth_scale(x, min_value=min_value, smooth_coef=smooth_coef)
    output_check(out_ref, out, 'torch_calculate_smooth_scale')

    n_repeat = 100

    if bench:
        ref_time = benchmark_func(torch_calculate_smooth_scale, x,
                                  n_repeat=n_repeat)
        benchmark_func(torch_calculate_smooth_scale, x,  n_repeat=n_repeat,
                       ref_time=ref_time, ref_bytes=N * 8)

if __name__ == '__main__':
    test_calculate_smooth_scale(N=4096*32)
    test_calculate_smooth_scale(N=4096*32-1897)
