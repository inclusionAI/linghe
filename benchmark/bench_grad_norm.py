import random

import torch
from transformer_engine.pytorch.optimizers import multi_tensor_applier, \
    multi_tensor_l2norm

from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.utils.reduce import triton_batch_norm


def bench_batch_norm(M=4096, N=2048, k=32):
    xs = [torch.randn(random.randint(M // 10, M), N, dtype=torch.float32,
                      device='cuda:0') for i in range(k)]

    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
    grad_norm_ref, _ = multi_tensor_applier(
        multi_tensor_l2norm,
        dummy_overflow_buf,
        [xs],
        False,  # no per-parameter norm
    )

    grad_norm = triton_batch_norm(xs, ord=2, norm=True)
    output_check(grad_norm_ref[0], grad_norm, 'l2_norm')

    ref_bytes = sum([x.numel() for x in xs]) * 4
    n_repeat = 100
    ref_time = benchmark_func(multi_tensor_applier,
                              multi_tensor_l2norm,
                              dummy_overflow_buf,
                              [xs],
                              False,
                              n_repeat=n_repeat,
                              ref_bytes=ref_bytes)
    benchmark_func(triton_batch_norm, xs, n_repeat=n_repeat,
                   ref_bytes=ref_bytes, ref_time=ref_time)


if __name__ == '__main__':
    bench_batch_norm(M=1024, N=2048, k=512)
