import torch

from flops.utils.add import triton_inplace_add
from flops.utils.benchmark import benchmark_func
from flops.tools.util import output_check


def torch_add(x, outputs, accum=True):
    if accum:
        x += outputs
        return x
    else:
        return x.float()


def test_triton_inplace_add(M=4096, N=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    outputs = torch.randn(M, N, dtype=dtype, device=device)
    x = torch.randn(M, N, dtype=dtype, device=device)

    out = outputs.clone()
    triton_inplace_add(out, x)
    out_ref = outputs + x
    output_check(out_ref, out, 'sum')

    n_repeat = 100

    if bench:
        ref_time = benchmark_func(torch_add, x, out, accum=False,
                                  n_repeat=n_repeat)
        benchmark_func(triton_inplace_add, out, x, accum=False, n_repeat=n_repeat,
                       ref_time=ref_time, ref_bytes=M * N * 4)

        ref_time = benchmark_func(torch_add, x, out, accum=True,
                                  n_repeat=n_repeat)
        benchmark_func(triton_inplace_add, out, x, accum=True, n_repeat=n_repeat,
                       ref_time=ref_time, ref_bytes=M * N * 6)


if __name__ == '__main__':
    test_triton_inplace_add(M=4096, N=4096)
