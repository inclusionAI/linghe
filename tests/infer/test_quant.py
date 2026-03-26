import torch
from linghe.tools.util import torch_group_quant
from linghe.infer.quant import triton_group_quant
from linghe.tools.check import output_check
from linghe.tools.benchmark import benchmark_func



def test_group_quant(M=4096, N=4096, coef=1.0,
                              bench=False):
    x = torch.randn((M, N), dtype=torch.bfloat16, device='cuda:0')
    x = x * coef

    round_scale = False
    y_q_ref, y_scale_ref = torch_group_quant(
        x, round_scale=round_scale)

    y_q, y_scale = triton_group_quant(x, round_scale=round_scale)
    output_check(y_q_ref, y_q, 'y_q', rtol=0.125)
    output_check(y_scale_ref, y_scale, 'y_scale')

    if bench:
        benchmark_func(triton_group_quant, x,
                       round_scale=round_scale,
                       n_repeat=100,
                       ref_bytes=M * N * 3)


if __name__ == '__main__':
   test_group_quant(M=4096, N=4096, coef=1.0)
   test_group_quant(M=1, N=4096, coef=1.0)
   test_group_quant(M=31, N=4096, coef=1.0)
