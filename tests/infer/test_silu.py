import torch
from linghe.tools.util import torch_group_quant
from linghe.infer.silu import triton_silu_and_block_quant
from linghe.tools.check import output_check
from linghe.tools.benchmark import benchmark_func


def torch_silu_and_block_quant(x, round_scale=True):
    M, N = x.shape
    x = x.float()
    x1, x2 = torch.split(x, N // 2, dim=1)
    y = torch.sigmoid(x1) * x1 * x2
    # blockwise
    y_q, y_scale = torch_group_quant(y, round_scale=round_scale)
    yt_q, yt_scale = torch_group_quant(y.t(), round_scale=round_scale)

    return y_q, y_scale, yt_q, yt_scale



def test_silu_and_block_quant(M=4096, N=4096, coef=1.0,
                              bench=False):
    x = torch.randn((M, N), dtype=torch.bfloat16, device='cuda:0')
    x = (x * coef).clone().detach()

    round_scale = False
    y_q_ref, y_scale_ref, _, _ = torch_silu_and_block_quant(
        x, round_scale=round_scale)

    y_q, y_scale = triton_silu_and_block_quant(x, round_scale=round_scale)
    output_check(y_q_ref, y_q, 'block.0.y_q', rtol=0.125)
    output_check(y_scale_ref, y_scale, 'block.0.y_scale')

    if bench:
        benchmark_func(triton_silu_and_block_quant, x,
                       round_scale=round_scale, n_repeat=100, ref_bytes=M * N * 3)


if __name__ == '__main__':
   test_silu_and_block_quant(M=4096, N=4096, coef=1.0)
   test_silu_and_block_quant(M=1, N=4096, coef=1.0)
   test_silu_and_block_quant(M=31, N=4096, coef=1.0)
