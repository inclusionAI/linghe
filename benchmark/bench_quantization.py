import random

import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import \
    Float8BlockQuantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

from linghe.quant.block import triton_block_quant, triton_blockwise_quant
from linghe.quant.mxfp8 import triton_batch_mxfp8_quant
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check


def bench_blockwise_quantization(M=8192, N=4096, round_scale=True):
    quantizer = Float8BlockQuantizer(TE_DType[torch.float8_e4m3fn],
                                     rowwise=True,
                                     columnwise=True, amax_epsilon=0,
                                     force_pow_2_scales=round_scale,
                                     block_scaling_dim=1)
    dtype = torch.bfloat16
    device = 'cuda:0'
    x = torch.randn((M, N), device=device, dtype=dtype)
    x[:, -2:] = 0.0

    qx = quantizer.make_empty((M, N), dtype=dtype, device=device,
                              requires_grad=False)
    qx = quantizer.update_quantized(x, qx)
    xq_ref = qx._rowwise_data.view(torch.float8_e4m3fn)
    xs_ref = qx._rowwise_scale_inv
    xt_q_ref = qx._columnwise_data.view(torch.float8_e4m3fn)
    xt_s_ref = qx._columnwise_scale_inv
    xq, xs, xt_q, xt_s = triton_blockwise_quant(x, round_scale=round_scale)

    output_check(xq_ref, xq, 'x.data')
    output_check(xs_ref, xs, 'x.scale')
    output_check(xt_q_ref, xt_q, 'xt.data')
    output_check(xt_s_ref, xt_s, 'xt.scale')


def bench_block_quantization(M=8192, N=4096, round_scale=True):
    dtype = torch.bfloat16
    device = 'cuda:0'
    weight_quantizer = Float8BlockQuantizer(TE_DType[torch.float8_e4m3fn],
                                            rowwise=True,
                                            columnwise=True, amax_epsilon=0,
                                            force_pow_2_scales=round_scale,
                                            block_scaling_dim=2)
    w = torch.randn((N, N), device=device, dtype=dtype)
    qw = weight_quantizer.make_empty((N, N), dtype=dtype, device=device,
                                     requires_grad=False)
    qw = weight_quantizer.update_quantized(w, qw)
    wq_ref = qw._rowwise_data.view(torch.float8_e4m3fn)
    ws_ref = qw._rowwise_scale_inv
    wq, ws = triton_block_quant(w, round_scale=round_scale)

    output_check(wq_ref, wq, 'w.data')
    output_check(ws_ref, ws, 'w.scale')


def bench_batch_mxfp8_quant(M=4096, N=4096, n_experts=32, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    splits = [max(random.randint(M - 256, M + 256), 0) for x in
              range(n_experts)]
    splits = [(x + 32) // 32 * 32 for x in splits]
    print(sum(splits))
    token_count_per_expert = torch.tensor(splits, device=device)
    quantizers = [
        MXFP8Quantizer(
            fp8_dtype=tex.DType.kFloat8E4M3
        )
        for _ in range(len(splits))
    ]

    x = torch.randn((sum(splits), N), dtype=dtype, device=device)

    inputmats = tex.split_quantize(x, splits, quantizers)

    x_q, x_scale, xt_q, xt_scale = triton_batch_mxfp8_quant(x,
                                                            token_count_per_expert,
                                                            splits,
                                                            output_mode=2)

    # output_check(x_q_ref, x_q, 'x_q')
    # output_check(x_scale_ref, x_scale, 'x_scale')
    # output_check(xt_q_ref, xt_q, 'xt_q')
    # output_check(xt_scale_ref, xt_scale, 'xt_scale')

    if bench:
        ref_bytes = M * N * n_experts * 4
        benchmark_func(tex.split_quantize, x, splits, quantizers)
        benchmark_func(triton_batch_mxfp8_quant, x, token_count_per_expert,
                       splits, output_mode=2, ref_bytes=ref_bytes)


if __name__ == '__main__':
    bench_blockwise_quantization(M=8192, N=4096, round_scale=True)
    # bench_blockwise_quantization(M=8192, N=4096, round_scale=False)
    # bench_blockwise_quantization(M=16, N=4096, round_scale=True)
    # bench_blockwise_quantization(M=16, N=4096, round_scale=False)
    # bench_block_quantization(M=128, N=4096, round_scale=True)
    # bench_block_quantization(M=128, N=4096, round_scale=False)
    # bench_batch_mxfp8_quant(M=4096, N=2048, n_experts=32, bench=False)
    # bench_batch_mxfp8_quant(M=4096, N=2048, n_experts=32, bench=True)
