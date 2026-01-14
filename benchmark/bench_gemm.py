# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.tools.benchmark import benchmark_func
from linghe.tools.util import fp16_forward
from linghe.utils.add import triton_inplace_add


def triton_accum_weight(x, w, out, x_scale, w_scale):
    output = torch._scaled_mm(
        x,
        w,
        scale_a=x_scale,
        scale_b=w_scale,
        out_dtype=torch.bfloat16,
        use_fast_accum=True
    )
    triton_inplace_add(out, output)
    return out


def torch_accum_weight(x, w, out, x_scale, w_scale):
    output = torch._scaled_mm(
        x,
        w,
        scale_a=x_scale,
        scale_b=w_scale,
        out_dtype=torch.bfloat16,
        use_fast_accum=True
    )
    out.add_(output)
    return out


def bench_cublas_channelwise_gemm(M=4096, N=4096, K=4096):
    dtype = torch.bfloat16
    device = 'cuda:0'
    n_repeat = 100

    x = torch.randn(M, K, dtype=dtype, device=device)
    w = torch.randn(N, K, dtype=dtype, device=device)

    xrs = x.abs().float().amax(dim=1, keepdim=True)
    wcs = w.abs().float().amax(dim=1, keepdim=True)
    x_q = (448 * x / xrs).to(torch.float8_e4m3fn)
    w_q = (448 * w / wcs).to(torch.float8_e4m3fn)
    ref_flops = M * N * K * 2
    ones = torch.ones((1,), dtype=torch.float32, device=device)

    out = torch.zeros((M, N), dtype=torch.float32, device=device)
    o = torch.empty((M, N), dtype=dtype, device=device)

    ref_time = benchmark_func(fp16_forward, x, w.t(), n_repeat=n_repeat,
                              ref_flops=ref_flops, name=f'M:{M}')
    benchmark_func(torch_accum_weight, x_q, w_q.t(), out, xrs, wcs.view(1, -1),
                   n_repeat=n_repeat, ref_flops=ref_flops, ref_time=ref_time,
                   name=f'M:{M}')
    benchmark_func(triton_accum_weight, x_q, w_q.t(), out, xrs, wcs.view(1, -1),
                   n_repeat=n_repeat, ref_flops=ref_flops, ref_time=ref_time,
                   name=f'M:{M}')


def bench_te_blockwise_gemm(M=4096, N=4096, K=4096, round_scale=False):
    # layout == 'TN':  # forward, y=x@w
    from linghe.quant.block import triton_block_quant, triton_blockwise_quant
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import \
        Float8BlockwiseQTensor, Float8BlockQuantizer
    from transformer_engine.pytorch.module.base import get_workspace
    from transformer_engine.pytorch.constants import TE_DType

    quantizer = Float8BlockQuantizer(TE_DType[torch.float8_e4m3fn],
                                     rowwise=True,
                                     columnwise=True, amax_epsilon=0,
                                     force_pow_2_scales=round_scale,
                                     block_scaling_dim=1)
    dtype = torch.bfloat16
    device = 'cuda:0'
    x = torch.randn((M, K), device=device, dtype=dtype) ** 3 * 1e-10
    x[-(M // 2):] = 0
    x[:, -(K // 2):] = 0

    weight_quantizer = Float8BlockQuantizer(TE_DType[torch.float8_e4m3fn],
                                            rowwise=True,
                                            columnwise=True, amax_epsilon=0,
                                            force_pow_2_scales=round_scale,
                                            block_scaling_dim=2)
    w = torch.randn((N, K), device=device, dtype=dtype)

    for manual in [False, True]:
        if manual:
            x_q, x_s, xt_q, xt_s = triton_blockwise_quant(x,
                                                          round_scale=round_scale)
            qx = Float8BlockwiseQTensor(shape=(M, K),
                                        dtype=torch.bfloat16,
                                        fp8_dtype=TE_DType[torch.float8_e4m3fn],
                                        rowwise_data=x_q,
                                        rowwise_scale_inv=x_s,
                                        columnwise_data=xt_q,
                                        columnwise_scale_inv=xt_s,
                                        quantizer=quantizer,
                                        requires_grad=False,
                                        is_2D_scaled=False
                                        )
            w_q, w_s = triton_block_quant(w, round_scale=round_scale)
            wt_q, wt_s = w_q.transpose(0, 1).contiguous(), w_s.transpose(0,
                                                                         1).contiguous()
            qw = Float8BlockwiseQTensor(shape=(N, K),
                                        dtype=torch.bfloat16,
                                        fp8_dtype=TE_DType[torch.float8_e4m3fn],
                                        rowwise_data=w_q,
                                        rowwise_scale_inv=w_s,
                                        columnwise_data=wt_q,
                                        columnwise_scale_inv=wt_s,
                                        quantizer=weight_quantizer,
                                        requires_grad=False,
                                        is_2D_scaled=True
                                        )
        else:
            qx = quantizer.make_empty((M, K), dtype=torch.bfloat16,
                                      device=device, requires_grad=False)
            qx = quantizer.update_quantized(x, qx)

            qw = weight_quantizer.make_empty((N, K), dtype=torch.bfloat16,
                                             device=device, requires_grad=False)
            qw = weight_quantizer.update_quantized(w, qw)

        # print(f'{qx._rowwise_data.shape=} {qx._rowwise_scale_inv.shape=} {qx._columnwise_data.shape=}  {qx._columnwise_scale_inv.shape=}')
        # print(f'{qw._rowwise_data.shape=} {qw._rowwise_scale_inv.shape=} {qw._columnwise_data.shape=}  {qw._columnwise_scale_inv.shape=}')

        A = qw
        transa = True
        B = qx
        transb = False
        # out = torch.randn( (M, N), device='cuda:0', dtype=torch.bfloat16) 
        out = None
        quantization_params = None
        out_dtype = TE_DType[torch.bfloat16]
        bias = None
        bias_dtype = TE_DType[torch.bfloat16]
        gelu = False
        gelu_in = None
        grad = False
        workspace = get_workspace()
        workspace_size = workspace.shape[0]
        accumulate = False
        use_split_accumulator = True
        args = (
            A,
            transa,  # transa
            B,
            transb,  # transb
            out,
            quantization_params,
            out_dtype,
            bias,
            bias_dtype,
            gelu,
            gelu_in,
            grad,  # grad
            workspace,
            workspace_size,
            accumulate,
            use_split_accumulator,
        )
        # kwargs = {
        #     "comm_overlap": None,
        #     "comm_type": None,
        #     "extra_output": None,
        #     "bulk_overlap": False,
        #     "alpha": 1.0,
        #     "beta": 0.0,
        # }
        out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args)

        ref_out = x @ w.t()

        rel_err = (
                              out - ref_out).abs().sum().item() / ref_out.abs().sum().item()
        print(
            f'rel:{rel_err:.6f} ref:{ref_out.abs().mean().item():.3f} out:{out.abs().mean().item():.3f}')

        ref_flops = M * N * K * 2
        ref_bytes = M * K + N * K + M * N * 2
        benchmark_func(tex.generic_gemm,
                       *args,
                       n_repeat=100,
                       ref_flops=ref_flops,
                       ref_bytes=ref_bytes)


def bench_te_mxfp8_gemm(M=4096, N=4096, K=4096):
    if torch.cuda.get_device_properties(0).major < 10:
        return

    # import transformer_engine_torch as tex
    from linghe.quant.mxfp8 import triton_mxfp8_quant
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
    from transformer_engine.pytorch.module.base import get_workspace
    from transformer_engine.pytorch.constants import TE_DType

    x = torch.randn((M, K), device='cuda:0', dtype=torch.bfloat16)
    x_q, x_scale, xt_q, xt_scale = triton_mxfp8_quant(x)

    B = MXFP8Tensor(shape=(M, K),
                    dtype=torch.bfloat16,
                    rowwise_data=x_q,
                    rowwise_scale_inv=x_scale,
                    columnwise_data=None,
                    columnwise_scale_inv=None,
                    fp8_dtype=TE_DType[torch.float8_e4m3fn],
                    quantizer=None,
                    )

    w = torch.randn((N, K), device='cuda:0', dtype=torch.bfloat16)
    w_q, w_scale, wt_q, wt_scale = triton_mxfp8_quant(w)

    A = MXFP8Tensor(shape=(N, K),
                    dtype=torch.bfloat16,
                    rowwise_data=w_q,
                    rowwise_scale_inv=w_scale,
                    columnwise_data=None,
                    columnwise_scale_inv=None,
                    fp8_dtype=TE_DType[torch.float8_e4m3fn],
                    quantizer=None,
                    )
    transa = True
    transb = False
    out = None
    quantization_params = None
    out_dtype = TE_DType[torch.bfloat16]
    bias = None
    bias_dtype = TE_DType[torch.bfloat16]
    gelu = False
    gelu_in = None
    grad = False
    workspace = get_workspace()
    workspace_size = workspace.shape[0]
    accumulate = False
    use_split_accumulator = True
    args = (
        A,
        transa,  # transa
        B,
        transb,  # transb
        out,
        quantization_params,
        out_dtype,
        bias,
        bias_dtype,
        gelu,
        gelu_in,
        grad,  # grad
        workspace,
        workspace_size,
        accumulate,
        use_split_accumulator,
    )
    out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args)

    out_ref = x @ w.t()
    error = (out - out_ref).abs().sum() / out_ref.abs().sum()
    print(error)

    ref_flops = M * N * K * 2
    ref_bytes = M * K + N * K + M * N * 2
    benchmark_func(tex.generic_gemm, *args,
                   n_repeat=100, ref_flops=ref_flops, ref_bytes=ref_bytes)


if __name__ == '__main__':
    # bench_cublas_channelwise_gemm(M=4096, N=4096, K=4096)
    bench_te_blockwise_gemm(M=128, N=128, K=128)
    bench_te_blockwise_gemm(M=4096, N=4096, K=4096)
    # bench_te_mxfp8_gemm(M=4096, N=4096, K=4096)
