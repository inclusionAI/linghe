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


def test_cublas_blockwise_gemm(M=4096, N=4096, K=4096):

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


def test_te_blockwise_gemm(M=4096, N=4096, K=4096):

    # layout == 'TN':  # forward, y=x@w
    # x_q = B._rowwise_data
    # x_scale = B._rowwise_scale_inv
    # w_q = A._rowwise_data 
    # w_scale = A._rowwise_scale_inv

    # import transformer_engine_torch as tex
    import transformer_engine as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockwiseQTensor
    from transformer_engine.pytorch.module.base import get_workspace
    from transformer_engine.pytorch.constants import TE_DType
    row_data = torch.randn((M,K), device='cuda:0').to(torch.float8_e4m3fn)
    row_scales = torch.randn((K//128,M), device='cuda:0')
    x = Float8BlockwiseQTensor(shape=(M,K),
                                dtype=torch.bfloat16,
                                fp8_dtype=TE_DType[torch.float8_e4m3fn],
                                rowwise_data=row_data,
                                rowwise_scale_inv=row_scales,
                                columnwise_data=None,
                                columnwise_scale_inv=None,
                                quantizer=None,
                                requires_grad=False,
                                is_2D_scaled=False
                            )
    
    row_data = torch.randn((N,K), device='cuda:0').to(torch.float8_e4m3fn)
    row_scales = torch.randn((K//128,N//128), device='cuda:0')
    w = Float8BlockwiseQTensor(shape=(N,K),
                                dtype=torch.bfloat16,
                                fp8_dtype=TE_DType[torch.float8_e4m3fn],
                                rowwise_data=row_data,
                                rowwise_scale_inv=row_scales,
                                columnwise_data=None,
                                columnwise_scale_inv=None,
                                quantizer=None,
                                requires_grad=False,
                                is_2D_scaled=True
                            )
    A = w 
    transa = True 
    B = x 
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

    ref_flops = M * N * K * 2
    ref_bytes = M * K + N * K + M * N *2 
    benchmark_func(tex.generic_gemm, *args,
                   n_repeat=100, ref_flops=ref_flops, ref_bytes=ref_bytes)


if __name__ == '__main__':
    test_cublas_blockwise_gemm(M=4096, N=4096, K=4096)
    test_te_blockwise_gemm(M=4096, N=4096, K=4096)

