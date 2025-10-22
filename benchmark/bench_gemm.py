# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch


from linghe.tools.benchmark import benchmark_func



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
    test_te_blockwise_gemm(M=4096, N=4096, K=4096)

