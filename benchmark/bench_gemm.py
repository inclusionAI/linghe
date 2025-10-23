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


def bench_cublas_blockwise_gemm(M=4096, N=4096, K=4096):

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


def bench_te_blockwise_gemm(M=4096, N=4096, K=4096):

    # layout == 'TN':  # forward, y=x@w
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



import torch 
import triton 
import triton.language as tl



@triton.jit
def mxfp8_quant_kernel(x_ptr,
                                        out_ptr, scale_ptr,
                                        transpose_output_ptr,
                                        transpose_scale_ptr,
                                        M,
                                        m,
                                        N: tl.constexpr,
                                        OUTPUT_MODE: tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)

    offs = rid * 32 * N + cid * 32 + tl.arange(0, 32)[:,
                                           None] * N + tl.arange(0, 32)[
                                                           None, :]
    indices = rid * 32 + tl.arange(0, 32)
    mask = indices[:, None] < m

    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    
    if OUTPUT_MODE % 2 == 0:
        scale = tl.maximum(tl.max(x.abs(), 1) / 448, 1e-30)
        log_scale = tl.ceil(tl.log2(scale))
        scale = tl.exp2(log_scale)
        b = N // 32
        tl.store(scale_ptr + rid * 32 * b + cid + tl.arange(0, 32) * b, log_scale+127,
                 mask=indices < M)
        xq = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + rid * 32 * N + cid * 32 + \
             tl.arange(0, 32)[:,None] * N + tl.arange(0,32)[None, :], xq,
             mask=mask)

    if OUTPUT_MODE > 0:
        scale = tl.maximum(tl.max(x.abs(), 0) / 448, 1e-30)
        log_scale = tl.ceil(tl.log2(scale))
        scale = tl.exp2(log_scale)
        tl.store(transpose_scale_ptr + rid * N + cid * 32 + tl.arange(0, 32),
                 log_scale + 127)
        xq = (x / scale).to(out_ptr.dtype.element_ty)
        tl.store(transpose_output_ptr + rid * 32 * N + \
             cid * 32 + tl.arange(0, 32)[:, None] * N + \
                 tl.arange(0, 32)[None, :],
                 xq, mask=mask)


def triton_mxfp8_quant(x,
                                        out=None,
                                        scale=None,
                                        output_mode=2):
    """
    fused silu and mxfp8 quantization, used in shared expert
    Args:
        x: input tensor
        output_mode: one of {0, 1, 2}
            0: only output non-transposed quantized tensor
            1: only output transposed quantized tensor
            2: output both

    Returns:
        - out: quantized tensor
        - scale: quantization scale
        - transpose_output: quantized tensor of transposed output
        - transpose_scale: quantization scale of transposed output
    """
    m, N = x.shape
    M = (m + 127) // 128 * 128
    assert N % 128 == 0  # transposed scaled should be multiplier of 128
    device = x.device
    if out is None:
        out = torch.empty((m, N), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.empty((M, N // 32), device=device,
                            dtype=torch.uint8)

    transpose_output = torch.empty((m, N), device=device,
                                   dtype=torch.float8_e4m3fn)
    transpose_scale = torch.empty((M // 32, N), device=device,
                                  dtype=torch.uint8)

    grid = (M // 32, N // 32)
    mxfp8_quant_kernel[grid](
        x,
        out,
        scale,
        transpose_output,
        transpose_scale,
        M,
        m,
        N,
        output_mode,
        num_stages=2,
        num_warps=2
    )

    return out, scale, transpose_output, transpose_scale


def bench_te_mxfp8_gemm(M=4096, N=4096, K=4096):

    # import transformer_engine_torch as tex
    # from linghe.quant.mxfp8 import triton_mxfp8_quant
    import transformer_engine as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
    from transformer_engine.pytorch.module.base import get_workspace
    from transformer_engine.pytorch.constants import TE_DType

    x = torch.randn((M,K), device='cuda:0', dtype=torch.bfloat16)
    x_q, x_scale, xt_q, xt_scale = triton_mxfp8_quant(x)

    B = MXFP8Tensor(shape=(M,K),
                                dtype=torch.bfloat16,
                                rowwise_data=x_q,
                                rowwise_scale_inv=x_scale,
                                columnwise_data=None,
                                columnwise_scale_inv=None,
                                fp8_dtype=TE_DType[torch.float8_e4m3fn],
                                quantizer=None,
                            )
    
    w = torch.randn((N,K), device='cuda:0', dtype=torch.bfloat16)
    w_q, w_scale, wt_q, wt_scale = triton_mxfp8_quant(w)

    A = MXFP8Tensor(shape=(N,K),
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

    out_ref = x@w.t() 
    error = (out-out_ref).abs().sum()/out_ref.abs().sum()
    print(error)

    ref_flops = M * N * K * 2
    ref_bytes = M * K + N * K + M * N *2 
    benchmark_func(tex.generic_gemm, *args,
                   n_repeat=100, ref_flops=ref_flops, ref_bytes=ref_bytes)


if __name__ == '__main__':
    # bench_cublas_blockwise_gemm(M=4096, N=4096, K=4096)
    # bench_te_blockwise_gemm(M=4096, N=4096, K=4096)
    bench_te_mxfp8_gemm(M=4096, N=4096, K=4096)
