import torch
import transformer_engine as te
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import \
    Float8BlockwiseQTensor, Float8BlockQuantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor, \
    MXFP8Quantizer

from linghe.facade.norm import rms_norm, block_rms_norm, mxfp8_rms_norm
from linghe.tools.benchmark import benchmark_func
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def bench_rmsnorm_for_shape(bs=1, M=4096, N=4096, n_repeat=100):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.randn(bs, M, N, dtype=dtype, requires_grad=True, device=device)
    weight = torch.randn(N, dtype=dtype, requires_grad=True, device=device)
    dy = torch.randn(bs, M, N, dtype=dtype, device=device)

    rmsnorm_torch = torch.nn.RMSNorm(
        normalized_shape=N,
        eps=1e-6,
        dtype=torch.bfloat16,
        device='cuda'
    )
    rmsnorm_torch = torch.compile(rmsnorm_torch)
    rmsnorm_te = te.pytorch.RMSNorm(normalized_shape=N, eps=1e-6)

    def torch_forward_backward(x_torch_back, dy):
        y_torch_back = rmsnorm_torch(x_torch_back)
        y_torch_back.backward(gradient=dy)
        return x_torch_back.grad, rmsnorm_torch.weight.grad

    def te_forward_backward(x_te_back, dy):
        y_te_back = rmsnorm_te(x_te_back)
        y_te_back.backward(gradient=dy)
        return x_te_back.grad, rmsnorm_te.weight.grad

    def triton_forward_backward(x_triton_back, g_triton_back, dy):
        y_triton_back = rms_norm(x_triton_back, g_triton_back)
        y_triton_back.backward(gradient=dy)
        return x_triton_back.grad, g_triton_back.grad

    results = {}
    
    print(f"\nBenchmarking shape: bs={bs}, M={M}, N={N}")
    ref_time = benchmark_func(rmsnorm_torch, x, n_repeat=n_repeat,
                              name="rms_torch", ref_bytes=M * N * 4)
    results["torch_forward"] = ref_time
    
    te_time = benchmark_func(rmsnorm_te, x, n_repeat=n_repeat, ref_bytes=M * N * 4,
                   name="rms_te", ref_time=ref_time)
    results["te_forward"] = te_time
    
    triton_time = benchmark_func(rms_norm, x, weight, n_repeat=n_repeat,
                   ref_bytes=M * N * 4, name="rms_triton", ref_time=ref_time)
    results["linghe_forward"] = triton_time


    ref_bwd_time = benchmark_func(torch_forward_backward, x, dy, n_repeat=n_repeat)
    results["torch_f+b"] = ref_bwd_time

    te_bwd_time = benchmark_func(te_forward_backward, x, dy, n_repeat=n_repeat,
                   ref_time=ref_bwd_time)
    results["te_f+b"] = te_bwd_time

    triton_bwd_time = benchmark_func(triton_forward_backward, x, weight, dy, n_repeat=n_repeat,
                   ref_time=ref_bwd_time)
    results["linghe_f+b"] = triton_bwd_time

    # rmsnorm + fp8 blockwise 
    quantizer = Float8BlockQuantizer(TE_DType[torch.float8_e4m3fn],
                                     rowwise=True,
                                     columnwise=True, amax_epsilon=0,
                                     force_pow_2_scales=True,
                                     block_scaling_dim=1)
    f8_time = benchmark_func(block_rms_norm, x, weight, None, quantizer,
                   Float8BlockwiseQTensor, is_recomputing=None,
                   n_repeat=n_repeat, ref_bytes=M * N * 4, name="rms_f8_blockwise",
                   ref_time=ref_time)
    results["f8_blockwise_forward"] = f8_time

    # rmsnorm + MXFP8 
    if torch.cuda.get_device_properties(0).major > 9:
        quantizer = MXFP8Quantizer(fp8_dtype=TE_DType[torch.float8_e4m3fn])
        mxfp8_time = benchmark_func(mxfp8_rms_norm, x, weight, None, quantizer, MXFP8Tensor,
                    is_recomputing=None,
                    n_repeat=n_repeat, ref_bytes=M * N * 4, name="rms_mxfp8",
                    ref_time=ref_time)
        # results["mxfp8_forward"] = mxfp8_time

    return results


def create_benchmark_charts(results_dict):
    
    shapes = list(results_dict.keys())
    
    forward_methods = ['torch_forward', 'te_forward', 'linghe_forward', 
                      'f8_blockwise_forward']
    
    forward_backward_methods = ['torch_f+b', 'te_f+b', 'linghe_f+b']
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    x_pos = np.arange(len(shapes))
    width = 0.15
    
    for i, method in enumerate(forward_methods):
        times = [results_dict[shape].get(method, 0) for shape in shapes]
        plt.bar(x_pos + i * width, times, width, label=method.replace('_forward', ''))
    
    plt.xlabel('Shape (bs, M, N)')
    plt.ylabel('Time (ms)')
    plt.title('RMSNorm Forward Pass Benchmark')
    plt.xticks(x_pos + width * 2, shapes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    for i, method in enumerate(forward_backward_methods):
        times = [results_dict[shape].get(method, 0) for shape in shapes]
        plt.bar(x_pos + i * width, times, width, label=method.replace('_backward', ''))
    
    plt.xlabel('Shape (bs, M, N)')
    plt.ylabel('Time (ms)')
    plt.title('RMSNorm F+B Pass Benchmark')
    plt.xticks(x_pos + width, shapes)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rmsnorm_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*80)
    print("RMSNorm Benchmark Results Summary")
    print("="*80)
    
    print("\nForward Pass Performance (ms):")
    forward_df = pd.DataFrame(index=shapes, columns=[m.replace('_forward', '') for m in forward_methods])
    for shape in shapes:
        for method in forward_methods:
            forward_df.loc[shape, method.replace('_forward', '')] = results_dict[shape].get(method, 'N/A')
    print(forward_df)
    
    print("\nF+B Pass Performance (ms):")
    backward_df = pd.DataFrame(index=shapes, columns=[m.replace('_backward', '') for m in forward_backward_methods])
    for shape in shapes:
        for method in forward_backward_methods:
            backward_df.loc[shape, method.replace('_backward', '')] = results_dict[shape].get(method, 'N/A')
    print(backward_df)


def main():

    shapes = [
        (1, 8192, 4096),  # bs, M, N
        (1, 4096, 6144),
        (1, 4096, 4096),
        (1, 4096, 8192),
    ]
    
    results_dict = {}
    
    print("RMSNorm Benchmark across different shapes")
    
    for i, (bs, M, N) in enumerate(shapes):
        shape_key = f"({bs},{M},{N})"
        print(f"\n{'='*50}")
        print(f"Testing shape {i+1}/{len(shapes)}: {shape_key}")
        print(f"{'='*50}")
        
        try:
            results = bench_rmsnorm_for_shape(bs=bs, M=M, N=N, n_repeat=50)
            results_dict[shape_key] = results
        except Exception as e:
            print(f"Error testing shape {shape_key}: {e}")
            results_dict[shape_key] = {}
    
    create_benchmark_charts(results_dict)
    
    print("\nBenchmark completed!")


if __name__ == '__main__':
    main()
