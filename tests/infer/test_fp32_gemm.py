import torch
from linghe.infer.fp32_gemm import triton_split_fp32_gemm, matmul_tma_persistent
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check

def torch_fp32_matmul(x, w):
    return torch.nn.functional.linear(x.to(torch.float32), w.to(torch.float32))


def torch_fp32_matmul_backward(dy, w):
    return dy @ w


M = 8192
N = 157184
K = 2048
dtype = torch.bfloat16

a = torch.randn((M, K), device="cuda", dtype=torch.bfloat16).to(dtype)
b = torch.randn((N, K), device="cuda", dtype=torch.bfloat16).to(dtype)
y_ref = torch_fp32_matmul(a, b)
y = matmul_tma_persistent(a, b)
output_check(y_ref, y, name="fp32_gemm", atol=1e-2)
ref_time = benchmark_func(torch_fp32_matmul, a, b, n_repeat=10)
benchmark_func(triton_split_fp32_gemm, a, b, n_repeat=10, ref_time=ref_time)
benchmark_func(matmul_tma_persistent, a, b, n_repeat=10, ref_time=ref_time)