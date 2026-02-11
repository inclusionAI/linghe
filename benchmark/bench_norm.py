import torch
import transformer_engine as te
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import (
    Float8BlockwiseQTensor,
    Float8BlockQuantizer,
)
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor, MXFP8Quantizer

from linghe.facade.norm import rms_norm, block_rms_norm, mxfp8_rms_norm
from linghe.tools.benchmark import benchmark_func


def bench_rmsnorm(bs=1, M=4096, N=4096):
    # M, N, K = 8192, 4096, 13312
    # M, N, K = 4096, 4096, 6144
    # M, N, K = 4096, 4096, 4096
    # M, N, K = 4096, 8192, 4096

    dtype = torch.bfloat16
    device = "cuda:0"
    n_repeat = 100

    x = torch.randn(bs, M, N, dtype=dtype, requires_grad=True, device=device)
    weight = torch.randn(N, dtype=dtype, requires_grad=True, device=device)
    dy = torch.randn(bs, M, N, dtype=dtype, device=device)

    rmsnorm_torch = torch.nn.RMSNorm(
        normalized_shape=N, eps=1e-6, dtype=torch.bfloat16, device="cuda"
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

    ref_time = benchmark_func(
        rmsnorm_torch, x, n_repeat=n_repeat, name="rms_torch", ref_bytes=M * N * 4
    )
    benchmark_func(
        rmsnorm_te,
        x,
        n_repeat=n_repeat,
        ref_bytes=M * N * 4,
        name="rms_te",
        ref_time=ref_time,
    )
    benchmark_func(
        rms_norm,
        x,
        weight,
        n_repeat=n_repeat,
        ref_bytes=M * N * 4,
        name="rms_triton",
        ref_time=ref_time,
    )

    quantizer = Float8BlockQuantizer(
        TE_DType[torch.float8_e4m3fn],
        rowwise=True,
        columnwise=True,
        amax_epsilon=0,
        force_pow_2_scales=True,
        block_scaling_dim=1,
    )
    y = block_rms_norm(
        x, weight, None, quantizer, Float8BlockwiseQTensor, is_recomputing=None
    )
    y[0].backward(dy)
    benchmark_func(
        block_rms_norm,
        x,
        weight,
        None,
        quantizer,
        Float8BlockwiseQTensor,
        is_recomputing=None,
        n_repeat=n_repeat,
        ref_bytes=M * N * 4,
        name="rms_triton",
        ref_time=ref_time,
    )

    quantizer = MXFP8Quantizer(fp8_dtype=TE_DType[torch.float8_e4m3fn])
    y = mxfp8_rms_norm(x, weight, None, quantizer, MXFP8Tensor, is_recomputing=None)
    y[0].backward(dy)
    benchmark_func(
        mxfp8_rms_norm,
        x,
        weight,
        None,
        quantizer,
        MXFP8Tensor,
        is_recomputing=None,
        n_repeat=n_repeat,
        ref_bytes=M * N * 4,
        name="rms_triton",
        ref_time=ref_time,
    )

    ref_time = benchmark_func(torch_forward_backward, x, dy, n_repeat=n_repeat)

    benchmark_func(te_forward_backward, x, dy, n_repeat=n_repeat, ref_time=ref_time)

    benchmark_func(
        triton_forward_backward, x, weight, dy, n_repeat=n_repeat, ref_time=ref_time
    )


if __name__ == "__main__":
    bench_rmsnorm(1, 4096, 4096)
