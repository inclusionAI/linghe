 import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import make_ptr
from linghe.tools.check import output_check
from linghe.tools.benchmark import benchmark_func
from linghe.quant.block import triton_block_quant
from linghe.tools.util import torch_block_quant
from linghe.experimental.block_quant import BlockQuantKernel

def test_dsl_block_quant(N=16384, K=2048, B=128, benchmark=False):
    assert N % B == 0 and K % B == 0

    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

    wq_ref, scale_ref = torch_block_quant(w, round_scale=False)

    wq_out = torch.empty(N, K, dtype=torch.uint8, device="cuda")
    scale_out = torch.empty(N // B, K // B, dtype=torch.float32, device="cuda")

    w_ptr = make_ptr(
        cutlass.BFloat16,
        w.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    wq_ptr = make_ptr(
        cutlass.Uint8,
        wq_out.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    scale_ptr = make_ptr(
        cutlass.Float32,
        scale_out.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    kernel = BlockQuantKernel(K, B=B, round_scale=False)
    kernel(w_ptr, wq_ptr, scale_ptr, N)

    wq_ref_u8 = wq_ref.view(torch.uint8)

    output_check(wq_out.float(), wq_ref_u8.float(), 'wq', rtol=0.125, atol=0.125)

    compiled_func = cute.compile(kernel, w_ptr, wq_ptr, scale_ptr, N)

    if benchmark:
      benchmark_func(triton_block_quant, w, round_scale=False)
      benchmark_func(compiled_func, w_ptr, wq_ptr, scale_ptr, N)

if __name__ == "__main__":
    test_dsl_block_quant()
