import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import make_ptr

from cutlass.base_dsl._mlir_helpers.op import dsl_user_op
from linghe.utils.silu import triton_weighted_silu_forward
from linghe.tools.benchmark import benchmark_func
from linghe.experimental.swiglu import SwiGLUKernel

def test(M=16384, N=4096, benchmark=False):
    input = torch.randn(M, N, dtype=torch.bfloat16, device='cuda')
    output = torch.empty(M, N // 2, dtype=torch.bfloat16, device='cuda')
    weight = torch.randn((M,), dtype=torch.float32, device='cuda')

    input_ptr = make_ptr(cutlass.BFloat16, input.data_ptr(),  cute.AddressSpace.gmem, assumed_align=16)
    output_ptr = make_ptr(cutlass.BFloat16, output.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    weight_ptr = make_ptr(cutlass.Float32,  weight.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    kernel = SwiGLUKernel(N)
    kernel(input_ptr, output_ptr, weight_ptr, M)

    x1, x2 = input.chunk(2, dim=1)
    output_ref = (x1 * torch.sigmoid(x1)) * x2 * weight.unsqueeze(1)
    torch.testing.assert_close(output, output_ref.to(torch.bfloat16))

    compiled_func = cute.compile(kernel, input_ptr, output_ptr, weight_ptr, M)

    if benchmark:
      benchmark_func(triton_weighted_silu_forward, input, weight,
                    n_repeat=100,
                    ref_bytes=M * N * 3)
      benchmark_func(compiled_func, input_ptr, output_ptr, weight_ptr, M,
                    n_repeat=100,
                    ref_bytes=M * N * 3)


if __name__ == "__main__":
    test(M=16384, N=4096, benchmark=False)