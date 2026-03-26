
import torch
from linghe.infer.norm import triton_rms_norm_and_block_quant
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.tools.util import torch_group_quant


def torch_residual_rms_and_block_quant_forward(x, weight, residual=None, round_scale=False):
    orig_dtype = x.dtype
    x = x.float()
    weight = weight.float()
    if residual is not None:
        x = x + residual.float()
        residual = x.to(orig_dtype)

    N = x.shape[-1]
    rmsnorm = torch.nn.RMSNorm(
        normalized_shape=N,
        eps=1e-6,
        dtype=torch.float32,
        device=x.device
    )
    with torch.no_grad():
        rmsnorm.weight.copy_(weight)
    y = rmsnorm(x)
    # blockwise
    y_q, y_scale = torch_group_quant(y, round_scale=round_scale)
    return y_q, y_scale, residual

def test_rmsnorm_and_block_quant_infer(M=4096, N=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    x = torch.randn(M, N, dtype=dtype, requires_grad=False, device=device) ** 2
    weight = torch.randn(N, dtype=dtype, requires_grad=False, device=device)
    residual = torch.randn(M, N, dtype=dtype, requires_grad=False, device=device)


    # blockwise wo residual
    q_ref, scale_ref, _ = torch_residual_rms_and_block_quant_forward(x,
                                                                    weight,
                                                                    round_scale=False)

    q, scale, _ = triton_rms_norm_and_block_quant(x, weight,
                                                                  round_scale=False)
    output_check(q_ref, q, name="0.block.wo_residual.data", rtol=0.125)
    output_check(scale_ref, scale, name='0.block.wo_residual.scale')

    # blockwise with residual
    q_ref, scale_ref, ro_ref = torch_residual_rms_and_block_quant_forward(x,
                                                                    weight,
                                                                    residual,
                                                                    round_scale=False)

    q, scale, ro = triton_rms_norm_and_block_quant(x,
                                                                  weight,
                                                                  residual=residual,
                                                                  round_scale=False)
    output_check(q_ref, q, name="1.block.with_residual.data", rtol=0.125)
    output_check(scale_ref, scale, name='1.block.with_residual.scale')
    output_check(ro_ref, ro, name='1.block.with_residual.residual')


    if bench:
        benchmark_func(triton_rms_norm_and_block_quant, x, weight,
                       round_scale=False,
                       ref_bytes=M * N * 3)
        benchmark_func(triton_rms_norm_and_block_quant, x, weight, residual=residual,
                round_scale=False,
                ref_bytes=M * N * 3)

             
if __name__ == '__main__':
    test_rmsnorm_and_block_quant_infer(M=4096, N=2048, bench=True)
