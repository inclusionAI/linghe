import torch
import transformer_engine.pytorch.triton.permutation as triton_permutation
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.module.fp8_padding import Fp8Padding
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import \
    Float8BlockQuantizer

from linghe.tools.benchmark import benchmark_func
from linghe.tools.util import torch_make_indices
from linghe.utils.gather import (triton_permute_with_mask_map,
                                 triton_make_row_id_map,
                                 triton_batch_block_pad_permute_with_indices,
                                 triton_make_row_id_map_and_index)
from linghe.utils.scatter import (triton_scatter_add,
                                  triton_unpermute_with_mask_map,
                                  )


def torch_index_select(y, indices):
    output = y.index_select(0, indices)
    return output


def torch_fp16_index_select(x, scales, indices):
    return x.index_select(0, indices), scales.index_select(0, indices)


def torch_fp16_scatter_add(x, outputs, indices, weights):
    if weights is not None:
        x = x * weights[:, None]
    dim = x.size(1)
    outputs.scatter_add_(0, indices.unsqueeze(1).expand(-1, dim), x)
    return outputs


def split_permute_pad_quantize(x, probs, mask_map, fp8_padding, out_tokens,
                               token_count_per_expert_list):
    M, N = x.shape
    n_experts = mask_map.size(1)
    row_id_map = triton_permutation.make_row_id_map(mask_map, M, n_experts)
    output, permuted_scale, permuted_probs = triton_permutation.permute_with_mask_map(
        x,
        row_id_map, probs, None, M,
        n_experts, out_tokens, N, 1)
    output, _ = fp8_padding(output, token_count_per_expert_list)
    permuted_probs, _ = fp8_padding(permuted_probs.view(-1, 1),
                                    token_count_per_expert_list)

    quantizer = Float8BlockQuantizer(TE_DType[torch.float8_e4m3fn],
                                     rowwise=True,
                                     columnwise=True, amax_epsilon=0,
                                     force_pow_2_scales=True,
                                     block_scaling_dim=1)

    qx = quantizer.make_empty(output.shape, dtype=x.dtype, device=x.device,
                              requires_grad=False)
    qx = quantizer.update_quantized(output, qx)

    return qx, permuted_probs


def fused_permute_pad_quantize(x, probs, mask_map, token_count_per_expert,
                               token_count_per_expert_list):
    num_out_tokens = sum(
        [(x + 15) // 16 * 16 for x in token_count_per_expert_list])
    row_id_map, pad_indices = triton_make_row_id_map_and_index(mask_map,
                                                               num_out_tokens,
                                                               multiple_of=16)
    x_q, x_s, xt_q, xt_s, p = triton_batch_block_pad_permute_with_indices(x,
                                                                          token_count_per_expert,
                                                                          pad_indices,
                                                                          token_count_per_expert_list,
                                                                          probs=probs,
                                                                          round_scale=True)
    return x_q, x_s, xt_q, xt_s, p


def bench_triton_permute_with_mask_map(M=4096, N=4096, n_experts=256, topk=8):
    device = 'cuda:0'
    dtype = torch.bfloat16
    x = torch.randn(M, N, dtype=dtype, device=device)
    scales = torch.randn(M, dtype=dtype, device=device)

    logits = torch.randn((M, n_experts), dtype=torch.float32, device=device)

    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
        logits, topk=topk, bias=0.0)
    out_tokens = sum(token_count_per_expert.tolist())

    mega_row_id_map = triton_permutation.make_row_id_map(mask_map, M, n_experts)

    n_repeat = 100
    ref_time = benchmark_func(torch_fp16_index_select, x, scales, indices,
                              n_repeat=n_repeat)
    benchmark_func(triton_permute_with_mask_map, x, scales, probs, row_id_map,
                   out_tokens, n_repeat=n_repeat, ref_time=ref_time)

    scales_m = torch.randn((M, 1), dtype=dtype, device=device)

    benchmark_func(triton_permutation.permute_with_mask_map, x,
                   mega_row_id_map, probs, scales_m, M,
                   n_experts, out_tokens, N, 1, n_repeat=n_repeat,
                   ref_time=ref_time)


def bench_permute_pad_quantization(M=4096, N=4096, n_experts=32, topk=2):
    device = 'cuda:0'
    dtype = torch.bfloat16
    x = torch.randn(M, N, dtype=dtype, device=device)
    fp8_padding = Fp8Padding(32, 16)

    logits = torch.randn((M, n_experts), dtype=torch.float32, device=device)
    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
        logits, topk=topk, bias=0.0)
    token_count_per_expert_list = token_count_per_expert.tolist()
    out_tokens = sum(token_count_per_expert_list)

    split_permute_pad_quantize(x, probs, mask_map, fp8_padding, out_tokens,
                               token_count_per_expert_list)
    fused_permute_pad_quantize(x, probs, mask_map, token_count_per_expert,
                               token_count_per_expert_list)

    ref_time = benchmark_func(split_permute_pad_quantize,
                              x, probs, mask_map, fp8_padding, out_tokens,
                              token_count_per_expert_list)
    benchmark_func(fused_permute_pad_quantize,
                   x, probs, mask_map, token_count_per_expert,
                   token_count_per_expert_list,
                   ref_time=ref_time)


def bench_triton_unpermute_with_mask_map(M=4098, N=4096, n_experts=32, topk=2):
    dtype = torch.bfloat16
    device = 'cuda:0'

    weights = torch.randn(M * topk, dtype=dtype, device=device)
    logits = torch.randn((M, n_experts), dtype=torch.float32, device=device)
    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
        logits, topk=topk, bias=0.0)

    token_count_per_expert_list = token_count_per_expert.tolist()
    out_tokens = sum(token_count_per_expert_list)

    x = torch.randn(out_tokens, N, dtype=dtype, device=device)

    outputs = torch.zeros((M, N), dtype=dtype, device=device)

    mega_row_id_map = triton_permutation.make_row_id_map(mask_map, M, n_experts)

    n_repeat = 100
    ref_time = benchmark_func(triton_scatter_add, x, outputs, indices,
                              n_repeat=n_repeat)
    benchmark_func(triton_unpermute_with_mask_map, x, row_id_map,
                   probs, n_repeat=n_repeat, ref_time=ref_time)
    benchmark_func(triton_permutation.unpermute_with_mask_map, x,
                   mega_row_id_map,
                   probs, None, M, n_experts, N)

    ref_time = benchmark_func(triton_permutation.make_row_id_map, mask_map,
                              M, n_experts, n_repeat=n_repeat)
    benchmark_func(triton_make_row_id_map, mask_map, n_repeat=n_repeat,
                   ref_time=ref_time)


if __name__ == '__main__':
    bench_triton_permute_with_mask_map(M=8192 * 4, N=2048, n_experts=32, topk=2)
    bench_triton_unpermute_with_mask_map(M=8192 * 4, N=2048, n_experts=32,
                                         topk=2)
    bench_permute_pad_quantization(M=8192 * 4, N=4096, n_experts=32, topk=2)
