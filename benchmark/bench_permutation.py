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

def test_batch_mxfp8_sort_with_indices(N=1536):
    # generate_row_id + sort + pad + quant
    device = 'cuda:0'
    import transformer_engine.pytorch.triton.permutation as triton_permutation
    x = None

    num_global_tokens_per_local_expert = torch.tensor(
        [
            [4525, 4427, 3276, 4024, 3826, 4012, 4298, 3438],
            [5014, 4114, 3877, 4015, 4355, 4121, 4083, 3680],
            [4930, 4496, 4060, 4492, 4386, 3628, 3729, 4043],
            [4454, 4218, 4002, 4245, 4145, 4233, 4233, 3955],
        ],
        dtype=torch.int32,
        device="cuda:0",
    )

    split_sizes = torch.tensor([4525, 4427, 3276, 4024, 3826, 4012, 4298, 3438, 5014, 4114, 3877, 4015,
        4355, 4121, 4083, 3680, 4930, 4496, 4060, 4492, 4386, 3628, 3729, 4043,
        4454, 4218, 4002, 4245, 4145, 4233, 4233, 3955], dtype=torch.int32, device='cuda:0')
    
    if x == None:
        x = torch.randn((num_global_tokens_per_local_expert.sum(), N), dtype=torch.bfloat16, device=device)
    probs = torch.randn((num_global_tokens_per_local_expert.sum(), ), dtype=torch.bfloat16, device=device)
    # probs = torch.arange(num_global_tokens_per_local_expert.sum(), dtype=torch.bfloat16, device=device)

    input_chunk_idxs = torch.arange(
            total_expert, device=device
    )
    sort_input_by_local_experts = input_chunk_idxs.reshape(
        -1, num_local_experts
    ).T.ravel()
    restore_output_by_local_experts = input_chunk_idxs.reshape(
        num_local_experts, -1
    ).T.ravel()

    # print(sort_input_by_local_experts)
    # print(restore_output_by_local_experts)

    '''
    te impl 
    '''

    row_id_map = triton_permutation.make_chunk_sort_map(
        num_global_tokens_per_local_expert.ravel(),
        sort_input_by_local_experts,
        x.size(0),
        sort_input_by_local_experts.size(0),
    )
    # print(row_id_map)

    ### te sort ###
    output, permuted_probs = triton_permutation.sort_chunks_by_map(
        x,
        row_id_map,
        probs,
        x.size(0),
        x.size(1),
        is_forward=True,
    )
    # print(output)

    ### te padding ###
    token_per_expert = num_global_tokens_per_local_expert.sum(0).tolist()
    padded_token_per_expert = [(m + 31) // 32 * 32 for m in token_per_expert]
    te_pad_out = torch.empty([sum(padded_token_per_expert), output.shape[-1]], dtype=torch.bfloat16, device=device)
    tex.fused_multi_row_padding(output.view(-1, output.shape[-1]), te_pad_out, token_per_expert, padded_token_per_expert)
    print(te_pad_out)

    ### te quantize ###
    quantizers = [
        MXFP8Quantizer(
            fp8_dtype=tex.DType.kFloat8E4M3
        )
        for _ in range(len(padded_token_per_expert))
    ]
    inputmats = tex.split_quantize(te_pad_out, padded_token_per_expert, quantizers)

    rowwise_data_list = []
    columnwise_data_list = []
    rowwise_scale_list = []
    colwise_scale_list = []
    for i, mat in enumerate(inputmats):
        rowwise_data = mat._rowwise_data  
        columnwise_data = mat._columnwise_data
        rowwise_scale = mat._rowwise_scale_inv
        colwise_scale = mat._columnwise_scale_inv
        rowwise_data_list.append(rowwise_data)
        columnwise_data_list.append(columnwise_data)
        rowwise_scale_list.append(rowwise_scale)
        colwise_scale_list.append(colwise_scale)

    te_x_q = torch.cat(rowwise_data_list, dim=0)
    te_xt_q = torch.cat(columnwise_data_list, dim=0)
    te_x_s = torch.cat(rowwise_scale_list, dim=0)
    te_xt_s = torch.cat(colwise_scale_list, dim=0)

    ### gemm pass ###

    ### te unpadding ###
    te_unpad_out = torch.empty([sum(token_per_expert), te_pad_out.shape[-1]], dtype=torch.bfloat16, device=device)
    tex.fused_multi_row_unpadding(te_pad_out.view(-1, te_pad_out.shape[-1]), te_unpad_out, padded_token_per_expert, token_per_expert)


    unsort_row_id_map = triton_permutation.make_chunk_sort_map(
        num_global_tokens_per_local_expert.T.ravel(),
        restore_output_by_local_experts,
        te_unpad_out.size(0),
        num_global_tokens_per_local_expert.T.ravel().size(0),
    )

    unsort_output, unpermuted_probs = triton_permutation.sort_chunks_by_map(
        te_unpad_out,
        unsort_row_id_map,
        permuted_probs,
        te_unpad_out.size(0),
        te_unpad_out.size(1),
        is_forward=True,
    )

    '''
    linghe impl
    '''
    linghe_row_id_map_tmp, linghe_resort_row_id_map_tmp = linghe_make_chunk_sort_map(num_global_tokens_per_local_expert, token_per_expert)
    
    all2all_split_sizes = num_global_tokens_per_local_expert.sum(0)

    x_q, x_s, xt_q, xt_s, prob_linghe = triton_batch_mxfp8_permute_with_indices(
        x, all2all_split_sizes, linghe_row_id_map, all2all_split_sizes.tolist(), probs=probs
    )

    output_check(te_x_q.view(torch.float8_e4m3fn).float(), x_q.float(), 'xq')
    output_check(te_xt_q.view(torch.float8_e4m3fn).float(), xt_q.float(), 'xt_q')
    output_check(te_x_s, x_s)
    output_check(te_xt_s, xt_s)
    # output_check(permuted_probs, prob_linghe)

    linghe_unsort_output, linghe_unsort_prob = triton_unpermute_with_row_id_map(te_pad_out, linghe_resort_row_id_map, prob_linghe)
    
    output_check(linghe_unsort_output, unsort_output, "unsort output")
    output_check(unpermuted_probs, linghe_unsort_prob, "unsort probs")

    benchmark_func(make_chunk_sort_map_pad_unified, num_global_tokens_per_local_expert, token_per_expert, n_repeat=50)
    benchmark_func(triton_unpermute_with_row_id_map, te_pad_out, linghe_resort_row_id_map, prob_linghe, n_repeat=100)



if __name__ == '__main__':
    bench_triton_permute_with_mask_map(M=8192 * 4, N=2048, n_experts=32, topk=2)
    bench_triton_unpermute_with_mask_map(M=8192 * 4, N=2048, n_experts=32,
                                         topk=2)
    bench_permute_pad_quantization(M=8192 * 4, N=4096, n_experts=32, topk=2)
