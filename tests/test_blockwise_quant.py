import torch

from linghe.quant.block import (
    triton_block_quant,
    triton_blockwise_quant,
    triton_batch_blockwise_quant,
)
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.tools.util import (
    torch_block_quant,
    torch_blockwise_quant,
    torch_make_indices,
)


def torch_batch_blockwise_quant(x, token_count_per_expert_list, round_scale=True):
    M, DIM = x.shape
    q_refs = []
    s_refs = []
    qt_refs = []
    st_refs = []
    s = 0
    for i, c in enumerate(token_count_per_expert_list):
        c = token_count_per_expert_list[i]
        if c == 0:
            continue
        y = x[s : s + c]
        y = y.float()

        y_q, y_scale, yt_q, yt_scale = torch_blockwise_quant(
            y, round_scale=round_scale, padding=False
        )
        q_refs.append(y_q.view(-1))
        s_refs.append(y_scale.view(-1))
        qt_refs.append(yt_q.view(-1))
        st_refs.append(yt_scale.view(-1))
        s += c
    q_ref = torch.cat(q_refs, 0)
    s_ref = torch.cat(s_refs, 0)
    qt_ref = torch.cat(qt_refs, 0)
    st_ref = torch.cat(st_refs, 0)
    return q_ref, s_ref, qt_ref, st_ref


def test_block_quant(M=8192, N=4096, bench=False):
    device = "cuda:0"
    x = torch.randn((M, N), dtype=torch.bfloat16, device=device) ** 3

    x_q_ref, x_s_ref = torch_block_quant(x, round_scale=True)
    x_q, x_s = triton_block_quant(x, round_scale=True)
    output_check(x_q_ref.float(), x_q.float(), "data")
    output_check(x_s_ref.float(), x_s.float(), "scale")

    if bench:
        benchmark_func(triton_block_quant, x, round_scale=True, ref_bytes=M * N * 4)


def test_blockwise_quant(M=8192, N=4096, bench=False):
    device = "cuda:0"
    x = torch.randn((M, N), dtype=torch.bfloat16, device=device) ** 3

    x_q_ref, x_s_ref, xt_q_ref, xt_s_ref = torch_blockwise_quant(
        x, round_scale=True, padding=False
    )
    x_q, x_s, xt_q, xt_s = triton_blockwise_quant(x, round_scale=True)
    output_check(x_q_ref.float(), x_q.float(), "data")
    output_check(x_s_ref.float(), x_s.float(), "scale")
    output_check(xt_q_ref.float(), xt_q.float(), "t.data")
    output_check(xt_s_ref.float(), xt_s.float(), "t.scale")

    if bench:
        benchmark_func(triton_blockwise_quant, x, round_scale=True, ref_bytes=M * N * 4)


def test_batch_block_quant(M=16384, N=2048, n_experts=32, topk=2, bench=False):
    device = "cuda:0"
    logits = torch.randn((M, n_experts), dtype=torch.float32, device=device) ** 3
    logits[:, 0] -= 1000
    logits[:, 2] -= 100
    probs, mask_map, token_count_per_expert, indices, row_id_map = torch_make_indices(
        logits, topk=topk, bias=-0.01
    )
    token_count_per_expert_list = token_count_per_expert.tolist()

    x = torch.randn((M, N), dtype=torch.bfloat16, device=device)
    x = x[indices]

    x_q_ref, x_s_ref, xt_q_ref, xt_s_ref = torch_batch_blockwise_quant(
        x, token_count_per_expert_list, round_scale=True
    )

    x_q, x_s, xt_q, xt_s = triton_batch_blockwise_quant(
        x, token_count_per_expert, token_count_per_expert_list, round_scale=True
    )
    output_check(x_q_ref.float(), x_q.view(-1).float(), "data")
    output_check(x_s_ref.float(), x_s.view(-1).float(), "scale")
    output_check(xt_q_ref.float(), xt_q.view(-1).float(), "t.data")
    output_check(xt_s_ref.float(), xt_s.view(-1).float(), "t.scale")

    if bench:
        benchmark_func(
            triton_batch_blockwise_quant,
            x,
            token_count_per_expert,
            token_count_per_expert_list,
            round_scale=True,
            ref_bytes=M * N * 4,
        )


if __name__ == "__main__":
    test_block_quant(M=8192, N=4096, bench=False)
    test_blockwise_quant(M=8192, N=4096, bench=False)
    test_batch_block_quant(M=16384, N=2048, n_experts=32, topk=2, bench=False)
