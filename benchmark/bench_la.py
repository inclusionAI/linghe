import torch
from fla.ops.lightning_attn import chunk_lightning_attn

from linghe.tools.benchmark import benchmark_func


def bench_la(B=1, S=4096, H=32, D=128):
    query = torch.randn(
        B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    key = torch.randn(
        B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    value = torch.randn(
        B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    grad = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
    # decay_scales = 2**(-0.5 * torch.arange(1, H+1, dtype=torch.float32, device='cuda'))

    core_attn_out, _ = chunk_lightning_attn(
        query,
        key,
        value,
        layer_idx=1,  # not used, starts from 0.
        num_layers=20,  # not used.
        initial_state=None,
        output_final_state=True,
        cu_seqlens=None,  # for varlen training
        head_first=False,
    )

    benchmark_func(
        chunk_lightning_attn,
        query,
        key,
        value,
        layer_idx=1,
        num_layers=20,
        output_final_state=True,
    )
    benchmark_func(core_attn_out.backward, grad, retain_graph=True)


if __name__ == "__main__":
    bench_la(B=2, S=4096, H=64, D=128)
