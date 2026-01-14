import logging
import os
import sys
import pathlib
from typing import Any, Dict, Tuple, Union

import torch

from transformer_engine.pytorch import DotProductAttention

import transformer_engine_torch as tex
from linghe.tools.benchmark import benchmark_func


class ModelConfig:
    def __init__(
            self,
            batch_size: int,
            max_seqlen_q: int,
            num_heads: int,
            head_dim_qk: int,
            max_seqlen_kv: int = None,
            num_gqa_groups: int = None,
            head_dim_v: int = None,
            softmax_type: str = "vanilla",
            dropout_p: float = 0.0,
            attn_mask_type: str = "no_mask",
            attn_bias_type: str = "no_bias",
            alibi_type: str = "none",
            bias_shape: str = "1hss",
            window_size: Tuple[int, int] = (-1, -1),
            context_parallel: bool = False,
            cp_comm_type: str = "p2p",
            return_max_logit=False,
            total_requests: int = None,
            max_ctx_len: int = None,
            num_layers: int = 1,
            eps: float = 1e-5,
    ):
        self.batch_size = batch_size
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_kv = max_seqlen_q if max_seqlen_kv is None else max_seqlen_kv
        self.num_heads = num_heads
        self.num_gqa_groups = num_heads if num_gqa_groups is None else num_gqa_groups
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_qk if head_dim_v is None else head_dim_v
        if self.head_dim_qk == self.head_dim_v:
            self.kv_channels = self.head_dim_qk
        else:
            self.kv_channels = (self.head_dim_qk, self.head_dim_v)
        self.hidden_size = self.num_heads * self.head_dim_qk
        self.hidden_size_kv = self.num_gqa_groups * self.head_dim_v
        self.softmax_type = softmax_type
        self.dropout_p = dropout_p
        self.attn_mask_type = attn_mask_type
        self.attn_bias_type = attn_bias_type
        self.alibi_type = alibi_type
        self.attn_type = "self" if (
                    self.max_seqlen_q == self.max_seqlen_kv) else "cross"
        self.bias_shape = bias_shape
        self.window_size = window_size
        self.context_parallel = context_parallel
        self.cp_comm_type = cp_comm_type
        self.return_max_logit = return_max_logit
        self.total_requests = total_requests
        self.max_ctx_len = max_ctx_len
        self.num_layers = num_layers
        self.eps = eps


def _run_dot_product_attention(
        dtype: torch.dtype,
        config,
        backend: str,
        ckpt_attn: bool,
        qkv_layout: str,
        workspace_opt: bool,
        pad_between_seqs: bool,
        is_training: bool,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Run DotProductAttention module with one forward pass and one backward pass"""
    # Set RNG and environment varables
    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        os.environ[
            "NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] = "1" if workspace_opt else "0"

    # Create seqlens
    qkv_format = "".join([i for i in qkv_layout.split("_")[0] if i.isalpha()])
    if ("padding" in config.attn_mask_type or qkv_format == "thd") and False:
        if config.attn_type == "self":
            seqlens_q = torch.randint(
                1, config.max_seqlen_q, [config.batch_size], dtype=torch.int32,
                device="cuda"
            )
            seqlens_kv = seqlens_q
        if config.attn_type == "cross":
            if config.max_seqlen_q > 1:
                seqlens_q = torch.randint(
                    1, config.max_seqlen_q, [config.batch_size],
                    dtype=torch.int32, device="cuda"
                )
            else:
                seqlens_q = torch.ones([config.batch_size], dtype=torch.int32,
                                       device="cuda")
            seqlens_kv = torch.randint(
                1, config.max_seqlen_kv, [config.batch_size], dtype=torch.int32,
                device="cuda"
            )
    else:
        seqlens_q = torch.full(
            [config.batch_size], config.max_seqlen_q, dtype=torch.int32,
            device="cuda"
        )
        seqlens_kv = torch.full(
            [config.batch_size], config.max_seqlen_kv, dtype=torch.int32,
            device="cuda"
        )
    cu_seqlens_q = torch.zeros(config.batch_size + 1, dtype=torch.int32,
                               device="cuda")
    cu_seqlens_kv = torch.zeros(config.batch_size + 1, dtype=torch.int32,
                                device="cuda")
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    cu_seqlens_kv[1:] = torch.cumsum(seqlens_kv, dim=0)

    seqlens_q_after_pad = seqlens_q.clone()
    seqlens_kv_after_pad = seqlens_kv.clone()
    cu_seqlens_q_after_pad = cu_seqlens_q.clone()
    cu_seqlens_kv_after_pad = cu_seqlens_kv.clone()
    pad_len = [0] * config.batch_size
    if pad_between_seqs:
        max_pad_len = 3
        pad_len = torch.randint(0, max_pad_len + 1, [config.batch_size],
                                device="cuda")  # 3
        seqlens_q_after_pad = seqlens_q + pad_len
        seqlens_kv_after_pad = seqlens_kv + pad_len
        cu_seqlens_q_after_pad[1:] = torch.cumsum(seqlens_q_after_pad, dim=0)
        cu_seqlens_kv_after_pad[1:] = torch.cumsum(seqlens_kv_after_pad, dim=0)

    # Create attention mask if padding
    attention_mask = None
    if "padding" in config.attn_mask_type:
        if config.attn_type == "self":
            attention_mask_q = torch.Tensor([]).to(dtype=torch.bool)
            for i in range(config.batch_size):
                attention_mask_q = torch.cat(
                    [
                        attention_mask_q,
                        torch.Tensor(
                            [False] * seqlens_q[i] + [True] * (
                                        config.max_seqlen_q - seqlens_q[i])
                        )
                        .to(dtype=torch.bool)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0),
                    ],
                    dim=0,
                )
            attention_mask = attention_mask_q.to(device="cuda")
        if config.attn_type == "cross":
            attention_mask_q = torch.Tensor([]).to(dtype=torch.bool)
            attention_mask_kv = torch.Tensor([]).to(dtype=torch.bool)
            for i in range(config.batch_size):
                attention_mask_q = torch.cat(
                    [
                        attention_mask_q,
                        torch.Tensor(
                            [False] * seqlens_q[i] + [True] * (
                                        config.max_seqlen_q - seqlens_q[i])
                        )
                        .to(dtype=torch.bool)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0),
                    ],
                    dim=0,
                )
                attention_mask_kv = torch.cat(
                    [
                        attention_mask_kv,
                        torch.Tensor(
                            [False] * seqlens_kv[i]
                            + [True] * (config.max_seqlen_kv - seqlens_kv[i])
                        )
                        .to(dtype=torch.bool)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .unsqueeze(0),
                    ],
                    dim=0,
                )
            attention_mask = (
                attention_mask_q.to(device="cuda"),
                attention_mask_kv.to(device="cuda"),
            )

    alibi_slopes = None

    # Create input tensors
    dim_to_num = {
        "b": config.batch_size,
        "sq": config.max_seqlen_q,
        "skv": config.max_seqlen_kv,
        "h": config.num_heads,
        "hg": config.num_gqa_groups,
        "dqk": config.head_dim_qk,
        "dv": config.head_dim_v,
        "t": cu_seqlens_q_after_pad[-1],
        "tg": cu_seqlens_kv_after_pad[-1],
        "3": 3,
        "2": 2,
        "1": 1,
    }
    inp = []
    inp_orig = []
    for i, layout in enumerate(qkv_layout.split("_")):
        layout = "_".join(layout)
        if i == 0:
            layout = layout.replace("s", "sq")
        else:
            layout = layout.replace("s", "skv")
            layout = layout.replace("h", "hg")
            layout = layout.replace("t", "tg")
        if i == 2:
            layout = layout.replace("d", "dv")
        else:
            layout = layout.replace("d", "dqk")
        tensor_shape = [dim_to_num[j] for j in layout.split("_")]
        tensor = 0.1 * torch.randn(tensor_shape, dtype=dtype, device="cuda")
        # tensor: with padding tokens
        # tensor_orig: without padding tokens
        tensor_orig = tensor
        if qkv_format == "thd" and pad_between_seqs:
            tensor_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
            if layout in ["t_h_dqk", "t_3_h_dqk", "t_h_3_dqk"]:
                for i in range(1, config.batch_size + 1):
                    valid_range = (
                        cu_seqlens_q_after_pad[i - 1],
                        cu_seqlens_q_after_pad[i] - pad_len[i - 1],
                    )
                    pad_range = (
                        cu_seqlens_q_after_pad[i] - pad_len[i - 1],
                        cu_seqlens_q_after_pad[i],
                    )
                    tensor[pad_range[0]: pad_range[1]] = 0.0
                    tensor_orig = torch.cat(
                        [tensor_orig, tensor[valid_range[0]: valid_range[1]]],
                        dim=0
                    )
            if layout in ["tg_hg_dqk", "tg_2_hg_dqk", "tg_hg_2_dqk",
                          "tg_hg_dv"]:
                for i in range(1, config.batch_size + 1):
                    valid_range = (
                        cu_seqlens_kv_after_pad[i - 1],
                        cu_seqlens_kv_after_pad[i] - pad_len[i - 1],
                    )
                    pad_range = (
                        cu_seqlens_kv_after_pad[i] - pad_len[i - 1],
                        cu_seqlens_kv_after_pad[i],
                    )
                    tensor[pad_range[0]: pad_range[1]] = 0.0
                    tensor_orig = torch.cat(
                        [tensor_orig, tensor[valid_range[0]: valid_range[1]]],
                        dim=0
                    )
        tensor_count = 1
        split_dim = 0
        for dim, l in enumerate(layout.split("_")):
            if l.isdigit():
                tensor_count = int(l)
                split_dim = dim
                break
        tensors = torch.split(tensor, 1, dim=split_dim) if split_dim != 0 else [
            tensor]
        tensors_orig = (
            torch.split(tensor_orig, 1, dim=split_dim) if split_dim != 0 else [
                tensor_orig]
        )
        for j in range(tensor_count):
            if split_dim != 0:
                inp.append(tensors[j].squeeze(split_dim))
                inp_orig.append(tensors_orig[j].squeeze(split_dim))
            else:
                inp.append(tensors[j])
                inp_orig.append(tensors_orig[j])
    for i in range(3):
        inp[i].requires_grad = True
        inp_orig[i].requires_grad = True

    # Create output gradient
    qkv_format_kv = "_".join(qkv_format)
    qkv_format_kv = qkv_format_kv.replace("s", "sq")
    qkv_format_kv = qkv_format_kv.replace("d", "dv")
    out_grad_shape = [dim_to_num[i] for i in qkv_format_kv.split("_")]
    out_grad_shape_new = [*out_grad_shape[:-2],
                          out_grad_shape[-2] * out_grad_shape[-1]]
    out_grad = 0.001 * torch.randint(0, 200, out_grad_shape_new, dtype=dtype,
                                     device="cuda")
    out_grad_orig = out_grad
    if qkv_format == "thd" and pad_between_seqs:
        out_grad_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
        if qkv_format_kv == "t_h_dv":
            for i in range(1, config.batch_size + 1):
                valid_range = (
                    cu_seqlens_q_after_pad[i - 1],
                    cu_seqlens_q_after_pad[i] - pad_len[i - 1],
                )
                pad_range = (cu_seqlens_q_after_pad[i] - pad_len[i - 1],
                             cu_seqlens_q_after_pad[i])
                out_grad[pad_range[0]: pad_range[1]] = 0.0
                out_grad_orig = torch.cat(
                    [out_grad_orig, out_grad[valid_range[0]: valid_range[1]]],
                    dim=0
                )

    # Create bias
    if config.attn_bias_type in ["no_bias"]:
        bias = None

    # # Create RNG
    # _DUMMY_CUDA_RNG_STATE_TRACKER = CudaRNGStatesTracker()
    # _DUMMY_CUDA_RNG_STATE_TRACKER.add("model-parallel-rng", seed)

    # def get_dummy_cuda_rng_tracker() -> CudaRNGStatesTracker:
    #     """Get cuda rng tracker."""
    #     return _DUMMY_CUDA_RNG_STATE_TRACKER

    # Set up model
    block = DotProductAttention(
        config.num_heads,
        (config.head_dim_qk, config.head_dim_v),
        num_gqa_groups=config.num_gqa_groups,
        attention_dropout=config.dropout_p,
        qkv_format=qkv_format,
        attn_mask_type=config.attn_mask_type,
        sequence_parallel=False,
        tp_size=1,
        get_rng_state_tracker=None,
        tp_group=None,
        layer_number=1,
        attention_type=config.attn_type,
        softmax_type=config.softmax_type,
        return_max_logit=config.return_max_logit,
    ).to(dtype=dtype, device="cuda")
    if not is_training:
        block = block.eval()
    if is_training and config.softmax_type != "vanilla":
        block.softmax_offset.requires_grad = True

    # Run a forward and backward pass
    if backend in ["FlashAttention", "UnfusedDotProductAttention"]:
        q = inp_orig[0]
        k = inp_orig[1]
        v = inp_orig[2]
        d_out = out_grad_orig
    if backend == "FusedAttention":
        q = inp[0]
        k = inp[1]
        v = inp[2]
        d_out = out_grad
    out = block(
        q,
        k,
        v,
        window_size=config.window_size,
        attention_mask=attention_mask,
        qkv_format=qkv_format,
        max_seqlen_q=config.max_seqlen_q,
        max_seqlen_kv=config.max_seqlen_kv,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        cu_seqlens_q_padded=cu_seqlens_q_after_pad if backend == "FusedAttention" else None,
        cu_seqlens_kv_padded=cu_seqlens_kv_after_pad if backend == "FusedAttention" else None,
        attn_mask_type=config.attn_mask_type,
        checkpoint_core_attention=ckpt_attn,
        core_attention_bias_type=config.attn_bias_type,
        core_attention_bias=bias,
        alibi_slopes=alibi_slopes,
        fast_zero_fill=True,
    )
    max_logit = None
    if config.return_max_logit:
        out, max_logit = out
    if is_training:
        out.backward(d_out)

    d_softmax_offset = None
    if is_training and config.softmax_type != "vanilla":
        d_softmax_offset = block.softmax_offset.grad

    if backend in ["FlashAttention", "UnfusedDotProductAttention"]:
        if is_training:
            return out, max_logit, (q.grad, k.grad, v.grad, d_softmax_offset)
        else:
            return out, max_logit, (None, None, None, d_softmax_offset)
    if backend == "FusedAttention":
        if qkv_format == "thd" and pad_between_seqs:
            out_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
            if is_training:
                q_grad_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
                k_grad_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
                v_grad_orig = torch.Tensor([]).to(device="cuda", dtype=dtype)
            for i in range(1, config.batch_size + 1):
                valid_range_q = (
                    cu_seqlens_q_after_pad[i - 1],
                    cu_seqlens_q_after_pad[i] - pad_len[i - 1],
                )
                valid_range_kv = (
                    cu_seqlens_kv_after_pad[i - 1],
                    cu_seqlens_kv_after_pad[i] - pad_len[i - 1],
                )
                out_orig = torch.cat(
                    [out_orig, out[valid_range_q[0]: valid_range_q[1]]], dim=0)
                if is_training:
                    q_grad_orig = torch.cat(
                        [q_grad_orig,
                         q.grad[valid_range_q[0]: valid_range_q[1]]], dim=0
                    )
                    k_grad_orig = torch.cat(
                        [k_grad_orig,
                         k.grad[valid_range_kv[0]: valid_range_kv[1]]], dim=0
                    )
                    v_grad_orig = torch.cat(
                        [v_grad_orig,
                         v.grad[valid_range_kv[0]: valid_range_kv[1]]], dim=0
                    )
            if is_training:
                return (
                    out_orig,
                    max_logit,
                    (q_grad_orig, k_grad_orig, v_grad_orig, d_softmax_offset),
                )
            else:
                return out_orig, max_logit, (None, None, None, d_softmax_offset)
        else:
            if is_training:
                return out, max_logit, (
                q.grad, k.grad, v.grad, d_softmax_offset)
            else:
                return out, max_logit, (None, None, None, d_softmax_offset)


def fused_attn(block, q, k, v, cu_seqlens_q, cu_seqlens_kv, mask):
    out = block(
        q,
        k,
        v,
        window_size=(-1, 0),
        attention_mask=mask,
        qkv_format='thd',
        max_seqlen_q=q.size(0),
        max_seqlen_kv=q.size(0),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        cu_seqlens_q_padded=cu_seqlens_q,
        cu_seqlens_kv_padded=cu_seqlens_kv,
        attn_mask_type='padding_causal',
        checkpoint_core_attention=False,
        core_attention_bias_type='no_bias',
        core_attention_bias=None,
        alibi_slopes=None,
        fast_zero_fill=True,
    )
    return out


def test_fused_attn(B=1, S=8192, H=64):
    dtype = torch.bfloat16
    config = ModelConfig(B, S, H, 192,
                         max_seqlen_kv=S, head_dim_v=128,
                         attn_mask_type='padding_causal', window_size=(-1, 0),
                         )
    backend = "FusedAttention"
    ckpt_attn = False
    qkv_layout = 'thd_thd_thd'
    workspace_opt = True
    pad_between_seqs = True
    is_training = True

    _run_dot_product_attention(dtype,
                               config,
                               backend,
                               ckpt_attn,
                               qkv_layout,
                               workspace_opt,
                               pad_between_seqs,
                               is_training,
                               )
    benchmark_func(_run_dot_product_attention,
                   dtype,
                   config,
                   backend,
                   ckpt_attn,
                   qkv_layout,
                   workspace_opt,
                   pad_between_seqs,
                   is_training,
                   n_profile=0,
                   trace_dir=None)


def bench_fused_attn(B=1, S=8192, H=64):
    dtype = torch.bfloat16
    config = ModelConfig(B, S, H, 192,
                         max_seqlen_kv=S, head_dim_v=128,
                         attn_mask_type='padding_causal', window_size=(-1, 0),
                         )
    backend = "FusedAttention"
    ckpt_attn = False
    # qkv_layout = 'sbhd_sbhd_sbhd'
    # qkv_layout = 'bshd_bshd_bshd'
    qkv_layout = 'thd_thd_thd'
    workspace_opt = True
    pad_between_seqs = True
    is_training = True

    os.environ["NVTE_FLASH_ATTN"] = "0"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    if backend == "FlashAttention":
        os.environ["NVTE_FLASH_ATTN"] = "1"
    if backend == "FusedAttention":
        os.environ["NVTE_FUSED_ATTN"] = "1"
        os.environ[
            "NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT"] = "1" if workspace_opt else "0"

    block = DotProductAttention(
        H,
        (192, 128),
        num_gqa_groups=64,
        attention_dropout=0.0,
        qkv_format='thd',
        attn_mask_type='padding_causal',
        sequence_parallel=False,
        tp_size=1,
        get_rng_state_tracker=None,
        tp_group=None,
        layer_number=1,
        attention_type='self',
        softmax_type='vanilla',
        return_max_logit=False,
    ).to(dtype=dtype, device="cuda")
    if not is_training:
        block = block.eval()
    seqlens_q = torch.full(
        [config.batch_size], config.max_seqlen_q, dtype=torch.int32,
        device="cuda"
    )
    seqlens_kv = torch.full(
        [config.batch_size], config.max_seqlen_kv, dtype=torch.int32,
        device="cuda"
    )

    cu_seqlens_q = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_kv = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_q[1:] = torch.cumsum(seqlens_q, dim=0)
    cu_seqlens_kv[1:] = torch.cumsum(seqlens_kv, dim=0)
    q = torch.randn((S, H, 192), device='cuda', dtype=dtype, requires_grad=True)
    k = torch.randn((S, H, 192), device='cuda', dtype=dtype, requires_grad=True)
    v = torch.randn((S, H, 128), device='cuda', dtype=dtype, requires_grad=True)
    g = torch.randn((S, H * 128), device='cuda', dtype=dtype)

    mask = torch.zeros((1, 1, 1, S), device='cuda', dtype=torch.bool)
    out = fused_attn(block, q, k, v, cu_seqlens_q, cu_seqlens_kv, mask)
    out.backward(g, retain_graph=True)

    benchmark_func(fused_attn,
                   block, q, k, v, cu_seqlens_q, cu_seqlens_kv, mask,
                   n_profile=1)

    benchmark_func(out.backward,
                   g, retain_graph=True,
                   n_profile=1)


if __name__ == '__main__':
    test_fused_attn(B=1, S=8192, H=64)
    bench_fused_attn(B=1, S=8192, H=64)
