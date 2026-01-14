# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from linghe.experimental.dla import (triton_cp_lightning_attention_forward,
                                     triton_cp_lightning_attention_backward)
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check


def torch_la(q, k, v, decay_scales, s=None, hp=False):
    dtype = q.dtype
    if hp:
        q = q.double()
        k = k.double()
        v = v.double()
        if s is not None:
            s = s.double()
    else:
        q = q.float()
        k = k.float()
        v = v.float()
        if s is not None:
            s = s.float()
    B, L, H, D = q.shape
    h = k.shape[2]
    assert L == k.shape[1]
    softmax_scale = 1.0 / math.sqrt(D)
    query = q.transpose(1, 2)  # [B, head, len, D]
    key = torch.permute(k, (0, 2, 3, 1))  # [B, head, D, len]
    value = v.transpose(1, 2)  # [B, head, len, D]
    if h != H:
        g = H // h
        key = torch.repeat_interleave(key, g, D=1)
        value = torch.repeat_interleave(value, g, D=1)

    arr = torch.arange(L, dtype=torch.float64 if hp else torch.float32,
                       device=q.device)
    decay_matrix = arr.view(-1, 1) - arr.view(1, -1)
    decay_matrix = torch.exp(-decay_scales[:, None, None] * decay_matrix[None])
    decay_matrix = torch.tril(decay_matrix, 0)

    score = torch.matmul(query, key) * softmax_scale
    score *= decay_matrix[None]
    att = torch.matmul(score, value)

    decay_arr = torch.exp(-decay_scales[:, None, None] * (arr[:, None] + 1))
    if s is not None:
        att = att + torch.matmul(query * decay_arr, s)

    att = torch.reshape(att.transpose(1, 2),
                        [B, L, H, D]).contiguous()

    decay_key = key * torch.exp(-decay_scales[:, None, None] * (L - 1 - arr))
    state = decay_key @ value
    if s is not None:
        state += s * torch.exp(-decay_scales[:, None, None])

    return att.to(dtype), state.to(torch.float32)


def rearange(x, group):
    B, L, H, D = x.shape
    group_size = group.size()
    X = torch.empty((group_size, B, L, H, D), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(X, x.detach(), group=group)
    X = torch.permute(torch.reshape(X, (group_size, B, 2, L // 2, H, D)),
                      (1, 2, 0, 3, 4, 5))
    X = torch.reshape(torch.cat([X[:, 0], torch.flip(X[:, 1], (1,))], 1),
                      (B, L * group_size, H, D))
    X = X.contiguous().requires_grad_()
    return X


def select(x, group):
    group_size = group.size()
    group_rank = group.rank()
    B, L, H, D = x.shape
    l = L // (2 * group_size)
    x1 = x[:, group_rank * l:(group_rank + 1) * l]
    x2 = x[:, (group_size * 2 - group_rank - 1) * l:(
                                                                group_size * 2 - group_rank) * l]
    return torch.cat([x1, x2], 1)


def test_dist_la(B=1, L=4096, H=16, D=128, group=None, hpc=True, digest=False,
                 coef=1.0, grad_coef=1.0, bench=False):
    group_size = group.size()
    group_rank = group.rank()

    device_module = torch.get_device_module("cuda")
    device_module.set_device(torch.device(f'cuda:{group_rank}'))

    device = torch.device('cuda')
    dtype = torch.bfloat16

    buffers = symm_mem.empty((B, H, 2, D, D), dtype=torch.float32,
                             device=device)
    hdl = symm_mem.rendezvous(buffers, group)

    q = torch.randn(B, L, H, D, dtype=dtype, device=device) * coef
    q = q.requires_grad_()

    k = torch.randn(B, L, H, D, dtype=dtype, device=device) * coef
    k = k.requires_grad_()

    v = torch.randn(B, L, H, D, dtype=dtype, device=device) * coef
    v = v.requires_grad_()

    g = torch.randn(B, L, H, D, dtype=dtype, device=device) * grad_coef

    decay_scales = 2 ** (-0.5 * torch.arange(1, H + 1, dtype=torch.float32,
                                             device=device))
    # decay_scales = 0.0 * torch.arange(1, H+1, dtype=torch.float32, device=device)

    Q = rearange(q.detach(), group).requires_grad_()
    K = rearange(k.detach(), group).requires_grad_()
    V = rearange(v.detach(), group).requires_grad_()
    G = rearange(g, group)

    global_output_ref, global_state_ref = torch_la(Q, K, V, decay_scales,
                                                   hp=False)
    global_output_ref.backward(G)
    DQ_ref = Q.grad
    DK_ref = K.grad
    DV_ref = V.grad
    Q.grad = None
    K.grad = None
    V.grad = None
    output_ref = select(global_output_ref, group)
    dq_ref = select(DQ_ref, group)
    dk_ref = select(DK_ref, group)
    dv_ref = select(DV_ref, group)

    output, state = triton_cp_lightning_attention_forward(q, k, v, decay_scales,
                                                          hdl, group, hpc=hpc)
    output_check(output_ref, output, atol=-0.2, rtol=0.05,
                 name=f'output:{group_rank}')

    dq, dk, dv = triton_cp_lightning_attention_backward(g, q, k, v, state,
                                                        decay_scales, hdl,
                                                        group, hpc=hpc)
    output_check(dq_ref, dq, name='dq', rtol=-0.1, atol=1.0)
    output_check(dk_ref, dk, name='dk', rtol=-0.1, atol=1.0)
    output_check(dv_ref, dv, name='dv', rtol=-0.1, atol=1.0)

    if bench:
        ref_bytes = (B * L * H * D * 8 + B * H * D * D * 8) * group_size
        benchmark_func(torch_la, Q, K, V, decay_scales, ref_bytes=ref_bytes)
        benchmark_func(triton_cp_lightning_attention_forward, q, k, v,
                       decay_scales, hdl, group, hpc=hpc, ref_bytes=ref_bytes)


if __name__ == '__main__':
    # torchrun --nproc_per_node=2 test_dla.py
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ['TORCH_NCCL_AVOID_RECORD_STREAMS'] = '1'
    print(f'{world_size=} {local_rank=}')
    dist.init_process_group(backend='nccl', init_method="env://",
                            world_size=world_size, rank=local_rank,
                            timeout=timedelta(seconds=10))
    group = dist.distributed_c10d._get_default_group()
    torch.distributed.distributed_c10d._set_pg_timeout(timedelta(seconds=10),
                                                       dist.group.WORLD)
    test_dist_la(B=1, L=4096, H=64, D=128, group=group, hpc=True, digest=False,
                 bench=False)
    # test_dist_la(B=1, L=4096, H=64, D=128, group=group, hpc=True, digest=False, bench=True)
