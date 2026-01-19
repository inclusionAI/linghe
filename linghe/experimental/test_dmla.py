# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math
from datetime import timedelta
import os

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


from linghe.experimental.dmla import (triton_cp_mla_forward,
                                      triton_cp_mla_backward)
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check


def torch_attn(q, k, v, causal=True, mask=None, safe=True, clip_value=None, hp=False):
    dtype = q.dtype
    if hp:
        q = q.float()
        k = k.float()
        v = v.float()
    bs, q_len, q_head, q_head_dim = q.shape
    v_head_dim = v.shape[-1]
    k_head = k.shape[2]
    k_len = k.shape[1]
    if mask is None:
        if causal:
            mask = -10000 * torch.triu(
                torch.ones((q_len, k_len), dtype=q.dtype, device="cuda"),
                k_len - q_len + 1,
            )
        else:
            mask = torch.zeros((q_len, k_len), dtype=q.dtype, device="cuda")

    query = q.transpose(1, 2)
    key = torch.permute(k, (0, 2, 3, 1))
    value = v.transpose(1, 2)
    if k_head != q_head:
        g = q_head // k_head
        key = torch.repeat_interleave(key, g, dim=1)
        value = torch.repeat_interleave(value, g, dim=1)
    qk = torch.matmul(query, key)
    if clip_value is not None:
        qk = torch.clamp_max_(qk, clip_value).detach() + qk - qk.detach()
    score = qk / math.sqrt(v_head_dim) + mask
    if safe:
        max_logits = torch.amax(score, -1)
        lse = torch.sum(torch.exp(score-max_logits[:,:,:,None]), -1)
    else:
        max_logits = 0.0 * torch.amax(score, -1)
        lse = torch.sum(torch.exp(score), -1)

    prob = torch.softmax(score, dim=-1, dtype=torch.float32)
    if not hp:
        prob = prob.to(dtype)
    att = torch.matmul(prob, value)
    att = torch.reshape(att.transpose(1, 2),
                        [bs, q_len, q_head, v_head_dim]).contiguous()
    return att.to(dtype), lse, max_logits


def torch_varlen_attn(qs, ks, vs, cu_seqlens, padded_cu_seqlens=None,
                      causal=True, hp=False):
    cu_seqlens = cu_seqlens.tolist()
    if padded_cu_seqlens is not None:
        padded_cu_seqlens = padded_cu_seqlens.tolist()
    else:
        padded_cu_seqlens = None
    outputs = []
    lses = []
    logits = []
    for i in range(len(cu_seqlens) - 1):
        if padded_cu_seqlens is None:
            s = cu_seqlens[i]
            e = cu_seqlens[i + 1]
            q = qs[s:e][None]
            k = ks[s:e][None]
            v = vs[s:e][None]
        else:
            s = padded_cu_seqlens[i]
            e = s + cu_seqlens[i + 1] - cu_seqlens[i]
            q = qs[s:e][None]
            k = ks[s:e][None]
            v = vs[s:e][None]
        out, lse, logit = torch_attn(q, k, v, causal=causal, hp=hp)
        outputs.append(out[0])
        lses.append(lse[0])
        logits.append(logit[0])
        if padded_cu_seqlens is not None:
            gap = (padded_cu_seqlens[i + 1] - padded_cu_seqlens[i]) - (
                        cu_seqlens[i + 1] - cu_seqlens[i])
            outputs.append(torch.zeros_like(out[0][:gap]))
            lses.append(torch.zeros_like(lse[0][:, :gap]))
            logits.append(torch.zeros_like(logit[0][:, :gap]))

    outputs = torch.cat(outputs, 0)
    lses = torch.cat(lses, 1)
    logits = torch.cat(logits, 1)
    return outputs, lses, logits



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
    x2 = x[:, (group_size * 2 - group_rank - 1) * l:(group_size * 2 - group_rank) * l]
    return torch.cat([x1, x2], 1)

def select_stat(x, group):
    group_size = group.size()
    group_rank = group.rank()
    B, H, L = x.shape
    l = L // (2 * group_size)
    x1 = x[:, :, group_rank * l:(group_rank + 1) * l]
    x2 = x[:, :, (group_size * 2 - group_rank - 1) * l:(group_size * 2 - group_rank) * l]
    return torch.cat([x1, x2], 2)


def test_cp_mla(B=2, L=4096, H=16, group=None, causal=True, hpc=False, safe=True, coef=1.0,
             clip_value=0.0, bench=False):
    group_size = group.size()
    group_rank = group.rank()

    device_module = torch.get_device_module("cuda")
    device_module.set_device(torch.device(f'cuda:{group_rank}'))
    dtype = torch.bfloat16
    device = 'cuda'

    buffers = symm_mem.empty((B, H, L, (192 + 128) * 2), dtype=dtype,
                             device=device)
    hdl = symm_mem.rendezvous(buffers, group)

    q = (torch.randn((B, L, H, 192), device=device,
                     dtype=dtype) * coef).requires_grad_()
    k = torch.randn((B, L, H, 192), device=device, dtype=dtype)
    k[:, :, :, 128:] = k[:, :, :1, 128:]  # rope
    k = k.requires_grad_()
    v = torch.randn((B, L, H, 128), device=device, dtype=dtype,
                    requires_grad=True)
    g = torch.randn((B, L, H, 128), device=device, dtype=dtype,
                    requires_grad=True)

    Q = rearange(q.detach(), group).requires_grad_()
    K = rearange(k.detach(), group).requires_grad_()
    V = rearange(v.detach(), group).requires_grad_()
    G = rearange(g, group)

    global_output_ref, global_lse_ref, global_max_logits_ref = torch_attn(Q, K, V, causal=causal,
                                                     hp=True, safe=safe)
    global_output_ref.backward(G, retain_graph=False)
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
    lse_ref = select_stat(global_lse_ref, group)
    ml_ref = select_stat(global_max_logits_ref, group)


    output, lse, max_logits = triton_cp_mla_forward(q, k, v, hdl, group, causal=causal,
                                                 safe=safe,
                                                 clip_value=clip_value)
    output_check(output_ref, output, atol=0.05, rtol=0.05, name=f'output:{group_rank}')
    output_check(lse_ref, lse, atol=0.05, rtol=0.05, name='lse')
    output_check(ml_ref, max_logits, atol=0.01, rtol=0.03, name='max_logits')

    gq, gk, gv = triton_cp_mla_backward(g, output, q, k, v, lse, max_logits,
                                      hdl, group,
                                     causal=causal, hpc=hpc,
                                     safe=safe, clip_value=clip_value)
    # if group_rank == 1:
    #     print(f'{dv_ref[0,0,0,:8]=}')
    #     print(f'{gv[0,0,0,:8]=}')
    if clip_value == 0.0:
        output_check(dv_ref, gv, atol=-0.05, rtol=0.05, name=f'gv:{group_rank}')
        output_check(dk_ref, gk, atol=-0.05 * coef, rtol=0.05, name=f'gk:{group_rank}')
        output_check(dq_ref, gq, atol=-0.05 * coef, rtol=0.05, name=f'gq:{group_rank}')

    if bench:
        ref_flops = B * L * L * H * (192 + 128) * (1 if causal else 2) * group_size
        benchmark_func(triton_cp_mla_forward, q, k, v,  hdl, group, causal=causal, safe=safe,
                       clip_value=clip_value, ref_flops=ref_flops)
        ref_flops = B * L * L * H * (192 + 128 * 2 + 192 * 2) * (
            1 if causal else 2) * group_size
        benchmark_func(triton_cp_mla_backward, g, output, q, k, v, lse, max_logits,
                        hdl, group,
                       causal=causal, hpc=hpc, safe=safe,
                       clip_value=clip_value,
                       ref_flops=ref_flops,
                       n_profile=0)


if __name__ == '__main__':
    # torchrun --nproc_per_node=2 test_dmla.py
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
    test_cp_mla(B=2, L=4096, H=16, group=group, causal=True, hpc=False, safe=True, coef=1.0,
                clip_value=0.0, bench=True)