# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.facade.emb import (embedding_lookup,
                               fused_accumulation_embedding_lookup)
from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.utils.emb import (triton_embedding_forward,
                              triton_embedding_backward,
                              triton_scan_and_count,
                              triton_sync_embedding_backward,
                              triton_atomic_embedding_backward
                              )


def test_scan(M=4096, bench=False):
    device = 'cuda:0'
    input_ids = torch.randint(0, 10000, (M,), dtype=torch.int32, device=device)

    sorted_ids, sorted_indices = torch.sort(input_ids, stable=False)
    unique_ids_ref, unique_counts_ref = torch.unique_consecutive(sorted_ids,
                                                                 return_counts=True)
    accum_counts_ref = torch.cumsum(
        torch.tensor([0] + unique_counts_ref.tolist(),
                     device=unique_counts_ref.device), 0)
    size = accum_counts_ref.size(0)

    accum_counts = triton_scan_and_count(sorted_ids)
    output_check(accum_counts_ref, accum_counts[:size], name='accum_counts')

    if bench:
        benchmark_func(triton_scan_and_count, sorted_ids)


def test_embedding(B=2, M=4096, V=150000, D=4096, transpose=False, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    embedding = torch.nn.Embedding(V, D, dtype=dtype, device=device)
    input_ids = torch.randint(0, V // 15, (B, M), dtype=torch.int32,
                              device=device)
    weights = embedding.weight
    grad_ref = torch.randn((V, D), dtype=dtype, device=device)
    weights.grad = grad_ref
    grad = grad_ref.clone().detach()

    y_ref = embedding(input_ids)
    if transpose:
        dy = torch.randn((M, B, D), device=device, dtype=dtype).permute(1, 0, 2)
    else:
        dy = torch.randn((B, M, D), device=device, dtype=dtype)
    y_ref.backward(dy, retain_graph=True)

    weights.grad = grad
    y = embedding_lookup(input_ids, weights)
    y.backward(dy, retain_graph=True)
    output_check(y_ref, y, name='y')
    output_check(grad_ref, grad.to(dtype), name='grad')

    if bench:
        ref_bytes = B * M * D * 4
        ref_time = benchmark_func(embedding.forward, input_ids,
                                  ref_bytes=ref_bytes)
        benchmark_func(embedding_lookup, input_ids, weights,
                       ref_time=ref_time, ref_bytes=ref_bytes)

        ref_time = benchmark_func(y_ref.backward, dy, retain_graph=True,
                                  ref_bytes=ref_bytes)
        benchmark_func(y.backward, dy, retain_graph=True,
                       ref_time=ref_time, ref_bytes=ref_bytes)


def test_fused_embedding(B=2, M=4096, V=150000, D=4096, use_main_grad=True,
                         transpose=False, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    embedding = torch.nn.Embedding(V, D, dtype=dtype, device=device)
    input_ids = torch.randint(0, V // 15, (B, M), dtype=torch.int32,
                              device=device)
    weights = embedding.weight
    main_grad = torch.randn((V, D), dtype=torch.float32, device=device)
    grad_ref = main_grad.to(dtype)
    grad =  main_grad.to(dtype)

    weights.grad = grad_ref
    y_ref = embedding(input_ids)
    if transpose:
        dy = torch.randn((M, B, D), device=device, dtype=dtype).permute(1, 0, 2)
    else:
        dy = torch.randn((B, M, D), device=device, dtype=dtype)
    y_ref.backward(dy, retain_graph=True)

    weights.grad = grad
    weights.main_grad = main_grad
    grad_name = 'main_grad' if use_main_grad else 'grad'
    y = fused_accumulation_embedding_lookup(input_ids, weights,
                                            grad_name=grad_name)
    y.backward(dy, retain_graph=True)
    output_check(y_ref, y, name='y')
    output_check(grad_ref, main_grad.to(dtype) if use_main_grad else grad, name='grad', atol=0.05)

    if bench:
        ref_bytes = B * M * D * 4
        ref_time = benchmark_func(embedding.forward, input_ids)
        benchmark_func(fused_accumulation_embedding_lookup, input_ids, weights,
                       grad_name=grad_name,
                       ref_time=ref_time, ref_bytes=ref_bytes)

        ref_time = benchmark_func(y_ref.backward, dy, retain_graph=True)
        benchmark_func(y.backward, dy, retain_graph=True,
                       ref_time=ref_time, ref_bytes=ref_bytes)
        grad_ptr = main_grad.data_ptr() if use_main_grad else grad.data_ptr()
        grad_dtype = main_grad.dtype if use_main_grad else grad.dtype
        benchmark_func(triton_atomic_embedding_backward, dy, input_ids,
                       grad_ptr, grad_dtype,
                       ref_time=ref_time, ref_bytes=ref_bytes)
        benchmark_func(triton_sync_embedding_backward, dy, input_ids,
                       grad_ptr, grad_dtype,
                       ref_time=ref_time, ref_bytes=ref_bytes)
        benchmark_func(triton_embedding_backward, dy, input_ids,
                       grad_ptr, grad_dtype,
                       ref_time=ref_time, ref_bytes=ref_bytes)


if __name__ == '__main__':
    test_scan(M=8192, bench=False)
    test_embedding(B=1, M=8192, V=150000, D=8192, transpose=False, bench=False)
    test_embedding(B=2, M=4096, V=150000, D=4096, transpose=True, bench=False)
    test_fused_embedding(B=1, M=8192, V=150000, D=8192, transpose=False,
                         bench=False)
    test_fused_embedding(B=1, M=4096, V=150000, D=8192, transpose=False,
                         bench=False)
    test_fused_embedding(B=2, M=4096, V=150000, D=8192, transpose=True,
                         bench=False)
    test_fused_embedding(B=0, M=4096, V=150000, D=8192, transpose=True,
                         bench=False)
