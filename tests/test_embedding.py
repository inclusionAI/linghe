# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.utils.emb import triton_embedding_forward, triton_embedding_backward
from linghe.facade.emb import embedding_lookup



def test_embedding(B=2, M=4096, V=150000, D=4096, transpose=False, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    embedding = torch.nn.Embedding(V, D, dtype=dtype, device=device)
    input_ids = torch.randint(0, V//15, (B, M), dtype=torch.int32, device=device)
    weights = embedding.weight
    dummy_tensor = torch.randn((D,), device=device, dtype=dtype, requires_grad=True)

    y_ref = embedding(input_ids)
    if transpose:
        dy = torch.randn((M,B,D), device=device, dtype=dtype).permute(1,0,2)
    else:
        dy = torch.randn((B,M,D), device=device, dtype=dtype)
    y_ref.backward(dy, retain_graph=True)
    grad_ref = weights.grad.clone().detach()

    grad = weights.grad.clone().detach()
    grad.zero_()
    y = triton_embedding_forward(input_ids, weights.data_ptr(), D, dtype)
    triton_embedding_backward(dy, input_ids, grad.data_ptr(), grad.dtype)
    output_check(y_ref, y, name='y')
    output_check(grad_ref, grad, name='grad')

    weights.grad.zero_()
    y = embedding_lookup(input_ids, weights.data_ptr(), weights.grad.data_ptr(), D, dtype, weights.grad.dtype, dummy_tensor)
    y.backward(dy, retain_graph=True)
    grad = weights.grad.clone().detach()
    output_check(y_ref, y, name='y')
    output_check(grad_ref, grad, name='grad')

    if bench:
        # benchmark_func(torch.unique, input_ids.view(-1), sorted=True, return_inverse=True, return_counts=True)
        # benchmark_func(torch.argsort, input_ids.view(-1), stable=False)
        # benchmark_func(torch.unique_consecutive, torch.argsort(input_ids.view(-1), stable=False), return_counts=True)

        ref_bytes = B*M*D*4
        ref_time = benchmark_func(embedding.forward, input_ids)
        benchmark_func(embedding_lookup, input_ids, weights.data_ptr(), weights.grad.data_ptr(), D, dtype, weights.grad.dtype, dummy_tensor,
                       ref_time=ref_time, ref_bytes=ref_bytes)

        ref_time = benchmark_func(y_ref.backward,dy, retain_graph=True)
        benchmark_func(y.backward, dy, retain_graph=True,
                       ref_time=ref_time, ref_bytes=ref_bytes)


if __name__ == '__main__':
    test_embedding(B=1, M=8192, V=150000, D=8192, transpose=False, bench=True)
    test_embedding(B=1, M=8192, V=150000, D=8192, transpose=True, bench=True)
    test_embedding(B=2, M=4096, V=150000, D=8192, transpose=False, bench=True)
    test_embedding(B=2, M=4096, V=150000, D=8192, transpose=True, bench=True)

