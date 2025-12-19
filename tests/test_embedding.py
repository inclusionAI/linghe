# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import torch

from linghe.tools.benchmark import benchmark_func
from linghe.tools.check import output_check
from linghe.facade.emb import embedding_lookup




def test_embedding(B=2, M=4096, V=150000, D=4096, bench=False):
    dtype = torch.bfloat16
    device = 'cuda:0'

    embedding = torch.nn.Embedding(V, 4096, dtype=dtype, device=device)
    input_ids = torch.randint(0, V, (2, 4096), dtype=torch.int32, device=device)
    weights = embedding.weight
    dummy_tensor = weights[:1].detach().requires_grad_()

    y_ref = embedding(input_ids)
    dy = y_ref.clone().detach()
    y_ref.backward(dy, retain_graph=True)
    grad_ref = weights.grad.clone().detach()

    weights.grad.zero_()
    y = embedding_lookup(input_ids, dummy_tensor, weights.data_ptr(), weights.grad.data_ptr(),  weights.grad.dtype)
    y.backward(dy, retain_graph=True)
    grad = weights.grad.clone().detach()
    output_check(y_ref, y, name='y')
    output_check(grad_ref, grad, name='grad')

    if bench:
        ref_bytes = B*M*D*4
        ref_time = benchmark_func(embedding.forward, input_ids)
        benchmark_func(embedding_lookup, input_ids, dummy_tensor, weights.data_ptr(), weights.grad.data_ptr(), weights.grad.dtype,
                       ref_time=ref_time, ref_bytes=ref_bytes)

        ref_time = benchmark_func(y_ref.backward,dy, retain_graph=True)
        benchmark_func(y.backward, dy, retain_graph=True,
                       ref_time=ref_time, ref_bytes=ref_bytes)


if __name__ == '__main__':
    test_embedding(B=2, M=4096, V=150000, D=4096, bench=True)
