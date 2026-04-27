# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import pytest
import torch

from linghe.facade.emb import embedding_lookup, fused_accumulation_embedding_lookup
from linghe.tools.check import output_check
from linghe.utils.emb import (
    triton_embedding_forward,
    triton_embedding_backward,
    triton_scan_and_count,
    triton_sync_embedding_backward,
    triton_atomic_embedding_backward,
)


@pytest.mark.parametrize(
    "B,M",
    [
        (None, 8192),
        (None, 4097),
        (4, 8192),
        (2, 4097),
    ],
)
def test_scan(B, M, benchmark):
    device = "cuda:0"
    if B is None or B == 0:
        input_ids = torch.randint(0, 10000, (M,), dtype=torch.int32, device=device)

        sorted_ids, sorted_indices = torch.sort(input_ids, stable=False)
        unique_ids_ref, unique_counts_ref = torch.unique_consecutive(
            sorted_ids, return_counts=True
        )
        accum_counts_ref = torch.cumsum(
            torch.tensor(
                [0] + unique_counts_ref.tolist(), device=unique_counts_ref.device
            ),
            0,
        )
        size = accum_counts_ref.size(0)

        accum_counts = triton_scan_and_count(sorted_ids)
        output_check(accum_counts_ref, accum_counts[:size], name="accum_counts")

        benchmark(triton_scan_and_count, sorted_ids)
    else:
        input_ids = torch.randint(0, 10000, (B, M), dtype=torch.int32, device=device)

        sorted_ids, sorted_indices = torch.sort(input_ids, dim=-1, stable=False)

        accum_counts = triton_scan_and_count(sorted_ids)
        for b in range(B):
            unique_ids_ref, unique_counts_ref = torch.unique_consecutive(
                sorted_ids[b], return_counts=True
            )
            accum_counts_ref = torch.cumsum(
                torch.tensor(
                    [0] + unique_counts_ref.tolist(), device=unique_counts_ref.device
                ),
                0,
            )
            size = accum_counts_ref.size(0)
            output_check(
                accum_counts_ref,
                accum_counts[b, :size],
                name=f"accum_counts (2D, B={B}, M={M}, batch={b})",
            )

        benchmark(triton_scan_and_count, sorted_ids)


@pytest.mark.parametrize(
    "B,M,V,D,transpose",
    [
        (1, 8192, 150000, 8192, False),
        (2, 4096, 150000, 4096, True),
        (1, 4097, 150000, 4096, False),
        (3, 4097, 150000, 4096, False),
    ],
)
def test_embedding(B, M, V, D, transpose, benchmark):
    dtype = torch.bfloat16
    device = "cuda:0"

    embedding = torch.nn.Embedding(V, D, dtype=dtype, device=device)
    input_ids = torch.randint(0, V // 15, (B, M), dtype=torch.int32, device=device)
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
    output_check(y_ref, y, name="y")
    output_check(grad_ref, grad.to(dtype), name="grad")

    ref_bytes = B * M * D * 4
    ref_time = benchmark(embedding.forward, input_ids, ref_bytes=ref_bytes)
    benchmark(
        embedding_lookup, input_ids, weights, ref_time=ref_time, ref_bytes=ref_bytes
    )

    ref_time = benchmark(y_ref.backward, dy, retain_graph=True, ref_bytes=ref_bytes)
    benchmark(y.backward, dy, retain_graph=True, ref_time=ref_time, ref_bytes=ref_bytes)


@pytest.mark.parametrize(
    "B,M,V,D,transpose",
    [
        (1, 8192, 150000, 8192, False),
        (1, 4096, 150000, 8192, False),
        (2, 4096, 150000, 8192, True),
        (2, 4097, 150000, 4096, True),
        (0, 4096, 150000, 8192, True),
        (1, 8100, 150000, 8192, False),
    ],
)
def test_fused_embedding(B, M, V, D, transpose, benchmark, use_main_grad=True):
    dtype = torch.bfloat16
    device = "cuda:0"

    embedding = torch.nn.Embedding(V, D, dtype=dtype, device=device)
    input_ids = torch.randint(0, V // 15, (B, M), dtype=torch.int32, device=device)
    weights = embedding.weight
    main_grad = torch.randn((V, D), dtype=torch.float32, device=device)
    grad_ref = main_grad.to(dtype)
    grad = main_grad.to(dtype)

    weights.grad = grad_ref
    y_ref = embedding(input_ids)
    if transpose:
        dy = torch.randn((M, B, D), device=device, dtype=dtype).permute(1, 0, 2)
    else:
        dy = torch.randn((B, M, D), device=device, dtype=dtype)
    y_ref.backward(dy, retain_graph=True)

    weights.grad = grad
    weights.main_grad = main_grad
    grad_name = "main_grad" if use_main_grad else "grad"
    y = fused_accumulation_embedding_lookup(input_ids, weights, grad_name=grad_name)
    y.backward(dy, retain_graph=True)
    output_check(y_ref, y, name="y")
    output_check(
        grad_ref, main_grad.to(dtype) if use_main_grad else grad, name="grad", atol=0.05
    )

    ref_bytes = B * M * D * 4
    ref_time = benchmark(embedding.forward, input_ids)
    benchmark(
        fused_accumulation_embedding_lookup,
        input_ids,
        weights,
        grad_name=grad_name,
        ref_time=ref_time,
        ref_bytes=ref_bytes,
    )

    ref_time = benchmark(y_ref.backward, dy, retain_graph=True)
    benchmark(y.backward, dy, retain_graph=True, ref_time=ref_time, ref_bytes=ref_bytes)
    grad_ptr = main_grad.data_ptr() if use_main_grad else grad.data_ptr()
    grad_dtype = main_grad.dtype if use_main_grad else grad.dtype
    benchmark(
        triton_atomic_embedding_backward,
        dy,
        input_ids,
        grad_ptr,
        grad_dtype,
        ref_time=ref_time,
        ref_bytes=ref_bytes,
    )
    benchmark(
        triton_sync_embedding_backward,
        dy,
        input_ids,
        grad_ptr,
        grad_dtype,
        ref_time=ref_time,
        ref_bytes=ref_bytes,
    )
    benchmark(
        triton_embedding_backward,
        dy,
        input_ids,
        grad_ptr,
        grad_dtype,
        ref_time=ref_time,
        ref_bytes=ref_bytes,
    )
