

import torch
import triton
import triton.language as tl
from triton import Config


@triton.jit
def split_and_cat_kernel(x_ptr, y_ptr, scale_ptr, scale_output_ptr, count_ptr, accum_ptr, rev_accum_ptr, index_ptr, M, N: tl.constexpr, SCALE: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, row-wise write
    index = tl.load(index_ptr+pid)
    count = tl.load(count_ptr+index)
    ei = tl.load(accum_ptr+index)
    si = ei - count
    rev_ei = tl.load(rev_accum_ptr+pid)
    rev_si = rev_ei - count
    for i in range(count):
        x = tl.load(x_ptr + si*N+i*N+tl.arange(0, N))
        tl.store(y_ptr+rev_si*N+i*N+tl.arange(0, N), x)


    if SCALE:
        for i in range(tl.cdiv(count, K)):
            scale = tl.load(scale_ptr + si+i*K+tl.arange(0, K), mask=i*K+tl.arange(0, K)<count)
            tl.store(scale_output_ptr+rev_si+i*K+tl.arange(0, K), scale, mask=i*K+tl.arange(0, K)<count)

"""
select and smooth and quant
x: [bs, dim]
counts: [n_split]
indices: [n_split]
"""
def triton_split_and_cat(x, counts, indices, scales=None):
    M, N = x.shape
    n_split = counts.shape[0]
    device = x.device 
    y = torch.empty((M, N), device=device, dtype=x.dtype)
    if scales is not None:
        output_scales = torch.empty((M, ), device=device, dtype=torch.float32)
        S = True 
    else:
        output_scales = None
        S = False
    accums = torch.cumsum(counts, 0)
    reverse_accums = torch.cumsum(counts[indices], 0)
    # TODO: adapt for n_expert <= 64
    K = 256
    grid = (n_split, )
    split_and_cat_kernel[grid](
        x,
        y,
        scales,
        output_scales,
        counts,
        accums,
        reverse_accums,
        indices,
        M, 
        N, 
        S,
        K,
        num_stages=3,
        num_warps=8
    )
    return y, output_scales


