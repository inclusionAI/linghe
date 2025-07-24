import itertools

import torch
import triton
import triton.language as tl
from triton import Config

from flops.utils.util import round_up


# os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

@triton.jit
def deprecated_transpose_kernel(x_ptr, t_ptr, M, N, H: tl.constexpr,
                                W: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(axis=0)
    # col-wise read, row-wise write
    offs = pid * W + tl.arange(0, H)[:, None] * N + tl.arange(0, W)[None, :]
    toffs = pid * W * M + tl.arange(0, W)[:, None] * M + tl.arange(0, H)[None,
                                                         :]
    m = tl.cdiv(M, H)
    for i in range(m):
        if EVEN:
            y = tl.trans(tl.load(x_ptr + offs))
            tl.store(t_ptr + toffs, y)
        else:
            y = tl.trans(tl.load(x_ptr + offs, mask=(pid * W + tl.arange(0, W)[
                                                               None, :] < N) & (
                                                                i * H + tl.arange(
                                                            0, H)[:,
                                                                        None] < M)))
            tl.store(t_ptr + toffs, y,
                     mask=(pid * W + tl.arange(0, W)[:, None] < N) & (
                                 i * H + tl.arange(0, H)[None, :] < M))
        offs += H * N
        toffs += H


def triton_depracated_transpose(x):
    M, N = x.shape
    device = x.device
    t = torch.empty((N, M), device=device, dtype=x.dtype)

    H = 512
    W = 32 if x.dtype.itemsize == 1 else 16
    EVEN = M % H == 0 and N % W == 0
    num_stages = 3
    num_warps = 8

    grid = (triton.cdiv(N, W),)
    deprecated_transpose_kernel[grid](
        x, t,
        M, N,
        H, W,
        EVEN,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return t


@triton.jit
def transpose_kernel(x_ptr, t_ptr, M, N, H: tl.constexpr, W: tl.constexpr,
                     EVEN: tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    offs = rid * H * N + cid * W + tl.arange(0, H)[:, None] * N + tl.arange(0,
                                                                            W)[
                                                                  None, :]
    toffs = rid * H + cid * M * W + tl.arange(0, W)[:, None] * M + tl.arange(0,
                                                                             H)[
                                                                   None, :]
    if EVEN:
        y = tl.trans(tl.load(x_ptr + offs))
        tl.store(t_ptr + toffs, y)
    else:
        y = tl.trans(tl.load(x_ptr + offs,
                             mask=(cid * W + tl.arange(0, W)[None, :] < N) & (
                                         rid * H + tl.arange(0, H)[:,
                                                   None] < M)))
        tl.store(t_ptr + toffs, y,
                 mask=(cid * W + tl.arange(0, W)[:, None] < N) & (
                             rid * H + tl.arange(0, H)[None, :] < M))


def triton_transpose(x):
    M, N = x.shape
    device = x.device
    t = torch.empty((N, M), device=device, dtype=x.dtype)
    H = 64
    W = 32 if x.dtype.itemsize == 1 else 16
    EVEN = M % H == 0 and N % W == 0
    num_stages = 5
    num_warps = 2

    grid = (triton.cdiv(M, H), triton.cdiv(N, W))
    transpose_kernel[grid](
        x, t,
        M, N,
        H, W,
        EVEN,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return t


@triton.jit
def transpose_and_pad_kernel(x_ptr, t_ptr, M, N, P, H: tl.constexpr,
                             W: tl.constexpr, EVEN: tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    offs = rid * H * N + cid * W + tl.arange(0, H)[:, None] * N + tl.arange(0,
                                                                            W)[
                                                                  None, :]
    toffs = rid * H + cid * P * W + tl.arange(0, W)[:, None] * P + tl.arange(0,
                                                                             H)[
                                                                   None, :]
    if EVEN:
        y = tl.trans(tl.load(x_ptr + offs))
    else:
        y = tl.trans(tl.load(x_ptr + offs,
                             mask=(rid * H + tl.arange(0, H)[:, None] < M)))
    # paddings are filled with 0
    tl.store(t_ptr + toffs, y)


"""
pad: M will be padded to mutiplier of 32
M is usually less than N without deepep
"""


def triton_transpose_and_pad(x, out=None, pad=True):
    # fat block, shape:[H,W]
    M, N = x.shape
    P = round_up(M, b=32) if pad else M
    device = x.device
    if out is None:
        out = torch.empty((N, P), device=device, dtype=x.dtype)

    H = 32
    W = 64
    num_stages = 5
    num_warps = 2
    EVEN = M % H == 0
    grid = (triton.cdiv(M, H), triton.cdiv(N, W))
    transpose_and_pad_kernel[grid](
        x, out,
        M, N, P,
        H, W,
        EVEN,
        num_stages=num_stages,
        num_warps=num_warps
    )
    return out


@triton.jit
def batch_transpose_kernel(xs_ptr, xts_ptr, M, N, H: tl.constexpr,
                           W: tl.constexpr):
    eid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    x_ptr = tl.load(xs_ptr + eid).to(tl.pointer_type(xts_ptr.dtype.element_ty))
    offs = cid * W + tl.arange(0, H)[:, None] * N + tl.arange(0, W)[None, :]
    toffs = eid * M * N + cid * W * M + tl.arange(0, W)[:,
                                        None] * M + tl.arange(0, H)[None, :]
    for i in range(0, M, H):
        y = tl.trans(tl.load(x_ptr + offs))
        tl.store(xts_ptr + toffs, y)
        offs += N * H
        toffs += H


"""
x: [M, N]*expert
x_t: [N,M]*expert
"""


def triton_batch_transpose(xs, xts=None):
    # block shape:[H,W]
    M, N = xs[0].shape
    n_experts = len(xs)
    if xts is None:
        xts = torch.empty((M * n_experts, N), device=xs[0].device,
                          dtype=xs[0].dtype)
    pointers = torch.tensor([x.data_ptr() for x in xs], device=xs[0].device)
    H = 64
    W = 64
    num_stages = 3
    num_warps = 8
    grid = (n_experts, N // W)
    batch_transpose_kernel[grid](
        pointers, xts,
        M, N,
        H, W,
        num_stages=num_stages,
        num_warps=num_warps
    )
    # outputs = torch.split(xts, n_experts)  # very slow
    outputs = torch.split(xts, [M] * n_experts)
    # outputs = []
    # for i in range(n_experts):
    #     outputs.append(xts[i*M:(i+1)*M])
    return outputs


@triton.jit
def batch_transpose_and_pad_kernel(x_ptr, t_ptr, count_ptr, accum_ptr,
                                   pad_accum_ptr, N, H: tl.constexpr,
                                   W: tl.constexpr):
    eid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    count = tl.load(count_ptr + eid)
    ei = tl.load(accum_ptr + eid)
    si = ei - count
    pad_si = tl.load(pad_accum_ptr + eid)
    P = tl.cdiv(count, 32) * 32
    offs = si * N + cid * W + tl.arange(0, H)[:, None] * N + tl.arange(0, W)[
                                                             None, :]
    toffs = pad_si * N + cid * W * P + tl.arange(0, W)[:, None] * P + tl.arange(
        0, H)[None, :]
    for i in range(0, P, H):
        y = tl.trans(
            tl.load(x_ptr + offs, mask=i + tl.arange(0, H)[:, None] < count))
        # paddings are filled with 0
        tl.store(t_ptr + toffs, y, mask=i + tl.arange(0, H)[None, :] < P)
        offs += N * H
        toffs += H


"""
pad: M will be padded to mutiplier of 32
padding should be filled with 0
M is usually less than N
x: [sum(bs), N]
x_t: [sum([pad(bs)*N for bs in bss])]
"""


def triton_batch_transpose_and_pad(x, count_list, x_t=None, pad=True):
    assert pad
    # block shape:[H,W]
    M, N = x.shape
    n_experts = len(count_list)
    # NOTE: b must be 32, or kernel will miscalculated padding size with original size
    pad_sizes = [round_up(x, b=32) for x in count_list]
    counts = torch.tensor(count_list, dtype=torch.int32, device=x.device)
    pad_accum_sizes = torch.tensor(
        list(itertools.accumulate(pad_sizes, initial=0)), dtype=torch.int32,
        device=x.device)
    accums = torch.cumsum(counts, 0)
    device = x.device
    if x_t is None:
        x_t = torch.empty((sum(pad_sizes) * N), device=device, dtype=x.dtype)
    H = 256
    W = 64
    num_stages = 2
    num_warps = 8
    grid = (n_experts, N // W)
    batch_transpose_and_pad_kernel[grid](
        x, x_t,
        counts, accums,
        pad_accum_sizes,
        N,
        H, W,
        num_stages=num_stages,
        num_warps=num_warps
    )
    split_size = [x * N for x in pad_sizes]
    chunks = torch.split(x_t.view(torch.uint8), split_size)
    outputs = []
    for i, p in enumerate(pad_sizes):
        outputs.append(chunks[i].view(torch.float8_e4m3fn).view(N, p))
    return outputs


configs = [
    Config({"H": H, "W": W}, num_stages=num_stages, num_warps=num_warps)
    for H in [32, 64, 128, 256, 512]
    for W in [16, 32, 64]
    for num_stages in [3, 4, 5, 6]
    for num_warps in [2, 4, 8]
]


@triton.autotune(configs=configs, key=["M", "N", "D"])
@triton.jit
def opt_transpose_kernel(x_ptr, t_ptr, M, N, D, H: tl.constexpr,
                         W: tl.constexpr):
    pid = tl.program_id(axis=0)
    # row-wise read, col-wise write
    offs = pid * W + tl.arange(0, H)[:, None] * N + tl.arange(0, W)[None, :]
    toffs = pid * W * M + tl.arange(0, W)[:, None] * M + tl.arange(0, H)[None,
                                                         :]
    m = tl.cdiv(M, H)
    for i in range(m):
        y = tl.trans(tl.load(x_ptr + offs))
        tl.store(t_ptr + toffs, y)
        offs += H * N
        toffs += H


def triton_opt_transpose(x):
    M, N = x.shape
    device = x.device
    D = 0 if x.dtype.itemsize == 1 else 1
    t = torch.empty((N, M), device=device, dtype=x.dtype)
    grid = lambda META: (N // META["W"],)
    opt_transpose_kernel[grid](
        x, t,
        M, N, D
    )
    return t
