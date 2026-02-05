import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def split_fp32_gemm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        SPLIT_COUNT: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_k = tl.program_id(axis=2)
    k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_COUNT)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + pid_k * K // SPLIT_COUNT + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + pid_k * K // SPLIT_COUNT + offs_n[None, :] * K + offs_k[:, None]
    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        current_k = pid_k * K // SPLIT_COUNT + i * BLOCK_SIZE_K + offs_k
        a = tl.load(a_ptrs, mask=(offs_m < M)[:, None] & (current_k < K)[None, :], other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=(offs_n < N)[None, :] & (current_k < K)[:, None], other=0.0).to(tl.float32)
        c = tl.dot(a, b, c)
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
    
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    if SPLIT_COUNT == 1:
        tl.store(c_ptrs, c, mask=(offs_m < M)[:, None] & (offs_n < N)[None, :])
    else:
        tl.atomic_add(c_ptrs, c, mask=(offs_m < M)[:, None] & (offs_n < N)[None, :], sem='relaxed')


def triton_split_fp32_gemm(x: torch.Tensor, w: torch.Tensor):
    """
    return fp32 gemm result with fp16/bf16 inputs,
        it's mainly used for MoE router GEMM
        and DO NOT suitable for large size GEMM
    Args:
        a: left matrix with fp16/bf16 precision
        b: right matrix with fp16/bf16 precision
    Returns:
        c: output with fp32 precision
    """
    assert x.is_contiguous() and w.is_contiguous()
    M, K = x.size()
    N, K = w.size()
    BLOCK_SIZE_K = 128
    if M <= 32:
        BLOCK_SIZE_M = 16
    elif M <= 64:
        BLOCK_SIZE_M = 32
    elif M <= 128:
        BLOCK_SIZE_M = 64
    else:
        BLOCK_SIZE_M = 128
    
    if N <= 64:
        BLOCK_SIZE_N = max([x for x in [16, 32, 64] if N % x == 0 or N < x])
    elif N <= 256:
        BLOCK_SIZE_N = max([x for x in [32, 64] if N % x == 0 or N < x])
    else:
        BLOCK_SIZE_N = max([x for x in [64, 128] if N % x == 0 or N < x])
    
    SPLIT_COUNT = min(triton.cdiv(K, 2048), 4)
    if SPLIT_COUNT == 1:
        c = torch.empty(M, N, dtype=torch.float32, device=x.device)
    else:
        c = torch.zeros(M, N, dtype=torch.float32, device=x.device)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]),
                         triton.cdiv(N, META["BLOCK_SIZE_N"]),
                         SPLIT_COUNT)  # noqa
    num_warps = 4
    num_stages = 3
    split_fp32_gemm_kernel[grid](x, w, c,
                           M, N, K,
                           BLOCK_SIZE_K,
                           BLOCK_SIZE_M,
                           BLOCK_SIZE_N,
                           SPLIT_COUNT,
                           num_warps=num_warps,
                           num_stages=num_stages
                           )
    return c


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def matmul_kernel_tma_persistent(
    a_desc,
    b_desc,
    c_desc,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SM: tl.constexpr,
):
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n

    tid_c = start_pid - SM
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tid in tl.range(start_pid, num_tiles, SM, flatten=True):
        pid_m, pid_n = _compute_pid(tid, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_a = pid_m * BLOCK_SIZE_M
        offs_b = pid_n * BLOCK_SIZE_N

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(k_tiles):
            offs_k = k * BLOCK_SIZE_K
            a = a_desc.load([offs_a, offs_k])
            b = b_desc.load([offs_b, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tid_c += SM
        pid_m, pid_n = _compute_pid(
            tid_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M
        )
        offs_a_acc = pid_m * BLOCK_SIZE_M
        offs_b_acc = pid_n * BLOCK_SIZE_N

        acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
        acc = tl.permute(acc, (0, 2, 1))
        acc0, acc1 = tl.split(acc)
        c_desc.store([offs_a_acc, offs_b_acc], acc0)
        c_desc.store([offs_a_acc, offs_b_acc + BLOCK_SIZE_N // 2], acc1)


def matmul_tma_persistent(a, b):
    M, K = a.shape
    N, K = b.shape
    dtype = torch.float32

    c = torch.empty((M, N), device=a.device, dtype=dtype)

    SM = torch.cuda.get_device_properties("cuda").multi_processor_count

    BLOCK_M = 128
    BLOCK_K = 64
    BLOCK_N = 64
    GROUP_SIZE_M = 8

    a_desc = TensorDescriptor(a, a.shape, a.stride(), [BLOCK_M, BLOCK_K])
    b_desc = TensorDescriptor(b, b.shape, b.stride(), [BLOCK_N, BLOCK_K])
    c_desc = TensorDescriptor(c, c.shape, c.stride(), [BLOCK_M, BLOCK_N // 2])

    def grid(META):
        nonlocal a_desc, b_desc, c_desc
        return (
            min(
                SM,
                triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),
            ),
        )

    matmul_kernel_tma_persistent[grid](
        a_desc,
        b_desc,
        c_desc,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_K,
        BLOCK_N,
        GROUP_SIZE_M,
        SM=SM,
    )
    return c