# -*- coding: utf-8 -*-
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import make_ptr
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm
from linghe.tools.check import output_check
from linghe.tools.benchmark import benchmark_func
from linghe.quant.block import triton_block_quant

def torch_block_quant(w, B=128, dtype=torch.float8_e4m3fn, round_scale=False):
    fmax = torch.finfo(dtype).max
    w = w.clone()
    N, K = w.shape

    wp = torch.reshape(w, (N // B, B, K // B, B)).permute(0, 2, 1, 3)
    scale = torch.amax(torch.amax(torch.abs(wp).float(), dim=2), dim=2) / fmax
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(scale)))
    wq = (wp / scale[:, :, None, None]).to(dtype)
    wq = wq.permute(0, 2, 1, 3)
    wq = torch.reshape(wq, (N, K)).contiguous()

    return wq, scale

FP8_E4M3_MAX = 448.0
B_QUANT = 128
THREADS = 256
N_VECS = B_QUANT // 8
N_ROWS_ITER = THREADS // N_VECS
N_ITERS = B_QUANT // N_ROWS_ITER
SMEM_BYTES = 9 * 4

from linghe.experimental.dsl_common import (
    ld_global_v4_u32,
    st_global_u64,
    get_ptr_as_int64,
    rcp_approx_ftz,
    fmax_f32,
    bf2_abs,
    bf2_max,
    bf2_max_scalar,
    bf2_to_f32x2,
    cvt_f32x2_e4m3,
    pack_e4m3x8,
    round_scale_pow2
)

@cute.jit
def warp_reduce_max(v: Float32) -> Float32:
    for i in cutlass.range_constexpr(5):
        v = fmax_f32(v, cute.arch.shuffle_sync_bfly(v, offset=(1 << i)))
    return v

class BlockQuantKernel:
    def __init__(self, K: int, B: int = B_QUANT, round_scale: bool = False):
        self.K = K
        self.B = B
        self.round_scale = round_scale
        assert K % B == 0, f"K ({K}) must be divisible by B ({B})"

    @cute.kernel
    def kernel(
        self,
        mW: cute.Tensor,
        mWq: cute.Tensor,
        mScale: cute.Tensor,
    ):
        tid   = cute.arch.thread_idx()[0]
        bid_x = cute.arch.block_idx()[0]
        bid_y = cute.arch.block_idx()[1]

        K = self.K
        B = self.B

        row_base     = bid_y * B
        col_base     = bid_x * B
        vec_id       = tid % N_VECS
        row_in_group = tid // N_VECS

        smem = utils.SmemAllocator()
        buf  = smem.allocate_tensor(
            cute.Float32, cute.make_layout((9,)), byte_alignment=4,
        )

        local_max = Float32(0.0)

        for it in cutlass.range_constexpr(N_ITERS):
            row = row_base + it * N_ROWS_ITER + row_in_group
            col = col_base + vec_id * 8

            ptr = get_ptr_as_int64(mW, row * K + col)
            v0, v1, v2, v3 = ld_global_v4_u32(ptr)

            a01  = bf2_max(bf2_abs(v0), bf2_abs(v1))
            a23  = bf2_max(bf2_abs(v2), bf2_abs(v3))
            amax = bf2_max(a01, a23)
            local_max = fmax_f32(local_max, bf2_max_scalar(amax))

        warp_max = warp_reduce_max(local_max)

        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()

        if lane == 0:
            buf[(warp,)] = warp_max
        cute.arch.barrier()

        bmax = Float32(0.0)
        if lane < 8:
            bmax = buf[(lane,)]
        bmax = warp_reduce_max(bmax)

        if tid == 0:
            scale = bmax / Float32(FP8_E4M3_MAX)
            if cutlass.const_expr(self.round_scale):
                scale = round_scale_pow2(scale)
            mScale[(bid_y, bid_x)] = scale
            buf[(8,)] = rcp_approx_ftz(scale)

        cute.arch.barrier()
        inv_s = buf[(8,)]

        for it in cutlass.range_constexpr(N_ITERS):
            row = row_base + it * N_ROWS_ITER + row_in_group
            col = col_base + vec_id * 8

            in_ptr = get_ptr_as_int64(mW, row * K + col)
            v0, v1, v2, v3 = ld_global_v4_u32(in_ptr)

            f0, f1 = bf2_to_f32x2(v0, inv_s)
            f2, f3 = bf2_to_f32x2(v1, inv_s)
            f4, f5 = bf2_to_f32x2(v2, inv_s)
            f6, f7 = bf2_to_f32x2(v3, inv_s)

            p0 = cvt_f32x2_e4m3(f0, f1)
            p1 = cvt_f32x2_e4m3(f2, f3)
            p2 = cvt_f32x2_e4m3(f4, f5)
            p3 = cvt_f32x2_e4m3(f6, f7)

            packed = pack_e4m3x8(p0, p1, p2, p3)
            out_ptr = get_ptr_as_int64(mWq, row * K + col)
            st_global_u64(out_ptr, packed)

    @cute.jit
    def __call__(
        self,
        w_ptr:     cute.Pointer,
        wq_ptr:    cute.Pointer,
        scale_ptr: cute.Pointer,
        N:         cute.Int32,
    ):
        K = self.K
        B = self.B

        blks_k = K // B
        blks_n = cute.size(cute.ceil_div(N, B))

        mW = cute.make_tensor(
            w_ptr,
            layout=cute.make_layout((N, K), stride=(K, 1)),
        )

        mWq = cute.make_tensor(
            wq_ptr,
            layout=cute.make_layout((N, K), stride=(K, 1)),
        )

        mScale = cute.make_tensor(
            scale_ptr,
            layout=cute.make_layout(
                (blks_n, blks_k),
                stride=(blks_k, 1),
            ),
        )

        self.kernel(mW, mWq, mScale).launch(
            grid=[blks_k, blks_n, 1],
            block=[THREADS, 1, 1],
            smem=SMEM_BYTES,
        )