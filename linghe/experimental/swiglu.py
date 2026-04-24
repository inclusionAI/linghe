import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import make_ptr
from cutlass.base_dsl._mlir_helpers.op import dsl_user_op

threads   = 128 # 128 for h200 256 for b200
threads_m = 4
threads_n = threads // threads_m
vm = 1
vn = 128 // cute.BFloat16.width
BM = threads_m * vm
BN = threads_n * vn

SMEM_BYTES = 2 * BM * BN * (cute.BFloat16.width // 8) + BM * 4


@dsl_user_op
def silu(x: cute.TensorSSA, *, loc=None, ip=None):
    exp_mius_x = cute.exp(-x)
    return x / (1 + exp_mius_x)


@dsl_user_op
def swiglu(x: cute.TensorSSA, y: cute.TensorSSA, *, loc=None, ip=None):
    return silu(x, loc=loc, ip=ip) * y


class SwiGLUKernel:
    def __init__(self, N: int):
        self.N = N
        self.N_halved = N // 2

    @cute.kernel
    def kernel(
        self,
        mInput: cute.Tensor,
        mOutput: cute.Tensor,
        weight: cute.Tensor,
    ):
        tid_x = cute.arch.thread_idx()[0]
        bid_x = cute.arch.block_idx()[0]
        bid_y = cute.arch.block_idx()[1]

        N_halved = self.N_halved
        blocks_n = N_halved // BN

        tiler = (BM, BN)
        input_tile_0 = cute.local_tile(mInput, tiler, (bid_y, bid_x))
        input_tile_1 = cute.local_tile(mInput, tiler, (bid_y, bid_x + blocks_n))
        output_tile  = cute.local_tile(mOutput, tiler, (bid_y, bid_x))

        thr_m = tid_x // threads_n
        thr_n = tid_x % threads_n

        smem = utils.SmemAllocator()
        tile_layout = cute.make_ordered_layout((BM, BN), order=(1, 0))
        s0 = smem.allocate_tensor(mInput.element_type, tile_layout, byte_alignment=16)
        s1 = smem.allocate_tensor(mInput.element_type, tile_layout, byte_alignment=16)
        sw = smem.allocate_tensor(cute.Float32, cute.make_layout((BM,)), byte_alignment=4)

        tiler_mn = (BM, BN)
        num_vec_blocks = 1
        tv_layout = cute.make_layout( 
            ((threads_n, threads_m), (vn, num_vec_blocks)), #thread 0 -> 0, 4, 8, 12, ....
            stride=(
                (vn * threads_m, 1),
                (threads_m, threads_m * vn * threads_n),
            ),
        )

        cp_async_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), mInput.element_type, num_bits_per_copy=128
        )
        tiled_copy_async = cute.make_tiled_copy(cp_async_atom, tv_layout, tiler_mn)
        thr_copy_async   = tiled_copy_async.get_slice(tid_x)

        gmem_thr_0 = thr_copy_async.partition_S(input_tile_0)
        gmem_thr_1 = thr_copy_async.partition_S(input_tile_1)
        smem_thr_0 = thr_copy_async.partition_D(s0)
        smem_thr_1 = thr_copy_async.partition_D(s1)

        cute.copy(cp_async_atom, gmem_thr_0, smem_thr_0)
        cute.copy(cp_async_atom, gmem_thr_1, smem_thr_1)
        cute.arch.cp_async_commit_group()

        weight_tile = weight[(None, bid_y)]
        if thr_n == 0:
            sw[(thr_m,)] = weight_tile[(thr_m * vm,)]

        cute.arch.cp_async_wait_group(0)
        cute.arch.barrier()

        input_frag_0 = cute.make_fragment_like(gmem_thr_0)
        input_frag_1 = cute.make_fragment_like(gmem_thr_1)
        cute.autovec_copy(smem_thr_0, input_frag_0)
        cute.autovec_copy(smem_thr_1, input_frag_1)

        weight_val = sw[(thr_m,)]

        result = (
            swiglu(input_frag_0.load(), input_frag_1.load()).to(cute.Float32) * weight_val
        ).to(mInput.element_type)

        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), mOutput.element_type, num_bits_per_copy=128
        )
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)
        thr_copy_store = tiled_copy_store.get_slice(tid_x)

        output_thr = thr_copy_store.partition_D(output_tile)
        output_frag = cute.make_fragment_like(output_thr)
        output_frag.store(result)
        cute.copy(copy_atom_store, output_frag, output_thr)

    @cute.jit
    def __call__(
        self,
        input_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        weight_ptr: cute.Pointer,
        M: cute.Int32,
    ):
        N = self.N
        N_halved = self.N_halved

        blocks_m = cute.size(cute.ceil_div(M, BM))
        blocks_n = cute.size(cute.ceil_div(N_halved, BN))

        mInput = cute.make_tensor(
            input_ptr,
            layout=cute.make_layout((M, N), stride=(N, 1)),
        )

        mOutput = cute.make_tensor(
            output_ptr,
            layout=cute.make_layout((M, N_halved), stride=(N_halved, 1)),
        )

        weight_tiled = cute.make_tensor(
            weight_ptr,
            layout=cute.make_layout(
                (BM, blocks_m),
                stride=(1, BM),
            ),
        )

        grid_size = [blocks_n, blocks_m, 1]
        block_size = [threads, 1, 1]

        self.kernel(mInput, mOutput, weight_tiled).launch(
            grid=grid_size, block=block_size, smem=SMEM_BYTES
        )
