import functools
import math
import operator
from typing import Callable, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


# =============================================================================
# Constants
# =============================================================================

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
SF_VEC_SIZE = 16
COPY_BITS = 128


# =============================================================================
# Architecture Detection
# =============================================================================


@functools.lru_cache(maxsize=16)
def get_sm_version(device: int | torch.device | str | None = None) -> int:
    """Get the SM version of a CUDA device.

    Args:
        device: CUDA device to query. Can be an int (device index), torch.device,
            device string (e.g., 'cuda:0'), or None to use current device.

    Returns:
        SM version as an integer (e.g., 100 for SM100).
    """
    if not torch.cuda.is_available():
        return 80
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


# =============================================================================
# PTX Intrinsics
# =============================================================================

@dsl_user_op
def ld_global_v4_u32(base_ptr: Int64, *, loc=None, ip=None):
    """Load 128 bits (4 x uint32) from global memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    v0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    v1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    v2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    v3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)
    return Uint32(v0), Uint32(v1), Uint32(v2), Uint32(v3)


@dsl_user_op
def st_global_u64(ptr: Int64, val: Uint64, *, loc=None, ip=None):
    """Store 64 bits to global memory."""
    llvm.inline_asm(
        None,
        [Int64(ptr).ir_value(loc=loc, ip=ip),
         Uint64(val).ir_value(loc=loc, ip=ip)],
        "st.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def get_ptr_as_int64(
    tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None
) -> Int64:
    elem_ptr = tensor.iterator + Int32(offset)
    return Int64(llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def rcp_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(), [Float32(a).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmax_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """max(a, b) for float32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip),
             Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bf2_abs(x: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(), [Uint32(x).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0x7FFF7FFF;",
            "=r,r",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bf2_max(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip),
             Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bf2_max_scalar(x: Uint32, *, loc=None, ip=None) -> Float32:
    """Extract max of two packed bf16 values, returned as float32."""
    return Float32(
        llvm.inline_asm(
            T.f32(), [Uint32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b32 lo, hi;
                .reg .f32 f0, f1;
                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 f0, lo;
                mov.b32 f1, hi;
                max.f32 $0, f0, f1;
            }
            """,
            "=f,r",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bf2_to_f32x2(bf2: Uint32, s: Float32, *, loc=None, ip=None):
    """Convert bf16x2 to (f32, f32) and multiply each by scalar s."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(bf2).ir_value(loc=loc, ip=ip),
         Float32(s).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            .reg .f32 f0, f1;
            and.b32 lo, $2, 0xFFFF;
            shr.b32 hi, $2, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 f0, lo;
            mov.b32 f1, hi;
            mul.f32 $0, f0, $3;
            mul.f32 $1, f1, $3;
        }
        """,
        "=f,=f,r,f",
        has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    return Float32(f0), Float32(f1)


@dsl_user_op
def cvt_f32x2_e4m3(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert two f32 to packed e4m3x2 in low 16 bits of uint32.

    Byte layout: low byte = e4m3(a), high byte = e4m3(b).
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip),
             Float32(b).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 p;
                cvt.rn.satfinite.e4m3x2.f32 p, $2, $1;
                cvt.u32.u16 $0, p;
            }
            """,
            "=r,f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def pack_e4m3x8(
    p0: Uint32, p1: Uint32, p2: Uint32, p3: Uint32, *, loc=None, ip=None
) -> Uint64:
    """Pack four e4m3x2 results (each low-16-bit of uint32) into one uint64.

    Memory byte order: p0[0], p0[1], p1[0], p1[1], p2[0], p2[1], p3[0], p3[1].
    """
    return Uint64(
        llvm.inline_asm(
            T.i64(),
            [Uint32(p0).ir_value(loc=loc, ip=ip),
             Uint32(p1).ir_value(loc=loc, ip=ip),
             Uint32(p2).ir_value(loc=loc, ip=ip),
             Uint32(p3).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 h0, h1, h2, h3;
                .reg .b32 lo, hi;
                cvt.u16.u32 h0, $1;
                cvt.u16.u32 h1, $2;
                cvt.u16.u32 h2, $3;
                cvt.u16.u32 h3, $4;
                mov.b32 lo, {h0, h1};
                mov.b32 hi, {h2, h3};
                mov.b64 $0, {lo, hi};
            }
            """,
            "=l,r,r,r,r",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )

@dsl_user_op
def round_scale_pow2(s: Float32, *, loc=None, ip=None) -> Float32:
    """exp2(ceil(log2(s))) -- round scale up to next power of 2."""
    return Float32(
        llvm.inline_asm(
            T.f32(), [Float32(s).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .f32 l, c;
                lg2.approx.f32 l, $1;
                cvt.rpi.f32.f32 c, l;
                ex2.approx.f32 $0, c;
            }
            """,
            "=f,f",
            has_side_effects=False, is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )

