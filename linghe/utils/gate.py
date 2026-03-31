import torch
import triton
import triton.language as tl
import math


@triton.jit
def group_rms_norm_gate_forward_kernel(
    x_ptr,
    gate_ptr,
    weight_ptr,
    out_ptr,
    stride_g,
    eps,
    bs,
    length,
    DIM: tl.constexpr,
    d: tl.constexpr,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SHARE: tl.constexpr,
    NATIVE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    bid = pid // length
    sid = pid % length

    if SHARE:
        weight = tl.load(weight_ptr + tl.arange(0, D),
                         mask=tl.arange(0, D) < d)[None, :]
    else:
        weight = tl.load(weight_ptr
                         + tl.arange(0, GROUP_SIZE)[:, None] * d
                         + tl.arange(0, D),
                         mask=tl.arange(0, D)[None, :] < d)

    if NATIVE:
        x_offs = (pid * DIM
                + tl.arange(0, GROUP_SIZE)[:, None] * d
                + tl.arange(0, D)[None, :])
    else:
        x_offs = (sid * bs * DIM
                + bid * DIM
                + tl.arange(0, GROUP_SIZE)[:, None] * d
                + tl.arange(0, D)[None, :])

    x_offs_mask = tl.arange(0, D)[None, :] < d
    x = tl.load(x_ptr + x_offs, mask=x_offs_mask).to(tl.float32)

    g_offs = (
        sid * bs * stride_g
        + bid * stride_g
        + tl.arange(0, GROUP_SIZE)[:, None] * d
        + tl.arange(0, D)[None, :]
    )

    g = tl.load(gate_ptr + g_offs, mask=tl.arange(0, D)[None, :] < d).to(tl.float32)

    rms = tl.rsqrt(tl.sum(x * x, axis=1) / d + eps)

    x = (x * rms[:, None]) * weight * tl.sigmoid(g)

    g_offs = (
        sid * bs * DIM
        + bid * DIM
        + tl.arange(0, GROUP_SIZE)[:, None] * d
        + tl.arange(0, D)[None, :]
    )
    tl.store(out_ptr + g_offs, x, mask=tl.arange(0, D)[None, :] < d)


def triton_group_rms_norm_gate_forward(x: torch.Tensor,
                                       gate: torch.Tensor,
                                       weight: torch.Tensor,
                                       eps=1e-6,
                                       group_size=4):
    """
    norm and gate in linear attention
    Args:
        x: output of attn, [bs, length, n_heads, head_dim]
        gate: gate tensor, [length, bs, dim] if transpose=True else [bs, length, dim]
        weight: rms norm weight, [dim]
        eps: epsilon of rms norm
        group_size: group size of group rms norm
        layout: layout of x, should in {'bsd', 'sbd}

    Returns:
        output tensor, [length, bs, dim]
    """
    length, bs, dim = gate.shape

    assert (dim <= 8192
            and triton.next_power_of_2(group_size) == group_size)
    assert x.is_contiguous() and weight.is_contiguous()
    assert gate.stride(2) == 1 and gate.stride(0) == gate.stride(1) * bs
    assert length != bs
    wd = weight.shape[0]
    SHARE = wd != dim  # all groups share the same weight
    NATIVE = x.size(0) == bs
    d = dim // group_size
    device = x.device

    D = triton.next_power_of_2(d)

    out = torch.empty((length, bs, dim), device=device, dtype=x.dtype)

    grid = (bs * length, )
    group_rms_norm_gate_forward_kernel[grid](
        x,
        gate,
        weight,
        out,
        gate.stride(1),
        eps,
        bs,
        length,
        dim,
        d,
        D,
        group_size,
        SHARE,
        NATIVE,
        num_stages=3,
        num_warps=4,
    )
    return out


@triton.jit
def group_rms_norm_gate_backward_kernel(
    grad_output_ptr,
    x_ptr,
    gate_ptr,
    w_ptr,
    dx_ptr,
    dg_ptr,
    dw_ptr,
    eps,
    bs,
    length,
    stride_g,
    DIM: tl.constexpr,
    d: tl.constexpr,
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    T: tl.constexpr,
    SHARE: tl.constexpr,
    NATIVE: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = pid * T // length
    sid = pid * T % length

    if SHARE:
        w = tl.load(w_ptr + tl.arange(0, d), mask=tl.arange(0, D) < D)[None, :]
    else:
        w = tl.load(w_ptr
                    + tl.arange(0, GROUP_SIZE)[:, None] * d
                    + tl.arange(0, D),
                    mask=tl.arange(0, D)[None, :] < d)

    if NATIVE:
        x_offs = (pid * DIM * T
                + tl.arange(0, GROUP_SIZE)[:, None] * d
                + tl.arange(0, D)[None, :])
    else:
        x_offs = (sid * bs * DIM
                + bid * DIM
                + tl.arange(0, GROUP_SIZE)[:, None] * d
                + tl.arange(0, D)[None, :])

    x_offs_mask = tl.arange(0, D)[None, :] < d
    offs = (
        sid * bs * DIM
        + bid * DIM
        + tl.arange(0, GROUP_SIZE)[:, None] * d
        + tl.arange(0, D)[None, :]
    )
    offs_mask = tl.arange(0, D)[None, :] < d
    g_offs = (
        sid * bs * stride_g
        + bid * stride_g
        + tl.arange(0, GROUP_SIZE)[:, None] * d
        + tl.arange(0, D)[None, :]
    )

    dw = tl.zeros((GROUP_SIZE, D), dtype=tl.float32)
    for i in range(T):
        x = tl.load(x_ptr + x_offs, mask=x_offs_mask).to(tl.float32)
        g = tl.load(grad_output_ptr + offs, mask=offs_mask).to(tl.float32)
        gate = tl.load(gate_ptr + g_offs, mask=offs_mask).to(tl.float32)
        gate = tl.sigmoid(gate)
        r = tl.rsqrt(tl.sum(x * x, 1) / d + eps)[:, None]
        w_grad = x * g * r * gate
        dw += w_grad

        dx = (
            r * g * w * gate
            - r * r * r * x * tl.sum(x * g * w * gate, 1, keep_dims=True) / d
        )

        tl.store(dx_ptr + x_offs, dx, mask=x_offs_mask)

        dg = x * r * w * g * gate * (1 - gate)
        tl.store(dg_ptr + offs, dg, mask=offs_mask)

        if NATIVE:
            x_offs += DIM
        else:
            x_offs += DIM * bs
        offs += DIM * bs
        g_offs += (bs * stride_g)

    if SHARE:
        dw = tl.sum(dw, 0)
        tl.store(dw_ptr + pid * d + tl.arange(0, d), dw,
                 mask=tl.arange(0, D) < d)
    else:
        tl.store(dw_ptr
                 + pid * DIM
                 + tl.arange(0, GROUP_SIZE)[:, None] * d
                 + tl.arange(0, D)[None, :],
                 dw,
                 mask=tl.arange(0, D)[None, :] < d)


def triton_group_rms_norm_gate_backward(grad_output,
                                        x,
                                        gate,
                                        weight,
                                        eps=1e-6,
                                        group_size=4):
    length, bs, dim = gate.shape
    assert dim <= 8192 and triton.next_power_of_2(group_size) == group_size
    assert grad_output.is_contiguous()
    assert length != bs
    d = dim // group_size
    wd = weight.shape[0]
    SHARE = wd != dim  # all groups share the same weight
    NATIVE = x.size(0) == bs

    device = x.device
    dx = torch.empty_like(x)
    dg = torch.empty_like(gate)

    T = 8
    g = (bs * length) // T
    if SHARE:
        tmp_dw = torch.empty(g, d, dtype=torch.float32, device=device)
    else:
        tmp_dw = torch.empty(g, dim, dtype=torch.float32, device=device)

    D = triton.next_power_of_2(d)
    grid = (g,)
    group_rms_norm_gate_backward_kernel[grid](
        grad_output,
        x,
        gate,
        weight,
        dx,
        dg,
        tmp_dw,
        eps,
        bs,
        length,
        gate.stride(1),
        dim,
        d,
        D,
        group_size,
        T,
        SHARE,
        NATIVE,
        num_stages=3,
        num_warps=8,
    )
    dw = tmp_dw.sum(dim=0)
    return dx, dg, dw

@triton.jit
def group_rms_norm_gate_and_mxfp8_quant_forward_kernel(
    x_ptr,
    gate_ptr,
    weight_ptr,
    x_q_ptr,
    x_s_ptr,
    xt_q_ptr,
    xt_s_ptr,
    stride_g,
    eps,
    bs,
    length,
    m,                      # bs * length
    DIM: tl.constexpr,
    d: tl.constexpr,        # TODO: add support for d not power of 2d, now d == D
    D: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SB: tl.constexpr,
    SHARE: tl.constexpr,
    NATIVE: tl.constexpr,   # True → x layout is [bs, length, DIM] or x layout is [length, bs, DIM]
    OUTPUT_MODE: tl.constexpr,
):
    rid = tl.program_id(axis=0)
    gid = tl.program_id(axis=1)

    out_rows = rid * 32 + tl.arange(0, 32)
    row_mask = out_rows < m
    col_mask = tl.arange(0, D)[None, :] < d

    if SHARE:
        w_off = tl.arange(0, D)
    else:
        w_off = gid * d + tl.arange(0, D)
    weight = tl.load(weight_ptr + w_off, mask=tl.arange(0, D) < d)

    if NATIVE:
        sids = out_rows // bs
        bids = out_rows % bs
        x_row_base = bids * length * DIM + sids * DIM
    else:
        x_row_base = out_rows * DIM         

    x_offs = x_row_base[:, None] + gid * d + tl.arange(0, D)[None, :] # [32, D]
    x = tl.load(x_ptr + x_offs,
                mask=row_mask[:, None] & col_mask).to(tl.float32)

    g_offs = out_rows[:, None] * stride_g + gid * d + tl.arange(0, D)[None, :]
    g = tl.load(gate_ptr + g_offs,
                mask=row_mask[:, None] & col_mask).to(tl.float32)
    
    rms = tl.rsqrt(tl.sum(x * x, axis=1) / d + eps) # [32] current group norm
    x = x * rms[:, None] * weight[None, :] * tl.sigmoid(g)  # [32, D]

    if OUTPUT_MODE % 2 == 0:
        x_blocks = tl.reshape(x, [32, SB, 32])
        scale_row = tl.maximum(
            tl.max(tl.abs(x_blocks), 2) / 448.0, 1e-30) # [32, SB]
        log_scale_row = tl.ceil(tl.log2(scale_row))
        x_q = tl.reshape(x_blocks / tl.exp2(log_scale_row)[:, :, None], [32, D])

        tl.store(x_q_ptr
                 + out_rows[:, None] * DIM
                 + gid * d
                 + tl.arange(0, D)[None, :],
                 x_q.to(x_q_ptr.dtype.element_ty),
                 mask=row_mask[:, None] & col_mask)

        tl.store(x_s_ptr
                 + out_rows[:, None] * (GROUP_SIZE * SB)
                 + gid * SB
                 + tl.arange(0, SB)[None, :],
                 (log_scale_row + 127).to(tl.uint8),
                 mask=row_mask[:, None])

    if OUTPUT_MODE > 0:
        scale_col = tl.maximum(
            tl.max(tl.abs(x), 0) / 448.0, 1e-30)
        log_scale_col = tl.ceil(tl.log2(scale_col))
        xt_q = x / tl.exp2(log_scale_col)[None, :]

        tl.store(xt_q_ptr
                 + out_rows[:, None] * DIM
                 + gid * d
                 + tl.arange(0, D)[None, :],
                 xt_q.to(xt_q_ptr.dtype.element_ty),
                 mask=row_mask[:, None] & col_mask)

        tl.store(xt_s_ptr
                 + rid * DIM
                 + gid * d
                 + tl.arange(0, D),
                 (log_scale_col + 127).to(tl.uint8),
                 mask=tl.arange(0, D) < d)


def triton_group_rms_norm_gate_and_mxfp8_quant_forward(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    group_size: int = 4,
    output_mode: int = 2,
):
    """
    Fused group RMSnorm + sigmoid-gate + MXFP8 quantization.

    """
    length, bs, dim = gate.shape
    m = length * bs
    M = (m + 127) // 128 * 128

    d = dim // group_size
    D = triton.next_power_of_2(d)
    assert D == d, (
        f"d = dim // group_size = {d} only support D==d now")
    assert d >= 32 and d % 32 == 0, \
        f"d = {d} must be >= 32 and divisible by 32"
    assert dim <= 8192 and triton.next_power_of_2(group_size) == group_size
    assert x.is_contiguous() and weight.is_contiguous()
    assert gate.stride(2) == 1 and gate.stride(0) == gate.stride(1) * bs

    SB = d // 32 # D // 32
    SHARE = weight.shape[0] != dim
    NATIVE = x.size(0) == bs
    device = x.device

    x_q  = torch.empty((m, dim), device=device, dtype=torch.float8_e4m3fn)
    x_s  = torch.empty((M, dim // 32), device=device, dtype=torch.uint8) 
    xt_q = torch.empty((m, dim), device=device, dtype=torch.float8_e4m3fn)
    xt_s = torch.empty((M // 32, dim), device=device, dtype=torch.uint8) 


    grid = (M // 32, group_size)
    group_rms_norm_gate_and_mxfp8_quant_forward_kernel[grid](
        x, gate, weight,
        x_q,
        x_s,
        xt_q,
        xt_s,
        gate.stride(1), eps, bs, length, m,
        dim, d, D, group_size, SB,
        SHARE, NATIVE, output_mode,
        num_stages=3,
        num_warps=4,
    )
    return x_q, x_s, xt_q, xt_s

@triton.jit
def group_rms_norm_gate_and_mxfp8_quant_backward_kernel(
    grad_output_ptr,
    x_ptr,
    gate_ptr,          # [length, bs, DIM]
    w_ptr,
    dx_ptr,
    dg_q_ptr,
    dg_s_ptr,
    dgt_q_ptr,
    dgt_s_ptr,
    tmp_dw_ptr,        # [num_row_blocks * GROUP_SIZE, D]  fp32
    stride_g,          # gate.stride(1)
    eps,
    bs,
    length,
    m,                 # = bs * length
    DIM: tl.constexpr,
    d: tl.constexpr,   # d = DIM // GROUP_SIZE
    D: tl.constexpr,   # next_power_of_2(d)
    GROUP_SIZE: tl.constexpr,
    SB: tl.constexpr,  # d // 32
    T: tl.constexpr,
    SHARE: tl.constexpr,
    NATIVE: tl.constexpr, # True → x layout is [bs, length, DIM] or x layout is [length, bs, DIM]
    OUTPUT_MODE: tl.constexpr,
):
    rid_base = tl.program_id(axis=0)   # eache block covers T*32 rows
    gid      = tl.program_id(axis=1)

    if SHARE:
        w = tl.load(w_ptr + tl.arange(0, D), mask=tl.arange(0, D) < d)
    else:
        w = tl.load(w_ptr + gid * d + tl.arange(0, D),
                    mask=tl.arange(0, D) < d)

    dw = tl.zeros([D], dtype=tl.float32)

    for t in tl.static_range(T):
        chunk_rid = rid_base * T + t
        out_rows  = chunk_rid * 32 + tl.arange(0, 32)
        row_mask  = out_rows < m
        col_mask  = tl.arange(0, D)[None, :] < d

        if NATIVE:
            sids = out_rows // bs
            bids = out_rows % bs
            x_row_base = bids * length * DIM + sids * DIM
        else:
            x_row_base = out_rows * DIM
        x_offs = x_row_base[:, None] + gid * d + tl.arange(0, D)[None, :]
        x = tl.load(x_ptr + x_offs,
                    mask=row_mask[:, None] & col_mask).to(tl.float32)

        base = out_rows[:, None] * DIM + gid * d + tl.arange(0, D)[None, :]
        go   = tl.load(grad_output_ptr + base,
                       mask=row_mask[:, None] & col_mask).to(tl.float32)
        gv   = tl.load(gate_ptr + out_rows[:, None] * stride_g
                       + gid * d + tl.arange(0, D)[None, :],
                       mask=row_mask[:, None] & col_mask).to(tl.float32)
        gs   = tl.sigmoid(gv)# [32, D]

        r  = tl.rsqrt(tl.sum(x * x, 1) / d + eps)
        r3 = r * r * r

        dw += tl.sum(x * go * r[:, None] * gs, 0) # [D]

        xgwgs = x * go * w[None, :] * gs
        dx = (r[:, None]  * go * w[None, :] * gs
              - r3[:, None] * x * tl.sum(xgwgs, 1, keep_dims=True) / d)

        tl.store(dx_ptr + x_offs, dx.to(tl.bfloat16),
                 mask=row_mask[:, None] & col_mask)

        dg = x * r[:, None] * w[None, :] * go * gs * (1.0 - gs)
        dg = tl.where(row_mask[:, None] & col_mask, dg, 0.0)

        if OUTPUT_MODE % 2 == 0:
            dg_b  = tl.reshape(dg, [32, SB, 32])
            s_row = tl.maximum(tl.max(tl.abs(dg_b), 2) / 448.0, 1e-30)
            ls_r  = tl.ceil(tl.log2(s_row))
            dg_qr = tl.reshape(dg_b / tl.exp2(ls_r)[:, :, None], [32, D])

            tl.store(dg_q_ptr
                     + out_rows[:, None] * DIM + gid * d + tl.arange(0, D)[None, :],
                     dg_qr.to(dg_q_ptr.dtype.element_ty),
                     mask=row_mask[:, None] & col_mask)
            tl.store(dg_s_ptr
                     + out_rows[:, None] * (GROUP_SIZE * SB)
                     + gid * SB + tl.arange(0, SB)[None, :],
                     (ls_r + 127).to(tl.uint8),
                     mask=row_mask[:, None])
            
        if OUTPUT_MODE > 0:
            s_col = tl.maximum(tl.max(tl.abs(dg), 0) / 448.0, 1e-30)
            ls_c  = tl.ceil(tl.log2(s_col))
            dg_qc = dg / tl.exp2(ls_c)[None, :]

            tl.store(dgt_q_ptr
                     + out_rows[:, None] * DIM + gid * d + tl.arange(0, D)[None, :],
                     dg_qc.to(dgt_q_ptr.dtype.element_ty),
                     mask=row_mask[:, None] & col_mask)
            tl.store(dgt_s_ptr + chunk_rid * DIM + gid * d + tl.arange(0, D),
                     (ls_c + 127).to(tl.uint8),
                     mask=tl.arange(0, D) < d)

    tmp_row = rid_base * GROUP_SIZE + gid
    tl.store(tmp_dw_ptr + tmp_row * D + tl.arange(0, D), dw,
             mask=tl.arange(0, D) < d)


def triton_group_rms_norm_gate_and_mxfp8_quant_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    group_size: int = 4,
    output_mode: int = 2,
):
    """
    Fused backward of group RMSnorm + sigmoid-gate + MXFP8 quantization.
    """
    length, bs, dim = gate.shape
    m = length * bs
    M = (m + 127) // 128 * 128

    lbs = length * bs
    T = 1 if lbs <= 8192 else 2**(math.ceil(math.log2(lbs / 8192)))

    d = dim // group_size
    D = triton.next_power_of_2(d)
    assert D == d, f"d = dim // group_size = {d} must be a power of 2"
    assert d >= 32 and d % 32 == 0, f"d = {d} must be >= 32 and divisible by 32"
    assert (M // 32) % T == 0, \
        f"M//32 = {M // 32} must be divisible by T = {T}"
    assert dim <= 8192 and triton.next_power_of_2(group_size) == group_size
    assert grad_output.is_contiguous()
    assert gate.stride(2) == 1 and gate.stride(0) == gate.stride(1) * bs
    assert length != bs

    SB     = d // 32
    SHARE  = weight.shape[0] != dim
    NATIVE = x.size(0) == bs
    device = x.device

    num_row_blocks = M // 32 // T

    dx    = torch.empty_like(x, dtype=torch.bfloat16)
    dg_q  = torch.empty((m, dim), device=device, dtype=torch.float8_e4m3fn)
    dg_s  = torch.empty((M, dim // 32), device=device, dtype=torch.uint8)
    dgt_q = torch.empty((m, dim), device=device, dtype=torch.float8_e4m3fn)
    dgt_s = torch.empty((M // 32, dim), device=device, dtype=torch.uint8)
    tmp_dw = torch.empty((num_row_blocks * group_size, D),
                         dtype=torch.float32, device=device)

    grid = (num_row_blocks, group_size)
    group_rms_norm_gate_and_mxfp8_quant_backward_kernel[grid](
        grad_output, x, gate, weight,
        dx,
        dg_q,
        dg_s,
        dgt_q,
        dgt_s,
        tmp_dw,
        gate.stride(1), eps, bs, length, m,
        dim, d, D, group_size, SB, T,
        SHARE, NATIVE, output_mode,
        num_stages=3,
        num_warps=4,
    )

    if SHARE:
        dw = tmp_dw.sum(0).to(x.dtype)
    else:
        dw = tmp_dw.view(num_row_blocks, group_size, D).sum(0).reshape(dim).to(x.dtype)
    return dx, dg_q, dg_s, dgt_q, dgt_s, dw