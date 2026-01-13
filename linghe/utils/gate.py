import torch
import triton
import triton.language as tl


# TOOD(nanxiao): opt performance
@triton.jit
def group_rms_norm_gate_forward_kernel(x_ptr, gate_ptr, weight_ptr, out_ptr,
                                       eps, bs, length,
                                       DIM: tl.constexpr,
                                       d: tl.constexpr,
                                       D: tl.constexpr,
                                       GROUP_SIZE: tl.constexpr,
                                       SHARE: tl.constexpr,
                                       TRANSPOSE: tl.constexpr):
    pid = tl.program_id(axis=0)
    bid = pid // length
    sid = pid % length

    if SHARE:
        weight = tl.load(weight_ptr + tl.arange(0, D),
                         mask=tl.arange(0, D) < d)[None, :]
    else:
        weight = tl.load(
            weight_ptr + tl.arange(0, GROUP_SIZE)[:, None] * d + tl.arange(0,
                                                                           D),
            mask=tl.arange(0, D)[None, :] < d)

    x_offs = (
            pid * DIM + tl.arange(0, GROUP_SIZE)[:, None] * d + tl.arange(0, D)[
                                                                None, :]
    )
    x_offs_mask = tl.arange(0, D)[None, :] < d
    x = tl.load(x_ptr + x_offs, mask=x_offs_mask).to(tl.float32)
    if TRANSPOSE:
        g_offs = (
                sid * bs * DIM
                + bid * DIM
                + tl.arange(0, GROUP_SIZE)[:, None] * d
                + tl.arange(0, D)[None, :]
        )
        g = tl.load(gate_ptr + g_offs, mask=tl.arange(0, D)[None, :] < d).to(
            tl.float32)
    else:
        g = tl.load(gate_ptr + x_offs, mask=x_offs_mask).to(tl.float32)

    rms = tl.rsqrt(tl.sum(x * x, axis=1) / d + eps)

    x = (x * rms[:, None]) * weight * tl.sigmoid(g)

    if TRANSPOSE:
        tl.store(out_ptr + g_offs, x, mask=tl.arange(0, D)[None, :] < d)
    else:
        tl.store(out_ptr + x_offs, x, mask=x_offs_mask)


def triton_group_rms_norm_gate_forward(x: torch.Tensor,
                                       gate: torch.Tensor,
                                       weight: torch.Tensor,
                                       eps=1e-6,
                                       group_size=4,
                                       transpose=True):
    """
    norm and gate in linear attention
    Args:
        x: output of attn, [bs, length, n_heads, head_dim]
        gate: gate tensor, [length, bs, dim] if transpose=True else [bs, length, dim]
        weight: rms norm weight, [dim]
        eps: epsilon of rms norm
        group_size: group size of group rms norm
        transpose: whether gate tensor has been transposed and output will be transposed

    Returns:
        output tensor, [length, bs, dim] if transpose=True else [bs, length, dim]
    """
    # row-wise read, row-wise write
    if transpose:
        length, bs, dim = gate.shape
    else:
        bs, length, dim = gate.shape
    assert (dim <= 8192
            and triton.next_power_of_2(group_size) == group_size)
    assert x.is_contiguous() and gate.is_contiguous() and weight.is_contiguous()
    wd = weight.shape[0]
    share = wd != dim  # all groups share the same weight
    d = dim // group_size
    device = x.device

    D = triton.next_power_of_2(d)

    if transpose:
        out = torch.empty((length, bs, dim), device=device, dtype=x.dtype)
    else:
        out = torch.empty((bs, length, dim), device=device, dtype=x.dtype)

    grid = (bs * length,)
    group_rms_norm_gate_forward_kernel[grid](
        x,
        gate,
        weight,
        out,
        eps,
        bs,
        length,
        dim,
        d,
        D,
        group_size,
        share,
        transpose,
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
        DIM: tl.constexpr,
        d: tl.constexpr,
        D: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        T: tl.constexpr,
        SHARE: tl.constexpr,
        TRANSPOSE: tl.constexpr
):
    pid = tl.program_id(0)
    bid = pid * T // length
    sid = pid * T % length

    if SHARE:
        w = tl.load(w_ptr + tl.arange(0, d), mask=tl.arange(0, D) < D)[None, :]
    else:
        w = tl.load(
            w_ptr + tl.arange(0, GROUP_SIZE)[:, None] * d + tl.arange(0, D),
            mask=tl.arange(0, D)[None, :] < d,
        )

    x_offs = (
            pid * DIM * T + tl.arange(0, GROUP_SIZE)[:, None] * d + tl.arange(0,
                                                                              D)[
                                                                    None, :]
    )
    x_offs_mask = tl.arange(0, D)[None, :] < d
    if TRANSPOSE:
        offs = (
                sid * bs * DIM
                + bid * DIM
                + tl.arange(0, GROUP_SIZE)[:, None] * d
                + tl.arange(0, D)[None, :]
        )
        offs_mask = tl.arange(0, D)[None, :] < d

    dw = tl.zeros((GROUP_SIZE, D), dtype=tl.float32)
    for i in range(T):
        x = tl.load(x_ptr + x_offs, mask=x_offs_mask).to(tl.float32)
        if TRANSPOSE:
            g = tl.load(grad_output_ptr + offs, offs_mask).to(tl.float32)
            gate = tl.load(gate_ptr + offs, offs_mask).to(tl.float32)
        else:
            g = tl.load(grad_output_ptr + x_offs, mask=x_offs_mask).to(
                tl.float32)
            gate = tl.load(gate_ptr + x_offs, mask=x_offs_mask).to(tl.float32)
        gate = tl.sigmoid(gate)
        r = tl.rsqrt(tl.sum(x * x, 1) / d + eps)[:, None]
        w_grad = x * g * r * gate
        dw += w_grad

        dx = (
                r * g * w * gate
                - r * r * r * x * tl.sum(x * g * w * gate, 1,
                                         keep_dims=True) / d
        )

        tl.store(dx_ptr + x_offs, dx, mask=x_offs_mask)

        dg = x * r * w * g * gate * (1 - gate)
        if TRANSPOSE:
            tl.store(dg_ptr + offs, dg, mask=offs_mask)
        else:
            tl.store(dg_ptr + x_offs, dg, mask=x_offs_mask)

        x_offs += DIM
        if TRANSPOSE:
            offs += DIM * bs

    if SHARE:
        dw = tl.sum(dw, 0)
        tl.store(dw_ptr + pid * d + tl.arange(0, d), dw,
                 mask=tl.arange(0, D) < d)
    else:
        tl.store(
            dw_ptr
            + pid * DIM
            + tl.arange(0, GROUP_SIZE)[:, None] * d
            + tl.arange(0, D)[None, :],
            dw,
            mask=tl.arange(0, D)[None, :] < d,
        )


def triton_group_rms_norm_gate_backward(grad_output, x, gate, weight, eps=1e-6,
                                        group_size=4, transpose=True):
    if transpose:
        length, bs, dim = gate.shape
    else:
        bs, length, dim = gate.shape
    assert dim <= 8192 and triton.next_power_of_2(group_size) == group_size
    assert grad_output.is_contiguous()
    d = dim // group_size
    wd = weight.shape[0]
    share = wd != dim  # all groups share the same weight

    device = x.device
    dx = torch.empty_like(x)
    dg = torch.empty_like(gate)

    T = 8
    g = (bs * length) // T
    if share:
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
        dim,
        d,
        D,
        group_size,
        T,
        share,
        transpose,
        num_stages=3,
        num_warps=8
    )
    dw = tmp_dw.sum(dim=0)
    return dx, dg, dw
