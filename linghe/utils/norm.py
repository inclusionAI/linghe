from re import S
import torch
import triton
import triton.language as tl



@triton.jit
def rms_norm_forward_kernel(x_ptr, weight_ptr, out_ptr, eps, M, T,
                            N: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(axis=0)
    weight = tl.load(weight_ptr + tl.arange(0, N)).to(tl.float32)[None, :]

    offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, N)[
                                                            None, :]
    for i in range(T):
        x = tl.load(x_ptr + offs,
                    mask=pid * W * T + i * W + tl.arange(0, W)[:, None] < M).to(
            tl.float32)
        rms = tl.sqrt(tl.sum(x * x, axis=1) / N + eps)

        x = (x / rms[:, None]) * weight

        tl.store(out_ptr + offs, x,
                 mask=pid * W * T + i * W + tl.arange(0, W)[:, None] < M)
        offs += N * W


def triton_rms_norm_forward(x, weight, eps=1e-6, out=None):
    # row-wise read, row-wise write
    M, N = x.shape
    W = 8192 // N
    T = 4
    assert N <= 8192 and M % (W*T) == 0
    device = x.device
    if out is None:
        out = torch.empty((M, N), device=device, dtype=x.dtype)

    grid = (M//(T*W),)
    rms_norm_forward_kernel[grid](
        x,
        weight,
        out,
        eps,
        M,
        T,
        N,
        W,
        num_stages=3,
        num_warps=4
    )
    return out


@triton.jit
def rms_norm_backward_kernel(
        grad_output_ptr,
        x_ptr,
        w_ptr,
        dx_ptr,
        dw_ptr,
        eps,
        M,
        T,
        N: tl.constexpr,
        W: tl.constexpr
):
    pid = tl.program_id(0)

    w = tl.load(w_ptr + tl.arange(0, N)).to(tl.float32)

    offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, N)[
                                                            None, :]
    w_grads = tl.zeros((N,), dtype=tl.float32)
    for i in range(T):
        mask = pid * T + i < M
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        g = tl.load(grad_output_ptr + offs, mask=mask).to(tl.float32)
        rms = tl.sqrt(tl.sum(x * x, 1) / N + eps)
        r = 1.0 / rms[:, None]
        w_grad = x * g * r
        w_grads += tl.sum(w_grad, 0)

        dx = r * g * w - r * r * r * x * tl.sum(x * g * w, 1, keep_dims=True) / N

        tl.store(dx_ptr + offs, dx, mask=mask)

        offs += N * W

    tl.store(dw_ptr + pid * N + tl.arange(0, N), w_grads)


def triton_rms_norm_backward(grad_output, x, w, eps=1e-6):
    M, N = x.shape
    dx = torch.empty(M, N, dtype=x.dtype, device=x.device)

    W = 8192 // N
    T = 16
    assert 8192 % N ==0 and M % (T*W) == 0
    g = M//(T*W)
    tmp_dw = torch.empty(g, N, dtype=torch.float32, device=w.device)
    grid = (g,)
    rms_norm_backward_kernel[grid](
        grad_output,
        x,
        w,
        dx,
        tmp_dw,
        eps,
        M,
        T,
        N,
        W,
        num_stages=3,
        num_warps=4
    )
    return dx, tmp_dw.sum(dim=0).to(w.dtype)


# output non-transposed and transposed together
# should used with batchsize >= 16384
@triton.jit
def rms_norm_and_block_quant_forward_kernel(x_ptr, 
                                      weight_ptr,
                                      out_ptr, 
                                      scale_ptr, 
                                      transpose_output_ptr, 
                                      transpose_scale_ptr,
                                      rms_ptr, 
                                      eps,
                                      M, 
                                      T: tl.constexpr,
                                      N: tl.constexpr,
                                      nb: tl.constexpr,
                                      W: tl.constexpr, 
                                      H : tl.constexpr,
                                      ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)

    # row-wise read, row-wise write
    weight = tl.load(weight_ptr + tl.arange(0, N)).to(tl.float32)[None, :]
    offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, N)[
                                                            None, :]
    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32)
        rms = 1/tl.sqrt(tl.sum(x * x, axis=1) / N + eps)
        tl.store(rms_ptr + indices, rms, mask=indices < M)
        x = (x * rms[:, None]) * weight
        x = tl.reshape(x, [W, nb, 128])
        scale = tl.maximum(tl.max(tl.abs(x), 2) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        x = (x / scale[:,:, None]).to(out_ptr.dtype.element_ty)
        x = tl.reshape(x, [W, N])
        tl.store(scale_ptr + indices[:, None] * nb + tl.arange(0, nb)[None, :], scale, mask=indices[:, None] < M)
        tl.store(out_ptr + offs, x, mask=indices[:, None] < M)
        offs += N * W


    offs = pid * W * T * N + tl.arange(0, 128)[:, None] * N + tl.arange(0, H)[
                                                            None, :]
    toffs = pid * 128 + tl.arange(0, H)[:, None] * M + tl.arange(0, 128)[
                                                            None, :]
    indices = pid * W * T + tl.arange(0, 128)
    tl.debug_barrier()
    rms = tl.load(rms_ptr + indices, mask=indices < M)[:, None]
    for i in range(N//H):
        x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32) 
        wgt = tl.load(weight_ptr + i * H +  tl.arange(0, H)).to(tl.float32)
        x = x * rms * wgt
        scale = tl.maximum(tl.max(x.abs(), 0) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(transpose_scale_ptr + pid * N + i * H + tl.arange(0, H), scale)
        x = (x/scale).to(transpose_output_ptr.dtype.element_ty)
        tl.store(transpose_output_ptr + toffs, tl.trans(x), mask=indices[None, :] < M)
        offs += H
        toffs += M * H



# output non-transposed tensor only
@triton.jit
def rms_norm_and_block_quant_forward_n_kernel(x_ptr, 
                                      weight_ptr,
                                      out_ptr, 
                                      scale_ptr, 
                                      rms_ptr, 
                                      eps,
                                      M: tl.constexpr, 
                                      T: tl.constexpr,
                                      N: tl.constexpr,
                                      nb: tl.constexpr,
                                      W: tl.constexpr, 
                                      ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)

    # row-wise read, row-wise write
    weight = tl.load(weight_ptr + tl.arange(0, N)).to(tl.float32)[None, :]
    offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, N)[
                                                            None, :]
    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32)
        rms = tl.rsqrt(tl.sum(x * x, axis=1) / N + eps)
        tl.store(rms_ptr + indices, rms, mask=indices < M)

        x = x * rms[:, None] * weight
        x = tl.reshape(x, [W, nb, 128])
        scale = tl.maximum(tl.max(tl.abs(x), 2) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        # x = (x / scale[:,:, None]).to(out_ptr.dtype.element_ty)
        # x = tl.reshape(x, [W, N])

        x = x / scale[:,:, None]
        x = tl.reshape(x, [W, N])

        tl.store(scale_ptr + indices[:, None] * nb + tl.arange(0, nb)[None, :], scale, mask=indices[:, None] < M)
        tl.store(out_ptr + offs, x, mask=indices[:, None] < M)
        offs += N * W


# output transposed tensor only
@triton.jit
def rms_norm_and_block_quant_forward_t_kernel(x_ptr, 
                                      weight_ptr,
                                      transpose_output_ptr, 
                                      transpose_scale_ptr,
                                      rms_ptr, 
                                      M, 
                                      N,
                                      W: tl.constexpr, 
                                      ROUND: tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)

    offs = rid * 128 * N + cid * W + tl.arange(0, 128)[:, None] * N + tl.arange(0, W)[
                                                            None, :]
    toffs = rid * 128 + cid * M * W + tl.arange(0, W)[:, None] * M + tl.arange(0, 128)[
                                                            None, :]

    weight = tl.load(weight_ptr + cid * W + tl.arange(0, W)).to(tl.float32)
    indices = rid * 128 + tl.arange(0, 128)
    rms = tl.load(rms_ptr + indices, mask=indices < M)[:, None]
    x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32) 
    x = x * rms * weight
    scale = tl.maximum(tl.max(x.abs(), 0) / 448.0, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(transpose_scale_ptr + rid * N + cid * W + tl.arange(0, W), scale)

    x = (tl.trans(x/scale)).to(transpose_output_ptr.dtype.element_ty)
    tl.store(transpose_output_ptr + toffs, x, mask=indices[None, :] < M)



def triton_rms_norm_and_block_quant_forward(x, weight, eps=1e-6,
                                      out=None, scale=None, rms=None,
                                      round_scale=False,
                                      output_mode=2):
    # row-wise read, row-wise write
    M, N = x.shape
    assert N <= 8192 and 8192 % N == 0
    device = x.device

    if out is None and  output_mode in (0, 2):
        out = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)

    if scale is None and output_mode in (0, 2):
        scale = torch.empty((M, N//128), device=device, dtype=torch.float32)
    if rms is None:
        rms = torch.empty((M,), dtype=torch.float32, device=device)
    # transpose_output should be initialized, or else can not make splitted tensors
    transpose_output = torch.empty((N, M), device=device, dtype=torch.float8_e4m3fn)
    transpose_scale = torch.empty(((M+127)//128, N), device=device, dtype=torch.float32)
    if output_mode == 0: # only output non-transpose tensor
        W = 8192 // N
        T = 16 // W  
        grid = (triton.cdiv(M, 16),)
        rms_norm_and_block_quant_forward_n_kernel[grid](
            x,
            weight,
            out,
            scale,
            rms,
            eps,
            M,
            T,
            N,
            N//128,
            W,
            round_scale,
            num_stages=3,
            num_warps=4
        )
        scale = scale.t().contiguous()

    elif output_mode == 1:  # only output transposed tensor
        # W = N//512
        # grid = (512,)
        W = 32 
        grid = (triton.cdiv(M, 128), N//W)
        rms_norm_and_block_quant_forward_t_kernel[grid](x, 
                                    weight,
                                    transpose_output, 
                                    transpose_scale,
                                    rms, 
                                    M, 
                                    N,
                                    W, 
                                    round_scale,
                                    num_stages=3,
                                    num_warps=4)
    
    elif output_mode == 2:  # output non-transposed and transposed tensor together
        W = 8192 // N
        T = 128 // W  # BLOCK SIZE
        H = 64
        grid = (triton.cdiv(M, 128),)
        rms_norm_and_block_quant_forward_kernel[grid](
            x,
            weight,
            out,
            scale,
            transpose_output,
            transpose_scale,
            rms,
            eps,
            M,
            T,
            N,
            N//128,
            W,
            H,
            round_scale,
            num_stages=3,
            num_warps=16
        )
        scale = scale.t().contiguous()

    return out, scale, rms, transpose_output, transpose_scale


# TOOD(nanxiao): opt performance
@triton.jit
def group_norm_gate_forward_kernel(x_ptr, gate_ptr, weight_ptr, out_ptr, eps, bs, length,
                            DIM: tl.constexpr, 
                            D: tl.constexpr, 
                            GROUP_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    bid = pid // length 
    sid = pid % length

    weight = tl.load(weight_ptr + tl.arange(0, DIM))
    weight = tl.reshape(weight, [GROUP_SIZE, D])

    x_offs = pid * DIM + tl.arange(0, GROUP_SIZE)[:, None] * D + tl.arange(0, D)[
                                                            None, :]
    x = tl.load(x_ptr + x_offs).to(tl.float32)
    offs = sid * bs * DIM + bid * DIM + tl.arange(0, GROUP_SIZE)[:, None] * D + tl.arange(0, D)[
                                                            None, :]
    g = tl.load(gate_ptr + offs).to(tl.float32)
    rms = tl.sqrt(tl.sum(x * x, axis=1) / D + eps)

    x = (x / rms[:, None]) * weight * tl.sigmoid(g)

    tl.store(out_ptr + offs, x)


"""
x: [bs, length, n_heads, head_dim], output of attn
gate: [length, bs, dim]
weight: [dim]
output: [length, bs, dim]
"""
def triton_group_norm_gate_forward(x, gate, weight, eps=1e-6, group_size=4):
    # row-wise read, row-wise write
    length, bs, dim = gate.shape
    assert dim <= 8192 and triton.next_power_of_2(dim) == dim and triton.next_power_of_2(group_size) == group_size
    d = dim // group_size
    device = x.device
    out = torch.empty((length, bs, dim), device=device, dtype=x.dtype)

    grid = (bs*length,)
    group_norm_gate_forward_kernel[grid](
        x,
        gate,
        weight.data,
        out,
        eps,
        bs,
        length,
        dim, 
        d,
        group_size,
        num_stages=3,
        num_warps=4
    )
    return out


@triton.jit
def group_rms_gate_backward_kernel(
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
        D: tl.constexpr, 
        GROUP_SIZE: tl.constexpr,
        T: tl.constexpr
):
    pid = tl.program_id(0)
    bid = pid * T // length 
    sid = pid * T % length

    w = tl.load(w_ptr + tl.arange(0, DIM))
    w = tl.reshape(w, [GROUP_SIZE, D])

    x_offs = pid * DIM * T + tl.arange(0, GROUP_SIZE)[:, None] * D + tl.arange(0, D)[
                                                            None, :]
    offs = sid * bs * DIM + bid * DIM + tl.arange(0, GROUP_SIZE)[:, None] * D + tl.arange(0, D)[
                                                            None, :]
    dw = tl.zeros((GROUP_SIZE, D), dtype=tl.float32)
    for i in range(T):
        x = tl.load(x_ptr + x_offs).to(tl.float32)
        g = tl.load(grad_output_ptr + offs).to(tl.float32)
        gate = tl.load(gate_ptr + offs).to(tl.float32)
        gate = tl.sigmoid(gate)
        rms = tl.sqrt(tl.sum(x * x, 1) / D + eps)
        r = 1.0 / rms[:, None]
        w_grad = x * g * r * gate
        dw += w_grad

        dx = r * g * w * gate - r * r * r * x * tl.sum(x * g * w * gate, 1, keep_dims=True) / D

        tl.store(dx_ptr + x_offs, dx)

        dg = x * r * w * g * gate * (1 - gate)
        tl.store(dg_ptr + offs, dg)

        x_offs += DIM
        offs += DIM * bs

    dw = tl.reshape(dw, [DIM])
    tl.store(dw_ptr + pid * DIM + tl.arange(0, DIM), dw)


def triton_group_norm_gate_backward(grad_output, x, gate, weight, eps=1e-6, group_size=4):
    length, bs, dim = gate.shape
    assert dim <= 8192 and triton.next_power_of_2(dim) == dim and triton.next_power_of_2(group_size) == group_size
    d = dim // group_size
    device = x.device
    dx = torch.empty_like(x)
    dg = torch.empty_like(gate)

    T = 8
    g = (bs*length)//T
    tmp_dw = torch.empty(g, dim, dtype=torch.float32, device=device)
    grid = (g,)
    group_rms_gate_backward_kernel[grid](
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
        group_size,
        T,
        num_stages=3,
        num_warps=8
    )
    dw = tmp_dw.sum(dim=0).to(weight.dtype)
    return dx, dg, dw


@triton.jit
def rms_norm_and_mxfp8_quant_forward_n_kernel(x_ptr, 
                                      weight_ptr,
                                      out_ptr, 
                                      scale_ptr, 
                                      rms_ptr, 
                                      eps,
                                      n, 
                                      M: tl.constexpr, 
                                      T: tl.constexpr,
                                      N: tl.constexpr,
                                      nb: tl.constexpr,
                                      W: tl.constexpr, 
                                      ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)

    # row-wise read, row-wise write
    weight_mask = tl.arange(0, N) < n
    weight = tl.load(weight_ptr + tl.arange(0, N),  mask=weight_mask).to(tl.float32)[None, :]
    offs_m = tl.arange(0, W)
    offs_n = tl.arange(0, N)
    offs = pid * W * T * n + offs_m[:, None] * n + offs_n[None, :]
    # offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, N)[None, :]
    
    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        mask = (indices[:, None] < M) & (offs_n < n)
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        # x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32)
        rms = tl.rsqrt(tl.sum(x * x, axis=1) / n + eps)
        tl.store(rms_ptr + indices, rms, mask=indices < M)

        x = x * rms[:, None] * weight
        x = tl.reshape(x, [W, nb, 32])
        scale = tl.maximum(tl.max(tl.abs(x), 2) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        # x = (x / scale[:,:, None]).to(out_ptr.dtype.element_ty)
        # x = tl.reshape(x, [W, N])
        x = x / scale[:,:, None]
        x = tl.reshape(x, [W, N])

        # tl.store(scale_ptr + indices[:, None] * nb + tl.arange(0, nb)[None, :], scale, mask=indices[:, None] < M)
        # tl.store(out_ptr + offs, x, mask=indices[:, None] < M)
        scale_mask = (indices[:, None] < M) & (tl.arange(0, nb)[None, :] < (n//32))
        tl.store(scale_ptr + indices[:, None] * (n//32) + tl.arange(0, nb)[None, :], scale, mask=scale_mask)
        tl.store(out_ptr + offs, x, mask=mask)
        offs += n * W

@triton.jit
def rms_norm_and_mxfp8_quant_forward_t_kernel(x_ptr, 
                                      weight_ptr,
                                      transpose_output_ptr, 
                                      transpose_scale_ptr,
                                      rms_ptr, 
                                      n,
                                      M, 
                                      N,
                                      W: tl.constexpr, 
                                      ROUND: tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)

    offs = rid * 32 * n + cid * W + tl.arange(0, 32)[:, None] * n + tl.arange(0, W)[
                                                            None, :]
    toffs = rid * 32 + cid * M * W + tl.arange(0, W)[:, None] * M + tl.arange(0, 32)[
                                                            None, :]

    weight = tl.load(weight_ptr + cid * W + tl.arange(0, W)).to(tl.float32)
    indices = rid * 32 + tl.arange(0, 32)
    rms = tl.load(rms_ptr + indices, mask=indices < M)[:, None]
    x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32) 
    x = x * rms * weight
    scale = tl.maximum(tl.max(x.abs(), 0) / 448.0, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(transpose_scale_ptr + rid * n + cid * W + tl.arange(0, W), scale)

    #transposed scale
    # tl.store(transpose_scale_ptr + cid * (M+31)//32 * W  + rid + tl.arange(0, W)[:, None] * ((M+31)//32) , scale.reshape(W,1))

    x = (tl.trans(x/scale)).to(transpose_output_ptr.dtype.element_ty)
    tl.store(transpose_output_ptr + toffs, x, mask=indices[None, :] < M)

@triton.jit
def rms_norm_and_mxfp8_quant_forward_kernel(x_ptr, 
                                      weight_ptr,
                                      out_ptr, 
                                      scale_ptr, 
                                      transpose_output_ptr, 
                                      transpose_scale_ptr,
                                      rms_ptr, 
                                      eps,
                                      n,
                                      M, 
                                      T: tl.constexpr,
                                      N: tl.constexpr,
                                      nb: tl.constexpr,
                                      W: tl.constexpr, 
                                      H : tl.constexpr,
                                      ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)

    # row-wise read, row-wise write
    weight_mask = tl.arange(0, N) < n
    weight = tl.load(weight_ptr + tl.arange(0, N),  mask=weight_mask).to(tl.float32)[None, :]
    offs_m = tl.arange(0, W)
    offs_n = tl.arange(0, N)
    offs = pid * W * T * n + offs_m[:, None] * n + offs_n[None, :]
    # offs = pid * W * T * N + tl.arange(0, W)[:, None] * N + tl.arange(0, N)[None, :]

    for i in range(T):
        indices = pid * W * T + i * W + tl.arange(0, W)
        mask = (indices[:, None] < M) & (offs_n < n)
        x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
        # x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32)
        rms = tl.rsqrt(tl.sum(x * x, axis=1) / n + eps)
        tl.store(rms_ptr + indices, rms, mask=indices < M)

        x = x * rms[:, None] * weight
        x = tl.reshape(x, [W, nb, 32])
        scale = tl.maximum(tl.max(tl.abs(x), 2) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        # x = (x / scale[:,:, None]).to(out_ptr.dtype.element_ty)
        # x = tl.reshape(x, [W, N])
        x = x / scale[:,:, None]
        x = tl.reshape(x, [W, N])

        # tl.store(scale_ptr + indices[:, None] * nb + tl.arange(0, nb)[None, :], scale, mask=indices[:, None] < M)
        # tl.store(out_ptr + offs, x, mask=indices[:, None] < M)
        scale_mask = (indices[:, None] < M) & (tl.arange(0, nb)[None, :] < (n//32))
        tl.store(scale_ptr + indices[:, None] * (n//32) + tl.arange(0, nb)[None, :], scale, mask=scale_mask)
        tl.store(out_ptr + offs, x, mask=mask)
        offs += n * W


    offs = pid * 32 * n + tl.arange(0, 32)[:, None] * n + tl.arange(0, H)[
                                                            None, :]
    toffs = pid * 32 + tl.arange(0, H)[:, None] * M + tl.arange(0, 32)[
                                                            None, :]
    indices = pid * 32 + tl.arange(0, 32)
    tl.debug_barrier()
    rms = tl.load(rms_ptr + indices, mask=indices < M)[:, None]
    for i in range(n//H):
        x = tl.load(x_ptr + offs, mask=indices[:, None] < M).to(tl.float32) 
        wgt = tl.load(weight_ptr + i * H +  tl.arange(0, H)).to(tl.float32)
        x = x * rms * wgt
        scale = tl.maximum(tl.max(x.abs(), 0) / 448.0, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(transpose_scale_ptr + pid * n + i * H + tl.arange(0, H), scale)
        x = (x/scale).to(transpose_output_ptr.dtype.element_ty)
        tl.store(transpose_output_ptr + toffs, tl.trans(x), mask=indices[None, :] < M)
        offs += H
        toffs += M * H



def triton_rms_norm_and_mxfp8_quant_forward(x, weight, eps=1e-6,
                                            out=None, scale=None, rms=None, round_scale=False,
                                            output_mode=2):
    # row-wise read, row-wise write
    M, n = x.shape
    N = triton.next_power_of_2(n)
    # assert N <= 8192 and 8192 % N == 0
    assert n % 32 == 0 
    device = x.device

    if out is None and  output_mode in (0, 2):
        out = torch.empty((M, n), device=device, dtype=torch.float8_e4m3fn)

    if scale is None and output_mode in (0, 2):
        scale = torch.empty((M, n//32), device=device, dtype=torch.float32)
    if rms is None:
        rms = torch.empty((M,), dtype=torch.float32, device=device)
    # transpose_output should be initialized, or else can not make splitted tensors
    transpose_output = torch.empty((n, M), device=device, dtype=torch.float8_e4m3fn)
    transpose_scale = torch.empty(((M+31)//32, n), device=device, dtype=torch.float32)#not trans
    # transpose_scale = torch.empty(n, ((M+31)//32), device=device, dtype=torch.float32)
    if output_mode == 0: # only output non-transpose tensor
        W = 8192 // N
        T = 16 // W  
        grid = (triton.cdiv(M, 16),)
        rms_norm_and_mxfp8_quant_forward_n_kernel[grid](
            x,
            weight,
            out,
            scale,
            rms,
            eps,
            n, 
            M,
            T,
            N,#N
            N//32,#N//32
            W,
            round_scale,
            num_stages=3,
            num_warps=4
        )
        scale = scale.t().contiguous()

    elif output_mode == 1:  # only output transposed tensor
        # W = N//512
        # grid = (512,)
        W = 64
        grid = (triton.cdiv(M, 32), n//W)
        rms_norm_and_mxfp8_quant_forward_t_kernel[grid](x, 
                                    weight,
                                    transpose_output, 
                                    transpose_scale,
                                    rms, 
                                    n,
                                    M, 
                                    N,
                                    W, 
                                    round_scale,
                                    num_stages=3,
                                    num_warps=4)
    
    elif output_mode == 2:  # output non-transposed and transposed tensor together
        W = 8192 // N  #一个grid一次最多几行
        T = 32 // W  # BLOCK SIZE 一个grid处理32行的话要重复几次
        H = 64
        grid = (triton.cdiv(M, 32),)
        rms_norm_and_mxfp8_quant_forward_kernel[grid](
            x,
            weight,
            out,
            scale,
            transpose_output,
            transpose_scale,
            rms,
            eps,
            n,
            M,
            T,
            N,
            N//32,
            W,
            H,
            round_scale,
            num_stages=3,
            num_warps=16
        )
        scale = scale.t().contiguous()

    return out, scale, rms, transpose_output, transpose_scale