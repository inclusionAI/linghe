import torch
import triton
import triton.language as tl




# n is power of 2
@triton.jit
def silu_and_smooth_quant_forward_kernel(x_ptr, smooth_scale_ptr, out_ptr, scale_ptr,
                                  max_ptr, M, T, n: tl.constexpr,
                                  W: tl.constexpr, ROUND: tl.constexpr,
                                  CALIBRATE: tl.constexpr):
    pid = tl.program_id(axis=0)

    row_offs = pid * T * W * n + tl.arange(0, W)[:, None] * n
    col_offs = tl.arange(0, n)[None, :]
    smooth_scale = tl.load(smooth_scale_ptr + tl.arange(0, n))
    smooth_scale = 1.0 / smooth_scale
    if CALIBRATE:
        maxs = tl.zeros((W, n), dtype=tl.float32)

    for i in range(T):
        indices = pid * T * W + i * W + tl.arange(0, W)
        mask = indices[:, None] < M
        x1 = tl.load(x_ptr + row_offs * 2 + col_offs, mask=mask).to(tl.float32)
        x2 = tl.load(x_ptr + n + row_offs * 2 + col_offs, mask=mask).to(
            tl.float32)
        x = x1 / (1 + tl.exp(-x1)) * x2
        if CALIBRATE:
            maxs = tl.maximum(x.abs(), maxs)
        x = x * smooth_scale
        scale = tl.maximum(tl.max(x.abs(), 1) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(scale_ptr + indices, scale, mask=indices < M)
        x = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row_offs + col_offs, x, mask=mask)
        row_offs += n * W

    if CALIBRATE:
        maxs = tl.max(maxs, 0)
        tl.store(max_ptr + pid * n + tl.arange(0, n), maxs)


# n is NOT power of 2
@triton.jit
def compatible_silu_and_smooth_quant_forward_kernel(x_ptr, smooth_scale_ptr, out_ptr,
                                             scale_ptr, max_ptr, M,
                                             T: tl.constexpr, n: tl.constexpr,
                                             B: tl.constexpr,
                                             ROUND: tl.constexpr,
                                             CALIBRATE: tl.constexpr):
    pid = tl.program_id(axis=0)

    # rowwise read with block size [T, B]
    row_offs = pid * T * n + tl.arange(0, T)[:, None] * n
    col_offs = tl.arange(0, B)[None, :]

    nb = n // B
    maxs = tl.zeros((T,), dtype=tl.float32)
    for i in range(nb):

        smooth_scale = tl.load(smooth_scale_ptr + i * B + tl.arange(0, B))
        x1 = tl.load(x_ptr + row_offs * 2 + col_offs).to(tl.float32)
        x2 = tl.load(x_ptr + n + row_offs * 2 + col_offs).to(tl.float32)
        x = x1 / (1 + tl.exp(-x1)) * x2
        if CALIBRATE:
            x_maxs = tl.max(x.abs(), 0)
            tl.store(max_ptr + pid * n + i * B + tl.arange(0, B), x_maxs)
        x = x / smooth_scale
        maxs = tl.maximum(tl.max(x.abs(), 1), maxs)
        col_offs += B

    scale = tl.maximum(maxs / 448, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(scale_ptr + pid * T + tl.arange(0, T), scale)

    col_offs = tl.arange(0, B)[None, :]
    for i in range(nb):
        smooth_scale = tl.load(smooth_scale_ptr + i * B + tl.arange(0, B))

        x1 = tl.load(x_ptr + row_offs * 2 + col_offs).to(tl.float32)
        x2 = tl.load(x_ptr + n + row_offs * 2 + col_offs).to(tl.float32)
        x = x1 / (1 + tl.exp(-x1)) * x2
        x = x / smooth_scale

        x = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row_offs + col_offs, x)
        col_offs += B


@triton.jit
def silu_and_block_quant_forward_kernel(x_ptr, 
                                        out_ptr, scale_ptr,
                                        transpose_output_ptr, 
                                        transpose_scale_ptr,
                                        M, 
                                        n: tl.constexpr, 
                                        ROUND: tl.constexpr,
                                        OUTPUT_MODE: tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)

    offs = rid * 128 * n * 2 + cid * 128 + tl.arange(0, 128)[:, None] * n * 2 + tl.arange(0, 128)[None, :]
    indices = rid * 128 + tl.arange(0, 128)
    mask = indices[:, None] < M


    x1 = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    x2 = tl.load(x_ptr + n + offs, mask=mask).to(tl.float32)
    x = x1 * tl.sigmoid(x1) * x2
    # x1 = tl.load(x_ptr + offs, mask=mask)
    # x2 = tl.load(x_ptr + n + offs, mask=mask)
    # x = tl.sigmoid(x1.to(tl.float32)) * x1 * x2

    if OUTPUT_MODE % 2 == 0:
        scale = tl.maximum(tl.max(x.abs(), 1) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        
        tl.store(scale_ptr + rid * 128 + cid * M + tl.arange(0, 128), scale, mask=indices < M)
        xq = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + rid * 128 * n + cid * 128 + tl.arange(0, 128)[:, None] * n + tl.arange(0, 128)[None, :], xq, mask=mask)

    if OUTPUT_MODE > 0:
        scale = tl.maximum(tl.max(x.abs(), 0) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(transpose_scale_ptr + rid * n + cid * 128 + tl.arange(0, 128), scale)
        xq = (x / scale).to(out_ptr.dtype.element_ty)
        tl.store(transpose_output_ptr + rid * 128 + cid * 128 * M + tl.arange(0, 128)[:, None] * M + tl.arange(0, 128)[
                                                            None, :], tl.trans(xq), mask=indices[None, :] < M)




# used in shared expert
def triton_silu_and_quant_forward(x, smooth_scale=None, out=None, scale=None,
                                  maxs=None, round_scale=False,
                                  calibrate=False, output_mode=2):
    # row-wise read, row-wise write
    M, N = x.shape
    n = N // 2
    device = x.device
    smooth = smooth_scale is not None 
    if out is None:
        out = torch.empty((M, N // 2), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        if smooth:
            scale = torch.empty((M,), device=device, dtype=torch.float32)
        else:
            scale = torch.empty((N // 2 // 128, M), device=device, dtype=torch.float32)

    if not smooth:
        transpose_output = torch.empty((N // 2, M), device=device, dtype=torch.float8_e4m3fn) 
        transpose_scale = torch.empty((triton.cdiv(M, 128), N // 2), device=device, dtype=torch.float32)
    else:
        transpose_output = None 
        transpose_scale = None

    if smooth:
        if triton.next_power_of_2(N) == N and N <= 8192:
            # sm = torch.cuda.get_device_properties(device).multi_processor_count
            W = 8192 // N
            T = 8 if M//W >= 1024 else 4
            assert M % (T*W) == 0
            g = M//(T*W)
            # T = triton.cdiv(M, sm * W)
            if maxs is None and calibrate:
                maxs = torch.empty((g, n), device=device, dtype=torch.float32)
            grid = (g,)
            silu_and_smooth_quant_forward_kernel[grid](
                x,
                smooth_scale,
                out,
                scale,
                maxs,
                M,
                T,
                n,
                W,
                round_scale,
                calibrate,
                num_stages=2,
                num_warps=16
            )
        else:
            B = 512
            T = 16
            assert n % B == 0 and M % T == 0
            grid = (M // T,)
            if maxs is None and calibrate:
                maxs = torch.empty((M // T, n), device=device, dtype=torch.float32)
            compatible_silu_and_smooth_quant_forward_kernel[grid](
                x,
                smooth_scale,
                out,
                scale,
                maxs,
                M,
                T,
                N // 2,
                B,
                round_scale,
                calibrate,
                num_stages=2,
                num_warps=16
            )

        if calibrate:
            maxs = maxs.amax(0)

    else:
        grid = (triton.cdiv(M, 128), n // 128)
        silu_and_block_quant_forward_kernel[grid](
            x,
            out,
            scale,
            transpose_output, 
            transpose_scale,
            M,
            n,
            round_scale,
            output_mode,
            num_stages=2,
            num_warps=16
        )

    return out, scale, maxs, transpose_output, transpose_scale



@triton.jit
def silu_and_smooth_quant_backward_kernel(g_ptr, x_ptr, 
                                              smooth_scale_ptr,
                                              transpose_smooth_scale_ptr,
                                              dx_ptr, dx_scale_ptr, 
                                              transpose_dx_ptr,
                                              transpose_dx_scale_ptr,
                                              M,
                                              n: tl.constexpr,
                                              T: tl.constexpr, 
                                              B: tl.constexpr,
                                              REVERSE: tl.constexpr,
                                              ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)

    offs = pid * T * n * 2 + tl.arange(0, T)[:, None] * n * 2 + tl.arange(0, B)[
                                                                None, :]
    hoffs = pid * T * n + tl.arange(0, T)[:, None] * n + tl.arange(0, B)[None,
                                                         :]
    toffs = pid * T + tl.arange(0, B)[:, None] * M + tl.arange(0, T)[None, :]
    nb = n // B
    maxs = tl.zeros((T, ), dtype=tl.float32)
    transpose_smooth_scale = tl.load(transpose_smooth_scale_ptr + pid * T + tl.arange(0, T))[:, None]
    for i in range(nb):
        smooth_scale_1 = tl.load(smooth_scale_ptr + i * B + tl.arange(0, B))
        smooth_scale_2 = tl.load(smooth_scale_ptr + n + i * B + tl.arange(0, B))
        if not REVERSE:
            smooth_scale_1 = 1 / smooth_scale_1
            smooth_scale_2 = 1 / smooth_scale_2

        x1 = tl.load(x_ptr + offs).to(tl.float32)
        x2 = tl.load(x_ptr + offs + n).to(tl.float32)
        g = tl.load(g_ptr + hoffs).to(tl.float32)
        sigmoid = 1 / (1 + tl.exp(-x1))

        # x1 = tl.load(x_ptr + offs)
        # x2 = tl.load(x_ptr + offs + n)
        # g = tl.load(g_ptr + hoffs)
        # sigmoid = 1 / (1 + tl.exp(-x1.to(tl.float32)))

        dx1 = g * x2 * sigmoid * (
                1 + x1 * (1 - sigmoid))
        dx2 = g * x1 * sigmoid

        t_dx = dx1 * transpose_smooth_scale
        t_s = tl.maximum(tl.max(tl.abs(t_dx), 0) / 448, 1e-30)
        if ROUND:
            t_s = tl.exp2(tl.ceil(tl.log2(t_s)))
        t_dx = t_dx/t_s
        tl.store(transpose_dx_ptr + toffs, tl.trans(t_dx.to(transpose_dx_ptr.dtype.element_ty)))
        tl.store(transpose_dx_scale_ptr + pid * n * 2 + i * B + tl.arange(0, B), t_s)

        t_dx = dx2 * transpose_smooth_scale
        t_s = tl.maximum(tl.max(tl.abs(t_dx), 0) / 448, 1e-30)
        if ROUND:
            t_s = tl.exp2(tl.ceil(tl.log2(t_s)))
        t_dx = t_dx/t_s
        tl.store(transpose_dx_ptr + M * n + toffs, tl.trans(t_dx.to(transpose_dx_ptr.dtype.element_ty)))
        tl.store(transpose_dx_scale_ptr + pid * n * 2 + n + i * B + tl.arange(0, B), t_s)

        dx1 = dx1 * smooth_scale_1
        dx2 = dx2 * smooth_scale_2

        # maxs = tl.maximum(
        #     tl.maximum(dx1.abs(), dx2.abs()), maxs)
        maxs = tl.maximum(
            tl.maximum(tl.max(dx1.abs(), 1), tl.max(dx2.abs(), 1)), maxs)

        offs += B
        hoffs += B
        toffs += B * M

    scale = tl.maximum(maxs / 448, 1e-30)
    # scale = tl.maximum(tl.max(maxs, 1) / 448, 1e-30)

    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(dx_scale_ptr + pid * T + tl.arange(0, T), scale)

    s = 1 / scale[:, None]
    offs = pid * T * n * 2 + tl.arange(0, T)[:, None] * n * 2 + tl.arange(0, B)[
                                                                None, :]
    hoffs = pid * T * n + tl.arange(0, T)[:, None] * n + tl.arange(0, B)[None,
                                                         :]
    for i in range(nb):
        smooth_scale_1 = tl.load(smooth_scale_ptr + i * B + tl.arange(0, B))
        smooth_scale_2 = tl.load(smooth_scale_ptr + n + i * B + tl.arange(0, B))
        if not REVERSE:
            smooth_scale_1 = 1 / smooth_scale_1
            smooth_scale_2 = 1 / smooth_scale_2

        x1 = tl.load(x_ptr + offs).to(tl.float32)
        x2 = tl.load(x_ptr + offs + n).to(tl.float32)
        g = tl.load(g_ptr + hoffs).to(tl.float32)
        sigmoid = 1 / (1 + tl.exp(-x1))
        dx1 = g * x2 * sigmoid * (
                1 + x1 * (1 - sigmoid)) * smooth_scale_1
        dx2 = g * x1 * sigmoid * smooth_scale_2

        dx1 = (dx1 * s).to(dx_ptr.dtype.element_ty)
        dx2 = (dx2 * s).to(dx_ptr.dtype.element_ty)

        tl.store(dx_ptr + offs, dx1)
        tl.store(dx_ptr + n + offs, dx2)
        offs += B
        hoffs += B

# requant multi-column quantized tensor
@triton.jit 
def _requant_kernel(x_ptr, scale_ptr, scales_ptr,
                    M, 
                    N,
                    H: tl.constexpr,
                    W: tl.constexpr
                    ):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    offs = rid * H * N + cid * W + tl.arange(0, H)[:, None] * N + tl.arange(0, W)[None, :]
    global_scale = tl.load(scale_ptr + rid * H + tl.arange(0, H))
    # scales is stored with column-major format
    local_scale = tl.load(scales_ptr + cid * M + rid * H + tl.arange(0, H))
    x = tl.load(x_ptr+offs).to(tl.float32)
    rescale = local_scale/global_scale
    x = x * rescale[:,None]
    tl.store(x_ptr+offs, x)


@triton.jit
def silu_and_block_quant_backward_kernel(g_ptr, x_ptr,  
                                        dx_ptr,
                                        dx_scale_ptr, 
                                        transpose_dx_ptr,
                                        transpose_dx_scale_ptr,
                                        M, 
                                        n: tl.constexpr,
                                        ROUND: tl.constexpr):
    rid = tl.program_id(axis=0)
    cid = tl.program_id(axis=1)
    nb = n // 128
    offs = rid * 128 * n * 2 + cid * 128 + tl.arange(0, 128)[:, None] * n * 2 + tl.arange(0, 128)[None, :]
    toffs = rid * 128 + cid * M * 128 + tl.arange(0, 128)[:, None] * M + tl.arange(0, 128)[
                                                            None, :]
    idx = rid * 128 + tl.arange(0, 128)
    mask = idx[:, None] < M
    x1 = tl.load(x_ptr + offs, mask=mask)#.to(tl.float32)
    x2 = tl.load(x_ptr + n + offs, mask=mask)#.to(tl.float32)
    g = tl.load(g_ptr + rid * 128 * n + cid * 128 + 
                    tl.arange(0, 128)[:, None] * n + 
                    tl.arange(0, 128)[None, :], mask=mask)#.to(tl.float32)
    sigmoid = tl.sigmoid(x1.to(tl.float32))
    dx1 = sigmoid * g * x2 * (1 + x1 * (1 - sigmoid))  # change order to trigger autocast
    scale1 = tl.maximum(
        tl.max(dx1.abs(), 1) / 448, 1e-30)
    if ROUND:
        scale1 = tl.exp2(tl.ceil(tl.log2(scale1)))
    tl.store(dx_scale_ptr + cid * M + rid * 128 + tl.arange(0, 128), scale1, mask=idx < M)

    qdx1 = (dx1 / scale1[:, None]).to(dx_ptr.dtype.element_ty)
    tl.store(dx_ptr + offs, qdx1, mask=mask)

    scale1 = tl.maximum(
        tl.max(dx1.abs(), 0) / 448, 1e-30)
    if ROUND:
        scale1 = tl.exp2(tl.ceil(tl.log2(scale1)))
    tl.store(transpose_dx_scale_ptr + rid * n * 2 + cid * 128 + tl.arange(0, 128), scale1)

    qdx1 = (dx1 / scale1[None, :]).to(dx_ptr.dtype.element_ty)
    tl.store(transpose_dx_ptr + toffs, tl.trans(qdx1), mask=idx[None, :] < M)

    dx2 = sigmoid * g * x1
    scale2 = tl.maximum(
        tl.max(dx2.abs(), 1) / 448, 1e-30)
    if ROUND:
        scale2 = tl.exp2(tl.ceil(tl.log2(scale2)))
    tl.store(dx_scale_ptr + cid * M + rid * 128 + M * nb + tl.arange(0, 128), scale2, mask=idx < M)

    qdx2 = (dx2 / scale2[:, None]).to(dx_ptr.dtype.element_ty)
    tl.store(dx_ptr + offs + n, qdx2, mask=idx[:, None] < M)

    scale2 = tl.maximum(
        tl.max(dx2.abs(), 0) / 448, 1e-30)
    if ROUND:
        scale2 = tl.exp2(tl.ceil(tl.log2(scale2)))
    tl.store(transpose_dx_scale_ptr + rid * n * 2 + n + cid * 128 + tl.arange(0, 128), scale2)

    qdx2 = (dx2 / scale2[None, :]).to(dx_ptr.dtype.element_ty)
    tl.store(transpose_dx_ptr + M * n + toffs, tl.trans(qdx2), mask=idx[None,:] < M)



# used in shared expert
def triton_silu_and_quant_backward(g, x, 
                                   smooth_scale=None, 
                                   transpose_smooth_scale=None, 
                                   reverse=True,
                                   round_scale=False):
    # row-wise read, row-wise write
    M, N = x.shape
    n = N // 2
    device = x.device
    smooth = smooth_scale is not None
    dx = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    if smooth:
        dx_scale = torch.empty((M,), device=device, dtype=torch.float32)
        scale_shape = (N, )
    else:
        dx_scale = torch.empty((N//128, M), device=device, dtype=torch.float32)
        scale_shape = (triton.cdiv(M, 128), N)
    transpose_dx = torch.empty((N, M), device=device, dtype=torch.float8_e4m3fn)
    transpose_dx_scale = torch.empty(scale_shape, device=device, dtype=torch.float32)

    if smooth:
        T = 32
        B = 32
        assert M % T == 0 and n % B == 0
        transpose_dx_scales = torch.empty((M // T, N), device=device, dtype=torch.float32)
        grid = (M // T,)
        silu_and_smooth_quant_backward_kernel[grid](
            g,
            x,
            smooth_scale,
            transpose_smooth_scale,
            dx,
            dx_scale,
            transpose_dx,
            transpose_dx_scales,
            M,
            n,
            T,
            B,
            reverse,
            round_scale,
            num_stages=3,
            num_warps=2
        )
        transpose_dx_scale = transpose_dx_scales.amax(0)
        grid = (N // B, M // T)
        _requant_kernel[grid](transpose_dx, transpose_dx_scale, transpose_dx_scales,
                N, 
                M,
                B,
                T)
    else:
        assert M % 128 == 0
        grid = (M//128, N // 256)
        silu_and_block_quant_backward_kernel[grid](
            g,
            x,
            dx,
            dx_scale,
            transpose_dx,
            transpose_dx_scale,
            M,
            n,
            round_scale,
            num_stages=2,
            num_warps=8
        )
    return dx, dx_scale, transpose_dx, transpose_dx_scale


@triton.jit
def batch_weighted_silu_and_smooth_quant_forward_kernel(x_ptr, weight_ptr,
                                                 smooth_scale_ptr, out_ptr,
                                                 scale_ptr, max_ptr, count_ptr,
                                                 accum_ptr, M,
                                                 n: tl.constexpr,
                                                 W: tl.constexpr,
                                                 ROUND: tl.constexpr,
                                                 REVERSE: tl.constexpr,
                                                 CALIBRATE: tl.constexpr):
    eid = tl.program_id(axis=0)
    tid = tl.program_id(axis=1)
    sm = tl.num_programs(axis=1)

    count = tl.load(count_ptr + eid)
    ei = tl.load(accum_ptr + eid)
    si = ei - count
    c = tl.cdiv(count, sm * W)

    row_offs = si * n + tid * c * W * n + tl.arange(0, W)[:, None] * n
    col_offs = tl.arange(0, n)[None, :]
    smooth_scale = tl.load(smooth_scale_ptr + n * eid + tl.arange(0, n))
    if not REVERSE:
        smooth_scale = 1.0 / smooth_scale

    if CALIBRATE:
        maxs = tl.zeros((W, n), dtype=tl.float32)

    for i in range(c):
        indices = tid * c * W + i * W + tl.arange(0, W)
        mask = indices[:, None] < count
        x1 = tl.load(x_ptr + row_offs * 2 + col_offs, mask=mask).to(tl.float32)
        x2 = tl.load(x_ptr + n + row_offs * 2 + col_offs, mask=mask).to(
            tl.float32)

        w = tl.load(weight_ptr + si + indices, mask=indices < count).to(
            tl.float32)[:,
            None]
        x = x1 / (1 + tl.exp(-x1)) * x2

        if CALIBRATE:
            maxs = tl.maximum(x.abs(), maxs)

        x *= w * smooth_scale
        scale = tl.maximum(tl.max(x.abs(), 1) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(scale_ptr + si + indices, scale, mask=indices < count)
        x = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + row_offs + col_offs, x, mask=mask)
        row_offs += n * W

    if CALIBRATE:
        maxs = tl.max(maxs, 0)
        tl.store(max_ptr + eid * sm * n + tid * n + tl.arange(0, n), maxs)



@triton.jit
def batch_weighted_silu_and_block_quant_forward_kernel(x_ptr, weight_ptr,
                                                 out_ptr,
                                                 scale_ptr, 
                                                 transpose_output_ptr, 
                                                 transpose_scale_ptr,
                                                 count_ptr,
                                                 accum_ptr, 
                                                 n: tl.constexpr,
                                                 E: tl.constexpr,
                                                 ROUND: tl.constexpr,
                                                 OUTPUT_MODE: tl.constexpr):
    eid = tl.program_id(axis=0)
    rid = tl.program_id(axis=1)
    cid = tl.program_id(axis=2)

    count = tl.load(count_ptr + eid)
    ei = tl.load(accum_ptr + eid)
    si = ei - count
    c = tl.cdiv(count, 128)

    if rid >= c:
        return

    nb = n // 128

    counts = tl.load(count_ptr + tl.arange(0, E))
    n_blocks = tl.cdiv(counts, 128)
    transpose_scale_off = tl.sum(tl.where(tl.arange(0, E)< eid, n_blocks, 0))

    offs = si * n * 2 + rid * 128 * n * 2 + cid * 128 + tl.arange(0, 128)[:, None] * n * 2 + tl.arange(0, 128)[None, :]
    hoffs = si * n + rid * 128 * n + cid * 128 + tl.arange(0, 128)[:, None] * n + tl.arange(0, 128)[None, :]
    toffs = si * n + rid * 128 + cid * count * 128 + tl.arange(0, 128)[:, None] * count + tl.arange(0, 128)[
                                                            None, :]
    indices = rid * 128 + tl.arange(0, 128)
    mask = indices[:, None] < count
    w = tl.load(weight_ptr + si + indices, mask=indices < count).to(
            tl.float32)
    x1 = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    x2 = tl.load(x_ptr + n + offs, mask=mask).to(
        tl.float32)

    x = x1 * tl.sigmoid(x1) * x2 * w[:, None]

    if OUTPUT_MODE % 2 == 0:
        scale = tl.maximum(tl.max(tl.abs(x), 1) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(scale_ptr + si * nb + cid * count + rid * 128 + tl.arange(0, 128), scale, mask=indices < count)

        xq = (x / scale[:, None]).to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + hoffs, xq, mask=mask)

    if OUTPUT_MODE > 0:
        scale = tl.maximum(tl.max(tl.abs(x), 0) / 448, 1e-30)
        if ROUND:
            scale = tl.exp2(tl.ceil(tl.log2(scale)))
        tl.store(transpose_scale_ptr + transpose_scale_off * n + rid * n + cid * 128 + tl.arange(0, 128), scale)

        xq = tl.trans((x / scale).to(out_ptr.dtype.element_ty))
        tl.store(transpose_output_ptr + toffs, xq, mask=indices[None, :] < count)


# used in routed experts
def triton_batch_weighted_silu_and_quant_forward(x, 
                                                 weight, 
                                                 counts, 
                                                 smooth_scale=None,
                                                 splits=None,
                                                 out=None, 
                                                 scale=None,
                                                 round_scale=False,
                                                 reverse=False,
                                                 calibrate=False,
                                                 output_mode=2):
    # row-wise read, row-wise write
    M, N = x.shape
    n = N // 2
    n_experts = counts.shape[0]
    smooth = smooth_scale is not None
    assert N <= 8192
    device = x.device
    if out is None:
        out = torch.empty((M, n), device=device, dtype=torch.float8_e4m3fn)

    if smooth:
        sm = torch.cuda.get_device_properties(device).multi_processor_count
        transpose_output = None
        transpose_scale = None
        tmp_maxs = None
        if scale is None:
            scale = torch.empty((M,), device=device, dtype=torch.float32)
        if M == 0:
            maxs = torch.zeros((n_experts, n), device=device,
                                    dtype=torch.float32)

        elif calibrate:
            tmp_maxs = torch.empty((n_experts, sm, n), device=device,
                                dtype=torch.float32)
            maxs = torch.empty((n_experts, n), device=device,
                            dtype=torch.float32)
        else:
            maxs = None
            
    else:
        assert splits is not None, 'batch mode need splits to launch kernels'
        maxs = None
        blocks = sum([(x+127)//128 for x in splits])
        transpose_output = torch.empty((M * n), device=device, dtype=torch.float8_e4m3fn)
        transpose_scale = torch.empty((blocks * n), device=device, dtype=torch.float32)
        # intra layout and inner layput are not consist,
        # tensors will be viewed after splitting
        scale = torch.empty((M * n // 128,), device=device, dtype=torch.float32)


    if M == 0:
        return out, scale, maxs, transpose_output, transpose_scale


    accums = torch.cumsum(counts, 0)
    if smooth:
        W = 8192 // N
        grid = (n_experts, sm)
        batch_weighted_silu_and_smooth_quant_forward_kernel[grid](
            x,
            weight,
            smooth_scale,
            out,
            scale,
            tmp_maxs,
            counts,
            accums,
            M,
            n,
            W,
            round_scale,
            reverse,
            calibrate,
            num_stages=3,
            num_warps=16
        )
        if calibrate:
            maxs = tmp_maxs.amax(1)
    else:
        # grid = (n_experts, triton.cdiv(max(splits), 128))
        grid = (n_experts, triton.cdiv(max(splits), 128), n//128)
        batch_weighted_silu_and_block_quant_forward_kernel[grid](
            x,
            weight,
            out,
            scale,
            transpose_output,
            transpose_scale,
            counts,
            accums,
            n,
            len(splits),
            round_scale,
            output_mode,
            num_stages=2,
            num_warps=8
        )


    return out, scale, maxs, transpose_output, transpose_scale


@triton.jit
def batch_weighted_silu_and_smooth_quant_backward_kernel(g_ptr, x_ptr, weight_ptr,
                                                  smooth_scale_ptr, transpose_smooth_scale_ptr,
                                                  count_ptr,
                                                  accum_ptr, 
                                                  dx_ptr,
                                                  dx_scale_ptr, 
                                                  transpose_dx_ptr, 
                                                  transpose_dx_scale_ptr,
                                                  dw_ptr, 
                                                  n: tl.constexpr,
                                                  T: tl.constexpr,
                                                  B: tl.constexpr,
                                                  E: tl.constexpr,
                                                  REVERSE: tl.constexpr,
                                                  ROUND: tl.constexpr):
    eid = tl.program_id(axis=0)
    pid = tl.program_id(axis=1)
    max_block = tl.num_programs(axis=1)

    count = tl.load(count_ptr + eid)
    round_count = tl.cdiv(count, 32) * 32
    si = tl.load(accum_ptr + eid) - count

    if pid >= tl.cdiv(count, T):
        return 

    round_off = tl.sum(tl.where(tl.arange(0, E) < eid, tl.cdiv(tl.load(count_ptr + tl.arange(0, E)), 32), 0)) * 32

    offs = si * n * 2 + pid * T * n * 2 + tl.arange(0, T)[:, None] * n * 2 + tl.arange(0, B)[
                                                                None, :]
    hoffs = si * n + pid * T * n + tl.arange(0, T)[:, None] * n + tl.arange(0, B)[None,
                                                         :]
    toffs = round_off * n * 2 + pid * T + tl.arange(0, B)[:, None] * round_count + tl.arange(0, T)[None, :]
    nb = n // B
    maxs = tl.zeros((T,), dtype=tl.float32)
    indices = pid * T + tl.arange(0, T)
    if REVERSE:
        transpose_smooth_scale = tl.load(transpose_smooth_scale_ptr + si + pid * T + tl.arange(0, T), mask=indices<count)[:, None]
    else:
        transpose_smooth_scale = 1/tl.load(transpose_smooth_scale_ptr + si + pid * T + tl.arange(0, T), mask=indices<count, other=1e-30)[:, None]

    w = tl.load(weight_ptr + si + pid * T + tl.arange(0, T), mask=indices < count)[:, None]
    dw = tl.zeros((T,), dtype=tl.float32)
    qdtype = transpose_dx_ptr.dtype.element_ty
    for i in range(nb):
        smooth_scale_1 = tl.load(smooth_scale_ptr + eid * n * 2 + i * B + tl.arange(0, B))
        smooth_scale_2 = tl.load(smooth_scale_ptr + eid * n * 2 + n + i * B + tl.arange(0, B))
        if not REVERSE:
            smooth_scale_1 = 1 / smooth_scale_1
            smooth_scale_2 = 1 / smooth_scale_2

        x1 = tl.load(x_ptr + offs, mask=indices[:, None] < count).to(tl.float32)
        x2 = tl.load(x_ptr + offs + n, mask=indices[:, None] < count).to(tl.float32)
        g = tl.load(g_ptr + hoffs, mask=indices[:, None] < count).to(tl.float32)
        sigmoid = 1 / (1 + tl.exp(-x1))
        dx1 = g * x2 * sigmoid * (
                1 + x1 * (1 - sigmoid)) * w
        dx2 = g * x1 * sigmoid * w

        dw += tl.sum(x1 * sigmoid * x2 * g, 1)

        t_dx = dx1 * transpose_smooth_scale
        t_s = tl.maximum(tl.max(tl.abs(t_dx), 0) / 448, 1e-30)
        if ROUND:
            t_s = tl.exp2(tl.ceil(tl.log2(t_s)))
        t_dx = t_dx/t_s
        tl.store(transpose_dx_ptr + toffs, tl.trans(t_dx.to(qdtype)), mask=indices[None, :] < round_count)
        tl.store(transpose_dx_scale_ptr + eid * max_block * n * 2 + pid * n * 2 + i * B + tl.arange(0, B), t_s)

        t_dx = dx2 * transpose_smooth_scale
        t_s = tl.maximum(tl.max(tl.abs(t_dx), 0) / 448, 1e-30)
        if ROUND:
            t_s = tl.exp2(tl.ceil(tl.log2(t_s)))
        t_dx = t_dx/t_s
        tl.store(transpose_dx_ptr + round_count * n + toffs, tl.trans(t_dx.to(qdtype)), mask=indices[None, :] < round_count)
        tl.store(transpose_dx_scale_ptr + eid * max_block * n * 2 + pid * n * 2 + n + i * B + tl.arange(0, B), t_s)

        dx1 = dx1 * smooth_scale_1
        dx2 = dx2 * smooth_scale_2
        maxs = tl.maximum(
            tl.maximum(tl.max(dx1.abs(), 1), tl.max(dx2.abs(), 1)), maxs)

        offs += B
        hoffs += B
        toffs += B * round_count


    tl.store(dw_ptr + si + pid * T + tl.arange(0, T), dw, mask=indices < count)
    scale = tl.maximum(maxs / 448, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(dx_scale_ptr +  si + pid * T + tl.arange(0, T), scale, mask=indices < count)

    s = 1 / scale[:, None]
    offs = si * n * 2 + pid * T * n * 2 + tl.arange(0, T)[:, None] * n * 2 + tl.arange(0, B)[
                                                                None, :]
    hoffs = si * n + pid * T * n + tl.arange(0, T)[:, None] * n + tl.arange(0, B)[None,
                                                         :]
    for i in range(nb):
        smooth_scale_1 = tl.load(smooth_scale_ptr + eid * n * 2 + i * B + tl.arange(0, B))
        smooth_scale_2 = tl.load(smooth_scale_ptr + eid * n * 2 + n + i * B + tl.arange(0, B))
        if not REVERSE:
            smooth_scale_1 = 1 / smooth_scale_1
            smooth_scale_2 = 1 / smooth_scale_2

        x1 = tl.load(x_ptr + offs, mask=indices[:, None] < count).to(tl.float32)
        x2 = tl.load(x_ptr + offs + n, mask=indices[:, None] < count).to(tl.float32)
        g = tl.load(g_ptr + hoffs, mask=indices[:, None] < count).to(tl.float32)
        sigmoid = 1 / (1 + tl.exp(-x1))
        dx1 = g * x2 * sigmoid * (
                1 + x1 * (1 - sigmoid)) * smooth_scale_1 * w
        dx2 = g * x1 * sigmoid * smooth_scale_2 * w

        dx1 = (dx1 * s).to(dx_ptr.dtype.element_ty)
        dx2 = (dx2 * s).to(dx_ptr.dtype.element_ty)

        tl.store(dx_ptr + offs, dx1, mask=indices[:, None] < count)
        tl.store(dx_ptr + n + offs, dx2, mask=indices[:, None] < count)
        offs += B
        hoffs += B


# requant multi-column quantized tensor
@triton.jit 
def _batch_requant_kernel(x_ptr, scale_ptr, scales_ptr,
                    count_ptr,
                    N,
                    H: tl.constexpr,
                    W: tl.constexpr,
                    E: tl.constexpr
                    ):
    eid = tl.program_id(axis=0)
    rid = tl.program_id(axis=1)
    cid = tl.program_id(axis=2)
    max_block = tl.num_programs(axis=2)

    count = tl.load(count_ptr + eid)
    round_count = tl.cdiv(count, 32) * 32
    if cid >= tl.cdiv(round_count, W):
        return 

    round_off = tl.sum(tl.where(tl.arange(0, E) < eid, tl.cdiv(tl.load(count_ptr + tl.arange(0, E)), 32) * 32, 0))

    offs = round_off * N + rid * H * round_count + cid * W + tl.arange(0, H)[:, None] * round_count + tl.arange(0, W)[None, :]
    global_scale = tl.load(scale_ptr + eid * N + rid * H + tl.arange(0, H))
    # scales is stored with column-major format
    local_scale = tl.load(scales_ptr + max_block * N * eid + cid * N + rid * H + tl.arange(0, H))
    x = tl.load(x_ptr+offs).to(tl.float32)
    rescale = local_scale/tl.maximum(global_scale, 1e-30)
    x = x * rescale[:, None]
    tl.store(x_ptr+offs, x)



@triton.jit
def batch_weighted_silu_and_block_quant_backward_kernel(g_ptr, x_ptr, weight_ptr,
                                                  count_ptr,
                                                  accum_ptr, 
                                                  dx_ptr,
                                                  dx_scale_ptr, 
                                                  transpose_dx_ptr, 
                                                  transpose_dx_scale_ptr,
                                                  dw_ptr, 
                                                  n: tl.constexpr,
                                                  E: tl.constexpr,
                                                  ROUND: tl.constexpr):
    eid = tl.program_id(axis=0)
    rid = tl.program_id(axis=1)
    cid = tl.program_id(axis=2)

    count = tl.load(count_ptr + eid)
    si = tl.load(accum_ptr + eid) - count

    if rid >= tl.cdiv(count, 128):
        return 
    
    nb = n // 128
    transpose_off = tl.sum(tl.where(tl.arange(0, E) < eid, tl.cdiv(tl.load(count_ptr + tl.arange(0, E)), 128), 0))

    offs = si * n * 2 + rid * 128 * n * 2 + cid * 128 + tl.arange(0, 128)[:, None] * n * 2 + tl.arange(0, 128)[None, :]
    # hoffs = si * n + tid * 128 * n + tl.arange(0, 128)[:, None] * n + tl.arange(0, 128)[None, :]
    # toffs = si * n * 2 + tid * 128 + tl.arange(0, 128)[:, None] * count + tl.arange(0, 128)[None, :]
    idx = rid * 128 + tl.arange(0, 128)
    w = tl.load(weight_ptr + si + idx, mask=idx < count).to(tl.float32)[:, None]

    x1 = tl.load(x_ptr + offs, mask=idx[:, None] < count) #.to(tl.float32)
    x2 = tl.load(x_ptr + n + offs, mask=idx[:, None] < count) #.to(tl.float32)
    g = tl.load(g_ptr +  si * n + rid * 128 * n + 128 * cid + 
                tl.arange(0, 128)[:, None] * n + 
                tl.arange(0, 128)[None, :], 
                mask=idx[:, None] < count) #.to(tl.float32)
    sigmoid = tl.sigmoid(x1.to(tl.float32))

    dw = tl.sum(sigmoid * x1 * x2 * g, 1)
    tl.store(dw_ptr + si * nb + cid + idx * nb, dw, mask=idx < count)

    dx = sigmoid * g * x2 * w * (1 + x1 * (1 - sigmoid))
    scale = tl.maximum(
        tl.max(dx.abs(), 1) / 448, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(dx_scale_ptr + si * nb * 2 + cid * count + rid * 128 + tl.arange(0, 128), scale, mask=idx < count)

    tl.store(dx_ptr + offs, dx / scale[:, None], mask=idx[:, None] < count)

    scale = tl.maximum(
        tl.max(dx.abs(), 0) / 448, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(transpose_dx_scale_ptr + transpose_off * n * 2 + rid * n * 2 + cid * 128 + tl.arange(0, 128), scale)

    qdx = tl.trans((dx / scale[None, :]).to(dx_ptr.dtype.element_ty))
    # tl.store(transpose_dx_ptr + toffs, qdx, mask=idx[None, :] < count)
    tl.store(transpose_dx_ptr + si * n * 2 + rid * 128 + cid * 128 * count + 
             tl.arange(0, 128)[:, None] * count + 
             tl.arange(0, 128)[None, :], 
             qdx, 
             mask=idx[None, :] < count)

    dx = sigmoid * g * x1 * w
    scale = tl.maximum(
        tl.max(dx.abs(), 1) / 448, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    tl.store(dx_scale_ptr + si * nb * 2 + cid * count + rid * 128 + count * nb + tl.arange(0, 128), scale, mask=idx < count)
    tl.store(dx_ptr + n + offs, dx / scale[:, None], mask=idx[:, None] < count)

    scale = tl.maximum(
        tl.max(dx.abs(), 0) / 448, 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(scale)))
    qdx = tl.trans((dx / scale[None, :]).to(dx_ptr.dtype.element_ty))
    tl.store(transpose_dx_scale_ptr + transpose_off * n * 2 + rid * n * 2 + n + cid * 128 + tl.arange(0, 128), scale)
    tl.store(transpose_dx_ptr + count * n + si * n * 2 + rid * 128 + cid * 128 * count + tl.arange(0, 128)[:, None] * count + tl.arange(0, 128)[None, :], qdx, mask=idx[None, :] < count)



# used in routed experts
def triton_batch_weighted_silu_and_quant_backward(g, x, weight, 
                                                  counts,
                                                  smooth_scale=None,
                                                  transpose_smooth_scale=None,
                                                  splits=None,
                                                  reverse=True,
                                                  round_scale=False):
    # row-wise read, row-wise write
    M, N = x.shape
    n = N // 2
    n_expert = counts.shape[0]
    smooth = smooth_scale is not None
    assert N <= 8192 and 8192 % N == 0
    assert splits is not None, 'batch mode need splits to launch kernels'

    device = x.device

    accums = torch.cumsum(counts, 0)

    dx = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn) 

    if smooth:
        dx_scale = torch.empty((M,), device=device, dtype=torch.float32)

        dw = torch.empty_like(weight)
        T = 32 
        B = 32
        assert n % B == 0 and T == 32
        max_block = triton.cdiv(max(splits), T)
        s = sum([(x+31)//32 for x in splits])*32
        transpose_dx = torch.empty((N * s,), device=device, dtype=torch.float8_e4m3fn)   

        if s == 0:
            transpose_dx_scale = torch.zeros((n_expert, N), device=device, dtype=torch.float32)   
            return dx, dx_scale, dw, transpose_dx, transpose_dx_scale
        else:
            transpose_dx_scales = torch.zeros((n_expert, max_block, N), device=device, dtype=torch.bfloat16)    

        grid = (n_expert, max_block)
        batch_weighted_silu_and_smooth_quant_backward_kernel[grid](
            g,
            x,
            weight,
            smooth_scale,
            transpose_smooth_scale,
            counts,
            accums,
            dx,
            dx_scale,
            transpose_dx,
            transpose_dx_scales,
            dw,
            n,
            T,
            B,
            n_expert,
            reverse,
            round_scale,
            num_stages=5,
            num_warps=4
        )
        transpose_dx_scale = transpose_dx_scales.amax(1).float()
        grid = (n_expert, N // B, max_block)
        _batch_requant_kernel[grid](transpose_dx, transpose_dx_scale, transpose_dx_scales,
                counts,
                N, 
                B,
                T,
                n_expert,
                num_stages=3,
                num_warps=2)
    else:
        # intra layout and inner layput are not consist,
        # tensors will be viewed after splitting
        dx_scale = torch.empty((N // 128 * M), device=device, dtype=torch.float32)

        s = sum([(x+127)//128 for x in splits])
        transpose_dx = torch.empty((N * M), device=device, dtype=torch.float8_e4m3fn)  
        transpose_dx_scale = torch.empty((s * N), device=device, dtype=torch.float32) 
        if s == 0:
            dw = torch.empty_like(weight)
            return dx, dx_scale, dw, transpose_dx, transpose_dx_scale

        # grid = (n_expert, triton.cdiv(max(splits), 128))
        grid = (n_expert, triton.cdiv(max(splits), 128), N//256)
        dws = torch.empty((M, N//256), device=device, dtype=torch.float32)
        batch_weighted_silu_and_block_quant_backward_kernel[grid](
            g,
            x,
            weight,
            counts,
            accums,
            dx,
            dx_scale,
            transpose_dx,
            transpose_dx_scale,
            dws,
            n,
            n_expert,
            round_scale,
            num_stages=3,
            num_warps=16
        )
        dw = dws.sum(1, keepdim=True).to(weight.dtype)
    return dx, dx_scale, dw, transpose_dx, transpose_dx_scale
