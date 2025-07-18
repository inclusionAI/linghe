import math
import torch
import triton
import triton.language as tl
from triton import Config



@triton.jit
def row_quant_kernel(x_ptr, q_ptr, s_ptr,  M, N,  BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    n_block = tl.cdiv(N, BLOCK_SIZE)
    indices = tl.arange(0, BLOCK_SIZE)
    max_val = 1e-30
    N = N.to(tl.int64)

    for j in range(n_block):
        offs = pid*N + j*BLOCK_SIZE + indices
        x = tl.load(x_ptr + offs, mask=j*BLOCK_SIZE + indices<N, other=0)
        max_val = tl.maximum(tl.max(tl.abs(x.to(tl.float32))), max_val)
    scale = max_val/448.0
    tl.store(s_ptr + pid, scale)
    s = 448.0/max_val
    for j in range(n_block):
        offs = pid*N + j*BLOCK_SIZE + indices
        x = tl.load(x_ptr + offs, mask=j*BLOCK_SIZE + indices<N, other=0)
        y = x.to(tl.float32) * s
        y = y.to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + offs, y, mask=j*BLOCK_SIZE + indices<N)


def triton_row_quant(x):
    M, N = x.shape 
    BLOCK_SIZE = 8192
    x_q = torch.empty((M,N),dtype=torch.float8_e4m3fn,device=x.device)
    x_scale = torch.empty((M,),dtype=torch.float32,device=x.device)
    grid = (M, )
    row_quant_kernel[grid](
        x, x_q, x_scale,
        M, N,
        BLOCK_SIZE,
        num_stages=5,
        num_warps=4
    )
    return x_q, x_scale



@triton.jit
def deprecated_tokenwise_row_quant_kernel(x_ptr, out_ptr, scale_ptr, M, T: tl.constexpr, N: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs = pid*T*N+tl.arange(0, N)
    for i in range(T):
        mask = pid*T+i<M
        x = tl.load(x_ptr+offs, mask=mask).to(tl.float32)

        scale = tl.maximum(tl.max(tl.abs(x)), 1e-30)/448.0

        x = (x/scale).to(out_ptr.dtype.element_ty)
        tl.store(scale_ptr+i,scale, mask=mask)
        tl.store(out_ptr+offs, x, mask=mask)

        offs += N



def triton_deprecated_tokenwise_row_quant(x, out=None, scale=None):
    # row-wise read, row-wise write
    M, N = x.shape
    device = x.device 
    if out is None:
        out = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.empty((M,), dtype=torch.float32, device=device)
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    T = triton.cdiv(M, sm)
    grid = (sm, )
    deprecated_tokenwise_row_quant_kernel[grid](
        x,
        out,
        scale,
        M, T, N, 
        num_stages=3,
        num_warps=16
    )
    return out, scale



@triton.jit
def tokenwise_row_quant_kernel(x_ptr, out_ptr, scale_ptr, N: tl.constexpr, ROUND: tl.constexpr):
    pid = tl.program_id(axis=0)
    x = tl.load(x_ptr+pid*N+tl.arange(0, N)).to(tl.float32)
    x_max = tl.maximum(tl.max(tl.abs(x)), 1e-30)
    if ROUND:
        scale = tl.exp2(tl.ceil(tl.log2(x_max/448.0)))
    else:
        scale = x_max/448.0
    tl.store(scale_ptr+pid, scale)
    x = (x/scale).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr+pid*N+tl.arange(0, N), x)


def triton_tokenwise_row_quant(x, out=None, scale=None, round_scale=False):
    # row-wise read, row-wise write
    M, N = x.shape
    device = x.device 
    if out is None:
        out = torch.empty((M, N), device=device, dtype=torch.float8_e4m3fn)
    if scale is None:
        scale = torch.empty((M,), dtype=torch.float32, device=device)
    grid = (M, )
    tokenwise_row_quant_kernel[grid](
        x,
        out,
        scale,
        N, 
        round_scale,
        num_stages=3,
        num_warps=16
    )
    return out, scale


# y = x @ w
# dx = y @ wT
# dwT = yT @ x
@triton.jit
def transpose_row_quant_kernel(x_ptr, q_ptr, s_ptr, M, N, H: tl.constexpr, W: tl.constexpr):

    pid = tl.program_id(axis=0)
    # col-wise read, row-wise write
    # read block: [BLOCK_SIZE, B]
    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:]
    indices = tl.arange(0, H)
    m = tl.cdiv(M, H)
    x_max = tl.zeros((W,),dtype=tl.float32)+1e-30
    for i in range(m):
        x = tl.load(x_ptr+offs,mask=i*H+indices[:,None]<M)
        x_max = tl.maximum(tl.max(tl.abs(x), axis=0),x_max)
        offs += H*N

    scale = x_max/448.0
    s = (1.0/scale)[:,None]

    tl.store(s_ptr+pid*W+tl.arange(0,W), scale)

    offs = pid*W + tl.arange(0, H)[:,None]*N + tl.arange(0, W)[None,:]
    toffs = pid*W*M + tl.arange(0, W)[:,None]*M + tl.arange(0, H)[None,:]
    for i in range(m):
        x = tl.trans(tl.load(x_ptr + offs,mask=i*H+indices[:,None]<M))
        x = (x*s).to(q_ptr.dtype.element_ty)
        tl.store(q_ptr + toffs, x, mask=i*H+indices[None,:]<M)
        offs += H*N
        toffs += H


def triton_transpose_row_quant(x, side=0):
    M, N = x.shape 
    H = 1024
    W = 16
    x_q = torch.empty((N, M),dtype=torch.float8_e4m3fn,device=x.device)
    x_scale = torch.empty((N, 1),dtype=torch.float32,device=x.device)
    grid = (N//W, )
    transpose_row_quant_kernel[grid](
        x, x_q, x_scale,
        M, N,
        H, W,
        num_stages=6,
        num_warps=4
    )
    return x_q, x_scale



def triton_channel_quant_nt(x,w):
    xq,x_scale = triton_row_quant(x)
    wq,w_scale = triton_row_quant(w)
    return xq,x_scale,wq,w_scale

def triton_channel_quant_nn(y,w):
    yq,y_scale = triton_row_quant(y)
    wq,w_scale = triton_transpose_row_quant(w)
    return yq,y_scale,wq,w_scale

def triton_channel_quant_tn(y,x):
    yq,y_scale = triton_transpose_row_quant(y)
    xq,x_scale = triton_transpose_row_quant(x)
    return yq,y_scale,xq,x_scale



def channel_quant_forward(x,w):
    x_q,x_scale,w_q,w_scale = triton_channel_quant_nt(x, w)
    output = torch._scaled_mm(x_q,
                                    w_q.t(),
                                    scale_a=x_scale,
                                    scale_b=w_scale.view(1,-1),
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,x_q,w_q,x_scale,w_scale

def channel_quant_backward(y,w):

    y_q,y_scale,w_q,w_scale = triton_channel_quant_nn(y, w)
    # print(f'{y.shape=} {w.shape=} {y_q.shape=} {y_scale.shape=} {w_q.shape=} {w_scale.shape=}')
    output = torch._scaled_mm(y_q,
                                    w_q.t(),
                                    scale_a=y_scale,
                                    scale_b=w_scale.view(1,-1),
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,w_q,y_scale,w_scale


def channel_quant_update(y,x):
    y_q,y_scale,x_q,x_scale = triton_channel_quant_tn(y, x)
    output = torch._scaled_mm(y_q,
                                    x_q.t(),
                                    scale_a=y_scale,
                                    scale_b=x_scale.view(1,-1),
                                    out_dtype=torch.bfloat16,
                                    use_fast_accum=True)
    return output,y_q,x_q,y_scale,x_scale


def fp8_channel_f_and_b(x,w,y):
    channel_quant_forward(x, w)
    channel_quant_backward(y, w)
    channel_quant_update(y,x)