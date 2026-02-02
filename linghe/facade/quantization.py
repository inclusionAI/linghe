# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import Optional, List

import torch

from linghe.quant.smooth import (triton_smooth_quant,
                                 triton_transpose_smooth_quant,
                                 triton_batch_smooth_quant,
                                 triton_batch_transpose_smooth_quant,
                                 triton_subrow_smooth_quant)


"""
smooth quantization v2 for fp8 training
1.1 calculate M = max(abs(w)).
1.2 calculate smooth_scale = sqrt(M)

2.1 quantize weight w_q, w_s = quant(w / smooth_scale)
2.2 set weight quantizer.smooth_scale = smooth_scale
2.3 set weight quantizer.smooth_scale_inv = w_s
3.4 set quantized weight._columnwise_scale_inv = smooth_scale

3.1 set activation quantizer.smooth_scale = 1/smooth_scale
3.1 quantize activation a_q, a_s = quant(a * smooth_scale)
3.3 set activation quantizer.transpose_smooth_scale_inv = a_s

4.1 set grad quantizer.smooth_scale_inv = w_s
4.2 set grad quantizer.transpose_smooth_scale_inv = a_s
4.3 quantize grad g_q, g_s = quant(g * smooth_scale_inv)
4.4 quantize transposed grad gt_q, gt_s = quant(g * transpose_smooth_scale_inv)
"""

class SmoothQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, quantizer, cls):
        shape = hidden_states.shape 
        if len(shape) == 3:
            hidden_states = hidden_states.view(-1,shape[-1])
        x_q, x_scale = triton_smooth_quant(hidden_states,
                                              quantizer.smooth_scale, 
                                              reverse=False, 
                                              round_scale=quantizer.force_pow_2_scales)
        output = cls(
            shape=shape,
            dtype=hidden_states.dtype,
            fp8_dtype=quantizer.dtype,
            rowwise_data=x_q,
            rowwise_scale_inv=x_scale,
            columnwise_data=None,
            columnwise_scale_inv=quantizer.smooth_scale,
            quantizer=quantizer,
            requires_grad=hidden_states.requires_grad,
        )
        return output

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output, None, None
    

def smooth_quantize(hidden_states, quantizer, cls):
    return SmoothQuantize.apply(hidden_states, quantizer, cls)


class ReverseSmoothQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, quantizer, cls):
        ctx.quantizer = quantizer  # grad_quantizer
        ctx.input_requires_grad = hidden_states.requires_grad
        ctx.cls = cls
        return hidden_states

    @staticmethod 
    def backward(ctx, grad_output):
        shape = grad_output.shape  # rank-3 tensor
        grad_output = grad_output.view(-1,shape[-1])
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)

        y_q, y_scale = triton_smooth_quant(grad_output,
                                              ctx.quantizer.smooth_scale_inv,
                                              reverse=True,
                                              round_scale=False)
        yt_q, yt_scale = triton_transpose_smooth_quant(grad_output,
                                                    ctx.quantizer.transpose_smooth_scale_inv,
                                                    reverse=True,
                                                    pad=True,
                                                    round_scale=False)
        # import math
        # if math.isnan(y_q.float().max()) or math.isnan(yt_q.float().max()):
        #     print(f'ReverseSmoothQuantize {grad_output.max()=} {y_scale.max()=} {yt_scale.max()=}')
        output = ctx.cls(
            shape=shape,
            dtype=grad_output.dtype,
            fp8_dtype=ctx.quantizer.dtype,
            rowwise_data=y_q,
            rowwise_scale_inv=y_scale,
            columnwise_data=yt_q,
            columnwise_scale_inv=yt_scale,
            quantizer=ctx.quantizer,
            requires_grad=ctx.input_requires_grad
        )
        return output, None, None



def reverse_smooth_quantize(hidden_states, quantizer, cls):
    return ReverseSmoothQuantize.apply(hidden_states, quantizer, cls)


class BatchSmoothQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, token_count_per_expert, quantizers, splits, cls):
        shape = hidden_states.shape 
        if len(shape) == 3:
            hidden_states = hidden_states.view(-1,shape[-1])
        if token_count_per_expert is None:
            token_count_per_expert = torch.tensor(splits).cuda(non_blocking=True)
        smooth_scales = [x.smooth_scale for x in quantizers]
        if any(x is None for x in smooth_scales):
            smooth_scales = torch.ones((len(smooth_scales), hidden_states.shape[-1]),
                                       dtype=torch.float32,
                                       device=hidden_states.device)
            for i, x in enumerate(quantizers):
                x.smooth_scale = smooth_scales[i]
        else:
            smooth_scales = torch.stack(smooth_scales, 0)
        x_q, x_scale = triton_batch_smooth_quant(hidden_states,
                                              smooth_scales, 
                                              token_count_per_expert,
                                              reverse=False, 
                                              round_scale=quantizers[0].force_pow_2_scales)
        output = cls(
            shape=shape,
            dtype=hidden_states.dtype,
            fp8_dtype=quantizers[0].dtype,
            rowwise_data=x_q,
            rowwise_scale_inv=x_scale,
            columnwise_data=None,
            columnwise_scale_inv=smooth_scales,
            quantizer=quantizers,
            requires_grad=hidden_states.requires_grad,
        )
        return output

    @staticmethod 
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None
    

def batch_smooth_quantize(hidden_states, token_count_per_expert, quantizers, splits, cls):
    return BatchSmoothQuantize.apply(hidden_states, token_count_per_expert, quantizers, splits, cls)


class BatchReverseSmoothQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, token_count_per_expert, quantizers, splits, cls):
        ctx.quantizers = quantizers  # grad_quantizer
        ctx.cls = cls
        ctx.splits = splits
        if token_count_per_expert is None:
            token_count_per_expert = torch.tensor(splits).cuda(non_blocking=True)
        ctx.token_count_per_expert = token_count_per_expert
        ctx.dtype = hidden_states.dtype
        ctx.device = hidden_states.device
        ctx.dim = hidden_states.shape[-1]
        return hidden_states

    @staticmethod 
    def backward(ctx, grad_output):
        shape = grad_output.shape  # rank-3 tensor
        grad_output = grad_output.view(-1,shape[-1])
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)

        smooth_scale_invs = [x.smooth_scale_inv for x in ctx.quantizers]
        smooth_scale_invs = torch.stack(smooth_scale_invs, 0)

        y_q, y_scale = triton_batch_smooth_quant(grad_output,
                                              smooth_scale_invs, 
                                              ctx.token_count_per_expert,
                                              reverse=True, 
                                              round_scale=ctx.quantizers[0].force_pow_2_scales)

        transpose_smooth_scale_invs = [x.transpose_smooth_scale_inv for x in ctx.quantizers]
        transpose_smooth_scale_invs = torch.cat(transpose_smooth_scale_invs, 0)
        assert all(x is not None for x in transpose_smooth_scale_invs)

        yt_q, yt_scale = triton_batch_transpose_smooth_quant(grad_output,
                                            transpose_smooth_scale_invs,
                                            ctx.token_count_per_expert,
                                            ctx.splits,
                                            reverse=True,
                                            round_scale=False)

        # import math
        # if math.isnan(yt_q.float().max()):
        #     print(f'BatchReverseSmoothQuantize {grad_output.max()=} {yt_scale.max()=}')
        output = ctx.cls(
            shape=shape,
            dtype=grad_output.dtype,
            fp8_dtype=ctx.quantizers[0].dtype,
            rowwise_data=y_q,
            rowwise_scale_inv=y_scale,
            columnwise_data=yt_q,
            columnwise_scale_inv=yt_scale,
            quantizer=ctx.quantizers,
            requires_grad=False
        )
        return output, None, None, None, None


def batch_reverse_smooth_quantize(hidden_states, token_count_per_expert, quantizers, splits, cls):
    return BatchReverseSmoothQuantize.apply(hidden_states, token_count_per_expert, quantizers, splits, cls)



"""
megatron fp8 training steps:
step 0: init w smooth scale w_smooth
step 1: smooth and quant w after w is updated by optimizer
step 2: in forward step, columnwise smooth x and rowwise quant x, calc y=x@w; 
            meanwhile, record the columnwise max of x, it is used to update w_smooth
step 3: in dgrad step, columnwise smooth y and rowwise quant y, transpose x, calc dx=y@wT 
step 4: in wgrad step, dequant then smooth an then quant y_q to get yt_q, calc dw=yT@x

alternative (it's not suitable for fp8 combine):
step 4: in wgrad step, rowwise smooth y and columnwise quant y and transpose to get yt_q, calc dw=yT@x

"""

"""
divide x by smooth_scale and row-wise quantization
smooth scale is updated by square root of x's column-wise maxs, and set in weight's x_maxs attr

transpose: transpose quantized x for wgrad
pad: # pad M to be multiplier of 32, including quant scales and transposed x

"""


# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_smooth_quant_activation(x, smooth_scale, x_q=None, x_scale=None,
                              xt_q=None,
                              transpose=True, pad=True, round_scale=False):
    """"""
    x_q, x_scale = triton_smooth_quant(x, smooth_scale, x_q=x_q,
                                               x_scale=x_scale, reverse=False,
                                               round_scale=round_scale)

    if transpose:
        xt_q = triton_transpose_and_pad(x_q, out=xt_q, pad=pad)
    else:
        xt_q = None
    xt_scale = smooth_scale

    return x_q, x_scale, xt_q, xt_scale


# y = x @ w
# dx = y @ wT
# dwT = yT @ x
def triton_smooth_quant_gradient(y,
                                 smooth_scale,
                                 transpose_smooth_scale,
                                 reverse=True,
                                 transpose=True,
                                 pad=True,
                                 round_scale=False):
    """"""
    assert reverse, ("args `smooth_scale` and/or `transpose_smooth_scale` "
                     "must be in reciprocal format in triton_smooth_quant_grad")
    y_q, y_scale = triton_smooth_quant(y, smooth_scale, reverse=True,
                                          round_scale=round_scale)
    if transpose:
        yt_q, yt_scale = triton_transpose_smooth_quant(y,
                                                       transpose_smooth_scale,
                                                       reverse=True,
                                                       pad=pad,
                                                       round_scale=round_scale)
    else:
        yt_q, yt_scale = None, None

    return y_q, y_scale, yt_q, yt_scale


def triton_smooth_quant_weight(w,
                               smooth_scale,
                               w_q,
                               quant_scale,
                               subrow_scales, offset=0,
                               round_scale=False):
    """"""
    assert w.ndim == 1
    assert w_q.size(1) == smooth_scale.size(0)

    size = w.numel()
    M, N = w_q.shape

    if size == M * N:
        triton_smooth_quant(w.view(M, N), smooth_scale, x_q=w_q,
                            x_scale=quant_scale,
                            round_scale=round_scale)
    elif offset % N == 0 and size % N == 0:
        n_row = size // N
        row_id = offset // N
        w_q_slice = w_q[row_id:row_id + n_row]
        quant_scale_slice = quant_scale[row_id:row_id + n_row]
        triton_smooth_quant(w.view(n_row, N), smooth_scale, x_q=w_q_slice,
                            x_scale=quant_scale_slice,
                            round_scale=round_scale)
    else:
        row_si = (offset - 1) // N + 1
        row_ei = (offset + size) // N
        col_si = offset % N
        col_ei = (offset + size) % N
        n_row = row_ei - row_si
        mw_offset = 0 if col_si == 0 else N - col_si
        w_q_slice = w_q[row_si:row_ei]
        quant_scale_slice = quant_scale[row_si:row_ei]
        w_slice = w[mw_offset:mw_offset + n_row * N].view(n_row, N)
        triton_smooth_quant(w_slice,
                            smooth_scale,
                            x_q=w_q_slice,
                            x_scale=quant_scale_slice,
                            round_scale=round_scale)

        # subrow scale is writed by the row with leading master weights
        if col_si > 0 or col_ei > 0:
            triton_subrow_smooth_quant(w,
                                       smooth_scale,
                                       w_q,
                                       quant_scale,
                                       subrow_scales,
                                       offset,
                                       size,
                                       reverse=False,
                                       round_scale=round_scale)
