import torch

from linghe.utils.silu import (triton_silu_and_block_quant_forward,
                               triton_silu_and_block_quant_backward,
                               triton_batch_weighted_silu_and_block_quant_forward,
                               triton_batch_weighted_silu_and_block_quant_backward,
                               triton_silu_and_mxfp8_quant_forward,
                               triton_silu_and_mxfp8_quant_backward,
                               triton_batch_weighted_silu_and_mxfp8_quant_forward,
                               triton_batch_weighted_silu_and_mxfp8_quant_backward,
                               triton_silu_and_smooth_quant_forward,
                               triton_silu_and_smooth_quant_backward,
                               triton_batch_weighted_silu_and_smooth_quant_forward,
                               triton_batch_weighted_silu_and_smooth_quant_backward)


class BlockSiluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, quantizer, grad_quantizer, cls, limit):
        shape = input.shape
        assert len(shape) == 3
        input_view = input.view(shape[0] * shape[1], shape[2])
        ctx.grad_quantizer = grad_quantizer
        ctx.input_requires_grad = input.requires_grad
        ctx.shape = shape
        ctx.cls = cls
        ctx.limit = limit
        ctx.save_for_backward(input)

        round_scale = quantizer.force_pow_2_scales
        x_q, x_scale, xt_q, xt_scale = triton_silu_and_block_quant_forward(input_view,
                                                                           round_scale=round_scale,
                                                                           limit=limit)
        output_shape = (shape[0], shape[1], shape[2] // 2)
        transpose_shape = (shape[2] // 2, shape[0], shape[1])
        output = cls(shape=output_shape,
                     dtype=input.dtype,
                     fp8_dtype=quantizer.dtype,
                     rowwise_data=x_q.view(output_shape),
                     rowwise_scale_inv=x_scale,
                     columnwise_data=xt_q.view(transpose_shape),
                     columnwise_scale_inv=xt_scale,
                     quantizer=quantizer,
                     requires_grad=input.requires_grad,
                     is_2D_scaled=False)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        shape = grad_output.shape
        grad_output_view = grad_output.view(shape[0] * shape[1], shape[2])
        input, = ctx.saved_tensors
        grad_quantizer = ctx.grad_quantizer
        input_view = input.view(shape[0] * shape[1], shape[2] * 2)
        round_scale = grad_quantizer.force_pow_2_scales
        x_q, x_scale, xt_q, xt_scale = triton_silu_and_block_quant_backward(grad_output_view,
                                                                            input_view,
                                                                            round_scale=round_scale,
                                                                            limit=ctx.limit)
        output = ctx.cls(shape=ctx.shape,
                         dtype=grad_output.dtype,
                         fp8_dtype=grad_quantizer.dtype,
                         rowwise_data=x_q.view(ctx.shape),
                         rowwise_scale_inv=x_scale,
                         columnwise_data=xt_q.view(ctx.shape[2], shape[0], shape[1]),
                         columnwise_scale_inv=xt_scale,
                         quantizer=grad_quantizer,
                         requires_grad=ctx.input_requires_grad,
                         is_2D_scaled=False)

        return output, None, None, None, None


def block_silu_impl(input, quantizer, grad_quantizer, cls, limit=None):
    output = BlockSiluFunction.apply(input, quantizer, grad_quantizer, cls, limit)
    return output


class BlockBatchWeightedSiluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                input,
                weights,
                counts,
                splits,
                quantizers,
                grad_quantizers,
                cls,
                limit,
                is_recomputing):
        shape = input.shape
        ctx.grad_quantizers = grad_quantizers
        ctx.input_requires_grad = input.requires_grad
        ctx.shape = shape
        ctx.splits = splits
        ctx.cls = cls
        ctx.limit = limit
        ctx.save_for_backward(input, weights, counts)

        if is_recomputing is None:
            output_mode = 2
        elif is_recomputing:
            output_mode = 1
        else:
            output_mode = 0

        round_scale = quantizers[0].force_pow_2_scales

        (x_q,
         x_scale,
         xt_q,
         xt_scale) = triton_batch_weighted_silu_and_block_quant_forward(input,
                                                                        weights,
                                                                        counts,
                                                                        splits=splits,
                                                                        limit=limit,
                                                                        round_scale=round_scale,
                                                                        output_mode=output_mode)

        output = cls(shape=x_q.shape,
                     dtype=input.dtype,
                     fp8_dtype=quantizers[0].dtype,
                     rowwise_data=x_q,
                     rowwise_scale_inv=x_scale,
                     columnwise_data=xt_q,
                     columnwise_scale_inv=xt_scale,
                     quantizer=quantizers,
                     requires_grad=input.requires_grad,
                     is_2D_scaled=False)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weights, counts = ctx.saved_tensors
        grad_quantizers = ctx.grad_quantizers
        round_scale = grad_quantizers[0].force_pow_2_scales

        (x_q,
         x_scale,
         wgrad,
         xt_q,
         xt_scale) = triton_batch_weighted_silu_and_block_quant_backward(grad_output,
                                                                         input,
                                                                         weights,
                                                                         counts,
                                                                         splits=ctx.splits,
                                                                         round_scale=round_scale,
                                                                         limit=ctx.limit)
        output = ctx.cls(shape=ctx.shape,
                         dtype=grad_output.dtype,
                         fp8_dtype=grad_quantizers[0].dtype,
                         rowwise_data=x_q,
                         rowwise_scale_inv=x_scale,
                         columnwise_data=xt_q,
                         columnwise_scale_inv=xt_scale,
                         quantizer=grad_quantizers,
                         requires_grad=ctx.input_requires_grad,
                         is_2D_scaled=False)

        return output, wgrad, None, None, None, None, None, None, None


def block_batch_weighted_silu_impl(input, weights, counts, splits, quantizers,
                                   grad_quantizers, cls,
                                   limit=None,
                                   is_recomputing=None):
    assert input.ndim == 2
    output = BlockBatchWeightedSiluFunction.apply(input,
                                                  weights,
                                                  counts,
                                                  splits,
                                                  quantizers,
                                                  grad_quantizers,
                                                  cls,
                                                  limit,
                                                  is_recomputing)
    return output


class MXFP8SiluFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, quantizer, grad_quantizer, cls, limit):
        shape = input.shape
        assert len(shape) == 3
        input = input.view(shape[0] * shape[1], shape[2])
        ctx.grad_quantizer = grad_quantizer
        ctx.input_requires_grad = input.requires_grad
        ctx.shape = shape
        ctx.cls = cls
        ctx.limit = limit
        ctx.save_for_backward(input)
        x_q, x_scale, xt_q, xt_scale = triton_silu_and_mxfp8_quant_forward(input, limit=limit)

        output_shape = (shape[0], shape[1], shape[2] // 2)
        # transpose_shape = (shape[2]//2, shape[0], shape[1])
        output = cls(shape=output_shape,
                     dtype=input.dtype,
                     fp8_dtype=quantizer.dtype,
                     rowwise_data=x_q.view(output_shape),
                     rowwise_scale_inv=x_scale,
                     columnwise_data=xt_q.view(output_shape),
                     columnwise_scale_inv=xt_scale,
                     quantizer=quantizer,
                     requires_grad=input.requires_grad, )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        shape = grad_output.shape
        grad_output = grad_output.view(shape[0] * shape[1], shape[2])
        input, = ctx.saved_tensors
        grad_quantizer = ctx.grad_quantizer
        x_q, x_scale, xt_q, xt_scale = triton_silu_and_mxfp8_quant_backward(grad_output,
                                                                            input,
                                                                            limit=ctx.limit)
        output = ctx.cls(shape=ctx.shape,
                         dtype=grad_output.dtype,
                         fp8_dtype=grad_quantizer.dtype,
                         rowwise_data=x_q.view(ctx.shape) if x_q is not None else None,
                         rowwise_scale_inv=x_scale,
                         columnwise_data=xt_q.view(ctx.shape) if xt_q is not None else None,
                         columnwise_scale_inv=xt_scale,
                         quantizer=grad_quantizer,
                         requires_grad=ctx.input_requires_grad, )

        return output, None, None, None, None


def mxfp8_silu_impl(input, quantizer, grad_quantizer, cls, limit=None):
    # input: [length,bs,dim]
    output = MXFP8SiluFunction.apply(input, quantizer, grad_quantizer, cls, limit)
    return output


class MXFP8BatchWeightedSiluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, counts, splits, quantizers,
                grad_quantizers, cls, limit, is_recomputing):
        shape = input.shape
        ctx.grad_quantizers = grad_quantizers
        ctx.input_requires_grad = input.requires_grad
        ctx.shape = shape
        ctx.splits = splits
        ctx.cls = cls
        ctx.limit = limit
        ctx.save_for_backward(input, weights, counts)

        if is_recomputing is None:
            output_mode = 2
        elif is_recomputing:
            output_mode = 1
        else:
            output_mode = 0

        (x_q,
         x_scale,
         xt_q,
         xt_scale) = triton_batch_weighted_silu_and_mxfp8_quant_forward(input,
                                                                        weights,
                                                                        counts,
                                                                        splits=splits,
                                                                        limit=limit,
                                                                        output_mode=output_mode)

        output = cls(shape=x_q.shape,
                     dtype=input.dtype,
                     fp8_dtype=quantizers[0].dtype,
                     rowwise_data=x_q,
                     rowwise_scale_inv=x_scale,
                     columnwise_data=xt_q,
                     columnwise_scale_inv=xt_scale,
                     quantizer=quantizers,
                     requires_grad=input.requires_grad, )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weights, counts = ctx.saved_tensors
        quantizers = ctx.grad_quantizers
        (x_q,
         x_scale,
         wgrad,
         xt_q,
         xt_scale) = triton_batch_weighted_silu_and_mxfp8_quant_backward(grad_output,
                                                                         input,
                                                                         weights,
                                                                         counts,
                                                                         splits=ctx.splits,
                                                                         limit=ctx.limit)
        output = ctx.cls(shape=ctx.shape,
                         dtype=grad_output.dtype,
                         fp8_dtype=quantizers[0].dtype,
                         rowwise_data=x_q,
                         rowwise_scale_inv=x_scale,
                         columnwise_data=xt_q,
                         columnwise_scale_inv=xt_scale,
                         quantizer=quantizers,
                         requires_grad=ctx.input_requires_grad, )

        return output, wgrad, None, None, None, None, None, None, None


def mxfp8_batch_weighted_silu_impl(input, weights, counts, splits, quantizers,
                                   grad_quantizers, cls, limit=None, is_recomputing=None):
    assert input.ndim == 2
    output = MXFP8BatchWeightedSiluFunction.apply(input,
                                                  weights,
                                                  counts,
                                                  splits,
                                                  quantizers,
                                                  grad_quantizers,
                                                  cls,
                                                  limit,
                                                  is_recomputing)
    return output


class SmoothSiluFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, quantizer, grad_quantizer, cls):
        shape = input.shape
        assert len(shape) == 3
        input = input.view(shape[0] * shape[1], shape[2])
        ctx.grad_quantizer = grad_quantizer
        ctx.shape = shape
        ctx.cls = cls
        ctx.save_for_backward(input)
        x_q, x_scale = triton_silu_and_smooth_quant_forward(input,
                                                            smooth_scale=quantizer.smooth_scale,
                                                            round_scale=quantizer.force_pow_2_scales)
        output = cls(shape=(shape[0], shape[1], shape[2] // 2),
                     dtype=input.dtype,
                     fp8_dtype=quantizer.dtype,
                     rowwise_data=x_q,
                     rowwise_scale_inv=x_scale,
                     columnwise_data=None,
                     columnwise_scale_inv=quantizer.smooth_scale,
                     quantizer=quantizer,
                     requires_grad=input.requires_grad, )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        shape = grad_output.shape
        grad_output = grad_output.view(shape[0] * shape[1], shape[2])
        input, = ctx.saved_tensors
        grad_quantizer = ctx.grad_quantizer
        # we use requant implementation,
        # so must use round_scale=True to avoid second quantization error
        round_scale = True  # quantizer.force_pow_2_scales 
        x_q, x_scale, xt_q, xt_scale = triton_silu_and_smooth_quant_backward(grad_output,
                                                                             input,
                                                                             smooth_scale=grad_quantizer.smooth_scale_inv,
                                                                             transpose_smooth_scale=grad_quantizer.transpose_smooth_scale_inv,
                                                                             reverse=True,
                                                                             round_scale=round_scale)

        output = ctx.cls(shape=ctx.shape,
                         dtype=grad_output.dtype,
                         fp8_dtype=grad_quantizer.dtype,
                         rowwise_data=x_q,
                         rowwise_scale_inv=x_scale,
                         columnwise_data=xt_q,
                         columnwise_scale_inv=xt_scale,
                         quantizer=grad_quantizer,
                         requires_grad=False, )

        return output, None, None, None


def smooth_silu_impl(input, quantizer, grad_quantizer, cls):
    # input: [length, bs, dim]
    output = SmoothSiluFunction.apply(input, quantizer, grad_quantizer, cls)
    return output


class SmoothBatchWeightedSiluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, counts, splits, quantizers,
                grad_quantizers, cls):
        shape = inputs.shape
        ctx.grad_quantizers = grad_quantizers
        ctx.shape = shape
        ctx.save_for_backward(inputs, weights, counts)
        ctx.splits = splits
        ctx.cls = cls
        smooth_scales = torch.stack([x.smooth_scale for x in quantizers], 0)
        round_scale = quantizers[0].force_pow_2_scales
        x_q, x_scale = triton_batch_weighted_silu_and_smooth_quant_forward(inputs,
                                                                           weights,
                                                                           counts,
                                                                           smooth_scale=smooth_scales,
                                                                           round_scale=round_scale)
        output = cls(shape=x_q.shape,
                     dtype=inputs.dtype,
                     fp8_dtype=quantizers[0].dtype,
                     rowwise_data=x_q,
                     rowwise_scale_inv=x_scale,
                     columnwise_data=None,
                     columnwise_scale_inv=smooth_scales,
                     quantizer=quantizers,
                     requires_grad=inputs.requires_grad)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights, counts = ctx.saved_tensors
        grad_quantizers = ctx.grad_quantizers
        # smooth_scale_inv may not exist in forward step
        smooth_scales = torch.stack([x.smooth_scale_inv for x in grad_quantizers], 0)
        transpose_smooth_scales = torch.cat([x.transpose_smooth_scale_inv for x in grad_quantizers], 0)
        # we use requant implementation, so must use round_scale=True to avoid second quantization error
        round_scale = True  # grad_quantizers[0].force_pow_2_scales
        x_q, x_scale, wgrad, xt_q, xt_scale = \
            triton_batch_weighted_silu_and_smooth_quant_backward(grad_output,
                                                                 inputs,
                                                                 weights,
                                                                 counts,
                                                                 smooth_scale=smooth_scales,
                                                                 transpose_smooth_scale=transpose_smooth_scales,
                                                                 splits=ctx.splits,
                                                                 reverse=True,
                                                                 round_scale=round_scale)
        output = ctx.cls(shape=ctx.shape,
                         dtype=grad_output.dtype,
                         fp8_dtype=grad_quantizers[0].dtype,
                         rowwise_data=x_q,
                         rowwise_scale_inv=x_scale,
                         columnwise_data=xt_q,
                         columnwise_scale_inv=xt_scale,
                         quantizer=grad_quantizers,
                         requires_grad=False, )

        return output, wgrad, None, None, None, None, None


def smooth_batch_weighted_silu_impl(inputs, weights, counts, splits, quantizers,
                                    grad_quantizers, cls):
    # TODO(nanxiao): support recomputation
    assert inputs.ndim == 2
    output = SmoothBatchWeightedSiluFunction.apply(inputs, weights, counts,
                                                   splits, quantizers,
                                                   grad_quantizers, cls)
    return output
