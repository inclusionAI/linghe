import torch

from linghe.utils.silu import (triton_silu_and_block_quant_forward,
                               triton_silu_and_block_quant_backward,
                               triton_batch_weighted_silu_and_block_quant_forward,
                               triton_batch_weighted_silu_and_block_quant_backward,
                               )


class BlockSiluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, quantizer, grad_quantizer, cls):
        shape = input.shape
        assert len(shape) == 3
        input_view = input.view(shape[0] * shape[1], shape[2])
        ctx.grad_quantizer = grad_quantizer
        ctx.input_requires_grad = input.requires_grad
        ctx.shape = shape
        ctx.cls = cls
        ctx.save_for_backward(input)

        x_q, x_scale, xt_q, xt_scale = triton_silu_and_block_quant_forward(
            input_view,
            round_scale=quantizer.force_pow_2_scales)
        output_shape = (shape[0], shape[1], shape[2] // 2)
        transpose_shape = (shape[2] // 2, shape[0], shape[1])
        output = cls(
            shape=output_shape,
            dtype=input.dtype,
            fp8_dtype=quantizer.dtype,
            rowwise_data=x_q.view(output_shape),
            rowwise_scale_inv=x_scale,
            columnwise_data=xt_q.view(transpose_shape),
            columnwise_scale_inv=xt_scale,
            quantizer=quantizer,
            requires_grad=input.requires_grad,
            is_2D_scaled=False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        shape = grad_output.shape
        grad_output_view = grad_output.view(shape[0] * shape[1], shape[2])
        input, = ctx.saved_tensors
        grad_quantizer = ctx.grad_quantizer
        input_view = input.view(shape[0] * shape[1], shape[2] * 2)
        x_q, x_scale, xt_q, xt_scale = triton_silu_and_block_quant_backward(
            grad_output_view,
            input_view,
            round_scale=grad_quantizer.force_pow_2_scales)
        output = ctx.cls(
            shape=ctx.shape,
            dtype=grad_output.dtype,
            fp8_dtype=grad_quantizer.dtype,
            rowwise_data=x_q.view(ctx.shape),
            rowwise_scale_inv=x_scale,
            columnwise_data=xt_q.view(ctx.shape[2], shape[0], shape[1]),
            columnwise_scale_inv=xt_scale,
            quantizer=grad_quantizer,
            requires_grad=ctx.input_requires_grad,
            is_2D_scaled=False
        )

        return output, None, None, None


def block_silu_impl(input, quantizer, grad_quantizer, cls):
    output = BlockSiluFunction.apply(input, quantizer, grad_quantizer, cls)
    return output


class BlockBatchWeightedSiluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, counts, splits, quantizers,
                grad_quantizers, cls, is_recomputing):
        shape = input.shape
        ctx.grad_quantizers = grad_quantizers
        ctx.input_requires_grad = input.requires_grad
        ctx.shape = shape
        ctx.splits = splits
        ctx.cls = cls
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
         xt_scale) = triton_batch_weighted_silu_and_block_quant_forward(input,
                                                                        weights,
                                                                        counts,
                                                                        splits=splits,
                                                                        round_scale=
                                                                        quantizers[
                                                                            0].force_pow_2_scales,
                                                                        output_mode=output_mode)

        output = cls(
            shape=x_q.shape,
            dtype=input.dtype,
            fp8_dtype=quantizers[0].dtype,
            rowwise_data=x_q,
            rowwise_scale_inv=x_scale,
            columnwise_data=xt_q,
            columnwise_scale_inv=xt_scale,
            quantizer=quantizers,
            requires_grad=input.requires_grad,
            is_2D_scaled=False
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weights, counts = ctx.saved_tensors
        grad_quantizers = ctx.grad_quantizers
        (x_q,
         x_scale,
         wgrad,
         xt_q,
         xt_scale) = triton_batch_weighted_silu_and_block_quant_backward(
            grad_output,
            input,
            weights,
            counts,
            splits=ctx.splits,
            round_scale=grad_quantizers[0].force_pow_2_scales)
        output = ctx.cls(
            shape=ctx.shape,
            dtype=grad_output.dtype,
            fp8_dtype=grad_quantizers[0].dtype,
            rowwise_data=x_q,
            rowwise_scale_inv=x_scale,
            columnwise_data=xt_q,
            columnwise_scale_inv=xt_scale,
            quantizer=grad_quantizers,
            requires_grad=ctx.input_requires_grad,
            is_2D_scaled=False
        )

        return output, wgrad, None, None, None, None, None, None


def block_batch_weighted_silu_impl(input, weights, counts, splits, quantizers,
                                   grad_quantizers, cls, is_recomputing=None):
    assert input.ndim == 2
    output = BlockBatchWeightedSiluFunction.apply(input,
                                                  weights,
                                                  counts,
                                                  splits,
                                                  quantizers,
                                                  grad_quantizers,
                                                  cls,
                                                  is_recomputing)
    return output
