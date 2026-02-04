# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

from typing import Optional, List

import torch

from linghe.quant.mxfp8 import triton_batch_mxfp8_quant
from linghe.utils.gather import (
    triton_permute_with_mask_map,
    triton_make_row_id_map,
    triton_make_row_id_map_and_index,
    triton_batch_block_pad_permute_with_indices,
    triton_batch_mxfp8_permute_with_indices,
    triton_batch_smooth_permute_with_indices,
    triton_batch_transpose_smooth_permute_with_indices,
    triton_batch_smooth_fused_permute_with_indices,
    triton_batch_transpose_smooth_fused_permute_with_indices
)
from linghe.utils.scatter import triton_unpermute_with_mask_map



class _PaddedPermute(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            tokens,
            probs,
            routing_map,
            tokens_per_expert_cuda_tensor,
            tokens_per_expert_list,
            multiple
    ):
        """Forward function."""
        num_tokens, hidden_dim = tokens.shape

        row_id_map = triton_make_row_id_map(routing_map, multiple_of=multiple)
        num_out_tokens = sum(
            [((x - 1) // multiple + 1) * multiple for x in tokens_per_expert_list])

        ctx.num_tokens = num_tokens
        ctx.hidden_dim = hidden_dim
        ctx.prob_shape = probs.shape
        ctx.shape = tokens.shape
        ctx.row_id_map = row_id_map
        permuted_tokens, _, permuted_probs = triton_permute_with_mask_map(
            tokens,
            None,
            probs,
            row_id_map,
            num_out_tokens,
            contiguous=False,
            tokens_per_expert=tokens_per_expert_cuda_tensor,
        )
        ctx.save_for_backward(row_id_map)
        return permuted_tokens, permuted_probs, row_id_map

    @staticmethod
    def backward(ctx, grad_output, grad_prob, grad_map):
        """Backward function."""
        (row_id_map,) = ctx.saved_tensors
        output, prob_output = triton_unpermute_with_mask_map(
            grad_output, row_id_map, grad_prob
        )
        return (
            output.view(ctx.shape),
            prob_output.view(ctx.prob_shape),
            None,
            None,
            None,
            None
        )


def padded_permute(
        tokens,
        routing_map,
        tokens_per_expert_cuda_tensor,
        tokens_per_expert_list,
        probs: Optional[torch.Tensor] = None,
        multiple: int = 32
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.
    When drop_and_pad=True, in routing_map, the number of non-zeros in each column equals to
    expert capacity. This function exploits this feature to use ops that support cuda graph.
    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        tokens_per_expert (torch.Tensor): cpu tensor
    """

    permuted_input, permuted_probs, row_id_map = _PaddedPermute.apply(
        tokens,
        probs,
        routing_map,
        tokens_per_expert_cuda_tensor,
        tokens_per_expert_list,
        multiple
    )
    return permuted_input, permuted_probs, row_id_map


class _PaddedUnpermute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, permuted_tokens, row_id_map, tokens_per_expert,
                restore_shape):
        """Forward function."""
        num_tokens, hidden_size = restore_shape
        num_out_tokens = permuted_tokens.shape[0]
        n_experts = row_id_map.size(1)
        ctx.save_for_backward(row_id_map)
        ctx.input_requires_grad = permuted_tokens.requires_grad
        ctx.num_experts = n_experts
        ctx.restore_shape = restore_shape
        ctx.num_tokens = num_tokens
        ctx.num_out_tokens = num_out_tokens
        ctx.hidden_size = hidden_size
        ctx.tokens_per_expert = tokens_per_expert

        output, _ = triton_unpermute_with_mask_map(permuted_tokens, row_id_map,
                                                   None)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        (row_id_map,) = ctx.saved_tensors
        permuted_tokens, _, _ = triton_permute_with_mask_map(
            grad_output,
            None,
            None,
            row_id_map,
            ctx.num_out_tokens,
            contiguous=False,
            tokens_per_expert=ctx.tokens_per_expert,
        )

        return permuted_tokens, None, None, None


def padded_unpermute(
        permuted_tokens: torch.Tensor,
        row_id_map: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        restore_shape: torch.Size,
):
    output = _PaddedUnpermute.apply(
        permuted_tokens, row_id_map, tokens_per_expert, restore_shape
    )
    return output


class _BlockPaddedPermute(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            tokens,
            probs,
            routing_map,
            tokens_per_expert_cuda_tensor,
            tokens_per_expert_list,
            quantizers,
            cls,
    ):
        """Forward function."""
        num_tokens, hidden_dim = tokens.shape

        num_out_tokens = sum(
            [(x + 15) // 16 * 16 for x in tokens_per_expert_list])
        row_id_map, row_id_index = triton_make_row_id_map_and_index(
            routing_map, num_out_tokens, multiple_of=16
        )

        ctx.num_tokens = num_tokens
        ctx.hidden_dim = hidden_dim
        ctx.prob_shape = probs.shape
        ctx.shape = tokens.shape
        ctx.cls = cls
        x_q, x_scale, xt_q, xt_scale, permuted_probs = (
            triton_batch_block_pad_permute_with_indices(
                tokens,
                tokens_per_expert_cuda_tensor,
                row_id_index,
                tokens_per_expert_list,
                probs=probs,
                round_scale=quantizers[0].force_pow_2_scales,
            )
        )

        output = cls(
            shape=x_q.shape,
            dtype=tokens.dtype,
            fp8_dtype=quantizers[0].dtype,
            rowwise_data=x_q,
            rowwise_scale_inv=x_scale,
            columnwise_data=xt_q,
            columnwise_scale_inv=xt_scale,
            quantizer=quantizers,
            requires_grad=tokens.requires_grad,
            is_2D_scaled=False,
        )
        ctx.save_for_backward(row_id_map)
        return output, permuted_probs, row_id_map, row_id_index

    @staticmethod
    def backward(ctx, grad_output, grad_prob, grad_map, grad_index):
        """Backward function."""
        (row_id_map,) = ctx.saved_tensors
        output, prob_output = triton_unpermute_with_mask_map(
            grad_output, row_id_map, grad_prob
        )
        return (
            output.view(ctx.shape),
            prob_output.view(ctx.prob_shape),
            None,
            None,
            None,
            None,
            None,
        )


def block_padded_permute(
        tokens,
        routing_map,
        tokens_per_expert_cuda_tensor,
        tokens_per_expert_list,
        quantizers,
        cls,
        probs: Optional[torch.Tensor] = None,
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.
    When drop_and_pad=True, in routing_map, the number of non-zeros in each column equals to
    expert capacity. This function exploits this feature to use ops that support cuda graph.
    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        tokens_per_expert (torch.Tensor): cpu tensor
    """

    permuted_input, permuted_probs, row_id_map, row_id_index = (
        _BlockPaddedPermute.apply(
            tokens,
            probs,
            routing_map,
            tokens_per_expert_cuda_tensor,
            tokens_per_expert_list,
            quantizers,
            cls,
        )
    )
    return permuted_input, permuted_probs, row_id_map, row_id_index


class _BlockPaddedUnpermute(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            permuted_tokens,
            row_id_map,
            row_id_index,
            tokens_per_expert,
            splits,
            restore_shape,
            quantizers,
            cls,
    ):
        """Forward function."""
        num_tokens, hidden_size = restore_shape
        num_out_tokens = permuted_tokens.shape[0]
        n_experts = row_id_map.size(1)
        ctx.save_for_backward(row_id_index)
        ctx.input_requires_grad = permuted_tokens.requires_grad
        ctx.num_experts = n_experts
        ctx.restore_shape = restore_shape
        ctx.num_tokens = num_tokens
        ctx.num_out_tokens = num_out_tokens
        ctx.hidden_size = hidden_size
        ctx.tokens_per_expert = tokens_per_expert
        ctx.splits = splits
        ctx.quantizers = quantizers
        ctx.cls = cls

        output, _ = triton_unpermute_with_mask_map(permuted_tokens, row_id_map,
                                                   None)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        (row_id_index,) = ctx.saved_tensors

        quantizers = ctx.quantizers
        x_q, x_scale, xt_q, xt_scale, _ = triton_batch_block_pad_permute_with_indices(
            grad_output,
            ctx.tokens_per_expert,
            row_id_index,
            ctx.splits,
            round_scale=quantizers[0].force_pow_2_scales,
        )

        output = ctx.cls(
            shape=x_q.shape,
            dtype=grad_output.dtype,
            fp8_dtype=quantizers[0].dtype,
            rowwise_data=x_q,
            rowwise_scale_inv=x_scale,
            columnwise_data=xt_q,
            columnwise_scale_inv=xt_scale,
            quantizer=quantizers,
            requires_grad=False,
            is_2D_scaled=False,
        )

        return output, None, None, None, None, None, None, None


def block_padded_unpermute(
        permuted_tokens: torch.Tensor,
        row_id_map: torch.Tensor,
        row_id_index: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        splits: List,
        restore_shape: torch.Size,
        quantizers,
        cls,
):
    output = _BlockPaddedUnpermute.apply(
        permuted_tokens,
        row_id_map,
        row_id_index,
        tokens_per_expert,
        splits,
        restore_shape,
        quantizers,
        cls,
    )
    return output


class _MXFP8Permute(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            tokens,
            probs,
            routing_map,
            tokens_per_expert_cuda_tensor,
            tokens_per_expert_list,
            quantizers,
            cls,
    ):
        """Forward function."""
        num_tokens, hidden_dim = tokens.shape

        num_out_tokens = sum(tokens_per_expert_list)
        row_id_map, row_id_index = triton_make_row_id_map_and_index(
            routing_map, num_out_tokens
        )

        ctx.num_tokens = num_tokens
        ctx.hidden_dim = hidden_dim
        ctx.prob_shape = probs.shape
        ctx.shape = tokens.shape
        ctx.cls = cls
        x_q, x_scale, xt_q, xt_scale, permuted_probs = (
            triton_batch_mxfp8_permute_with_indices(
                tokens,
                tokens_per_expert_cuda_tensor,
                row_id_index,
                tokens_per_expert_list,
                probs=probs,
            )
        )

        output = cls(
            shape=x_q.shape,
            dtype=tokens.dtype,
            fp8_dtype=quantizers[0].dtype,
            rowwise_data=x_q,
            rowwise_scale_inv=x_scale,
            columnwise_data=xt_q,
            columnwise_scale_inv=xt_scale,
            quantizer=quantizers,
            requires_grad=tokens.requires_grad,
        )
        ctx.save_for_backward(row_id_map)
        return output, permuted_probs, row_id_map, row_id_index

    @staticmethod
    def backward(ctx, grad_output, grad_prob, grad_map, grad_index):
        """Backward function."""
        (row_id_map,) = ctx.saved_tensors
        output, prob_output = triton_unpermute_with_mask_map(
            grad_output, row_id_map, grad_prob
        )
        return (
            output.view(ctx.shape),
            prob_output.view(ctx.prob_shape),
            None,
            None,
            None,
            None,
            None,
        )


def mxfp8_permute(
        tokens,
        routing_map,
        tokens_per_expert_cuda_tensor,
        tokens_per_expert_list,
        quantizers,
        cls,
        probs: Optional[torch.Tensor] = None,
):
    permuted_input, permuted_probs, row_id_map, row_id_index = _MXFP8Permute.apply(
        tokens,
        probs,
        routing_map,
        tokens_per_expert_cuda_tensor,
        tokens_per_expert_list,
        quantizers,
        cls,
    )
    return permuted_input, permuted_probs, row_id_map, row_id_index


class _MXFP8Unpermute(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            permuted_tokens,
            row_id_map,
            row_id_index,
            tokens_per_expert,
            splits,
            restore_shape,
            quantizers,
            cls,
    ):
        """Forward function."""
        num_tokens, hidden_size = restore_shape
        num_out_tokens = permuted_tokens.shape[0]
        n_experts = row_id_map.size(1)
        ctx.save_for_backward(row_id_index)
        ctx.input_requires_grad = permuted_tokens.requires_grad
        ctx.num_experts = n_experts
        ctx.restore_shape = restore_shape
        ctx.num_tokens = num_tokens
        ctx.num_out_tokens = num_out_tokens
        ctx.hidden_size = hidden_size
        ctx.tokens_per_expert = tokens_per_expert
        ctx.splits = splits
        ctx.quantizers = quantizers
        ctx.cls = cls

        output, _ = triton_unpermute_with_mask_map(permuted_tokens, row_id_map,
                                                   None)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        (row_id_index,) = ctx.saved_tensors

        quantizers = ctx.quantizers
        x_q, x_scale, xt_q, xt_scale, _ = triton_batch_mxfp8_permute_with_indices(
            grad_output,
            ctx.tokens_per_expert,
            row_id_index,
            ctx.splits,
        )

        output = ctx.cls(
            shape=x_q.shape,
            dtype=grad_output.dtype,
            fp8_dtype=quantizers[0].dtype,
            rowwise_data=x_q,
            rowwise_scale_inv=x_scale,
            columnwise_data=xt_q,
            columnwise_scale_inv=xt_scale,
            quantizer=quantizers,
            requires_grad=False,
        )

        return output, None, None, None, None, None, None, None


def mxfp8_unpermute(
        permuted_tokens: torch.Tensor,
        row_id_map: torch.Tensor,
        row_id_index: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        splits: List,
        restore_shape: torch.Size,
        quantizers,
        cls,
):
    output = _MXFP8Unpermute.apply(
        permuted_tokens,
        row_id_map,
        row_id_index,
        tokens_per_expert,
        splits,
        restore_shape,
        quantizers,
        cls,
    )
    return output


class _MXFP8QuantDispatch(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, tokens, tokens_per_expert_cuda, tokens_per_expert, quantizers,
            cls
    ):
        """Forward function."""
        num_tokens, hidden_dim = tokens.shape

        ctx.num_tokens = num_tokens
        ctx.hidden_dim = hidden_dim
        ctx.shape = tokens.shape
        ctx.cls = cls

        inp_q, inp_scale, inpt_q, inpt_scale = triton_batch_mxfp8_quant(
            tokens, tokens_per_expert_cuda, tokens_per_expert.tolist(),
            output_mode=2
        )

        output = cls(
            shape=inp_q.size(),
            dtype=tokens.dtype,
            fp8_dtype=quantizers[0].dtype,
            rowwise_data=inp_q,
            rowwise_scale_inv=inp_scale,
            columnwise_data=inpt_q,
            columnwise_scale_inv=inpt_scale,
            quantizer=None,
            requires_grad=tokens.requires_grad,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


def mxfp8_quant_dispatch(
        tokens, tokens_per_expert_cuda, tokens_per_expert, quantizers, cls
):
    output = _MXFP8QuantDispatch.apply(
        tokens,
        tokens_per_expert_cuda,
        tokens_per_expert,
        quantizers,
        cls,
    )
    return output


class _MXFP8QuantCombine(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, tokens, tokens_per_expert_cuda, tokens_per_expert, quantizers,
            cls
    ):
        """Forward function."""
        num_tokens, hidden_dim = tokens.shape

        ctx.num_tokens = num_tokens
        ctx.hidden_dim = hidden_dim
        ctx.quantizers = quantizers
        ctx.tokens_per_expert_cuda = tokens_per_expert_cuda
        ctx.tokens_per_expert = tokens_per_expert
        ctx.cls = cls

        return tokens

    @staticmethod
    def backward(ctx, grad_output):
        quantizers = ctx.quantizers
        tokens_per_expert = ctx.tokens_per_expert
        tokens_per_expert_cuda = ctx.tokens_per_expert_cuda

        inp_q, inp_scale, inpt_q, inpt_scale = triton_batch_mxfp8_quant(
            grad_output,
            tokens_per_expert_cuda,
            tokens_per_expert.tolist(),
            output_mode=2,
        )

        grad_output = ctx.cls(
            shape=inp_q.size(),
            dtype=grad_output.dtype,
            fp8_dtype=quantizers[0].dtype,
            rowwise_data=inp_q,
            rowwise_scale_inv=inp_scale,
            columnwise_data=inpt_q,
            columnwise_scale_inv=inpt_scale,
            quantizer=None,
            requires_grad=grad_output.requires_grad,
        )

        return grad_output, None, None, None, None


def mxfp8_quant_combine(
        tokens, tokens_per_expert_cuda, tokens_per_expert, quantizers, cls
):
    output = _MXFP8QuantCombine.apply(
        tokens,
        tokens_per_expert_cuda,
        tokens_per_expert,
        quantizers,
        cls,
    )
    return output



# bf16 forward and bf16 backward
class _SmoothPermute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, probs, routing_map, tokens_per_expert, splits, quantizers, cls):
        """Forward function."""
        num_tokens, hidden_dim = tokens.shape
        
        smooth_scales = torch.stack([x.smooth_scale for x in quantizers], 0)
        # num_out_tokens should including padding tokens
        row_id_map, row_id_indices = triton_make_row_id_map_and_index(routing_map, sum(splits))
        ctx.num_tokens = num_tokens
        ctx.hidden_dim = hidden_dim
        ctx.prob_shape = probs.shape
        ctx.shape = tokens.shape
        ctx.row_id_map = row_id_map
        permuted_input_data, permuted_input_scales, permuted_probs = \
            triton_batch_smooth_permute_with_indices(tokens,
                                                     smooth_scales,
                                                     tokens_per_expert,
                                                     row_id_indices,
                                                     probs=probs,
                                                     reverse=False,
                                                     round_scale=False)
        permuted_input = cls(
            shape=permuted_input_data.shape,
            dtype=tokens.dtype,
            fp8_dtype=quantizers[0].dtype,
            rowwise_data=permuted_input_data,
            rowwise_scale_inv=permuted_input_scales,
            columnwise_data=None,
            columnwise_scale_inv=smooth_scales,
            quantizer=quantizers,
            requires_grad=tokens.requires_grad,
        )
        return permuted_input, permuted_probs, row_id_map, row_id_indices

    @staticmethod
    def backward(ctx, grad_output, grad_prob, grad_map, grad_indices):
        """Backward function."""
        output, prob_output = triton_unpermute_with_mask_map(grad_output, ctx.row_id_map, grad_prob)
        return output.view(ctx.shape), prob_output.view(ctx.prob_shape), None, None, None, None, None


def smooth_permute(
    tokens,
    routing_map,
    tokens_per_expert,
    splits,
    quantizers,
    cls,
    probs: Optional[torch.Tensor] = None

):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.
    When drop_and_pad=True, in routing_map, the number of non-zeros in each column equals to
    expert capacity. This function exploits this feature to use ops that support cuda graph.
    Args:
        tokens (torch.Tensor): The fp8 input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        num_out_tokens (int, optional): The number of output tokens. If None, it's set to
                                        the number of input tokens.
        fused (bool, optional): Whether use the fused permute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.
                                       If set to true, routing_map has a fixed number of non-zeros
                                       in each column.
    """
    (permuted_input,
     permuted_probs,
     row_id_map,
     row_id_indices) = _SmoothPermute.apply(tokens,
                                                 probs,
                                                 routing_map,
                                                 tokens_per_expert,
                                                 splits,
                                                 quantizers,
                                                 cls)
    return permuted_input, permuted_probs, row_id_map, row_id_indices


# bf16 forward and bf16 backward
class _SmoothUnpermute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, permuted_tokens, row_id_map, row_id_indices, token_count_per_expert, splits, restore_shape, quantizers, cls):
        """Forward function."""
        num_tokens, hidden_size = restore_shape
        num_out_tokens = permuted_tokens.shape[0]
        n_experts = row_id_map.size(1)
        ctx.save_for_backward(row_id_map, row_id_indices, token_count_per_expert)
        ctx.num_experts = n_experts
        ctx.restore_shape = restore_shape
        ctx.num_tokens = num_tokens
        ctx.num_out_tokens = num_out_tokens
        ctx.hidden_size = hidden_size
        ctx.splits = splits
        ctx.quantizers = quantizers
        ctx.cls = cls
        output, _ = triton_unpermute_with_mask_map(permuted_tokens, row_id_map, None)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function. """
        row_id_map, row_id_indices, token_count_per_expert = ctx.saved_tensors
        quantizers = ctx.quantizers
        # TODO(nanxiao): smooth_scale_inv will updated in every forward, it will cause error with PP
        grad_smooth_scales = torch.stack([x.smooth_scale_inv for x in quantizers], 0)
        transpose_grad_smooth_scales = [x.transpose_smooth_scale_inv for x in quantizers 
                                            if x.transpose_smooth_scale_inv.numel() > 0]
        if len(transpose_grad_smooth_scales) > 0:
            transpose_grad_smooth_scales = torch.cat(transpose_grad_smooth_scales, 0)
        else:
            transpose_grad_smooth_scales = None
        round_scale = quantizers[0].force_pow_2_scales

        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        permuted_grad_data, permuted_grad_scales, _ = triton_batch_smooth_permute_with_indices(
                                grad_output,
                                grad_smooth_scales,
                                token_count_per_expert,
                                row_id_indices,
                                probs=None,
                                reverse=True,
                                round_scale=round_scale)
        permuted_grad_data_t, permuted_grad_scales_t = triton_batch_transpose_smooth_permute_with_indices(
                                                grad_output, 
                                                transpose_grad_smooth_scales, 
                                                row_id_indices, 
                                                token_count_per_expert, 
                                                ctx.splits, 
                                                round_scale=round_scale)
    
        input_grad = ctx.cls(
            shape=permuted_grad_data.shape,
            dtype=grad_output.dtype,
            fp8_dtype=quantizers[0].dtype,
            rowwise_data=permuted_grad_data,
            rowwise_scale_inv=permuted_grad_scales,
            columnwise_data=permuted_grad_data_t,
            columnwise_scale_inv=permuted_grad_scales_t,
            quantizer=quantizers,
            requires_grad=False
        )
        return input_grad, None, None, None, None, None, None, None


def smooth_unpermute(
    permuted_tokens: torch.Tensor,
    row_id_map: torch.Tensor,
    row_id_indices: torch.Tensor,
    token_count_per_expert: torch.Tensor, 
    splits: List[int],
    restore_shape: torch.Size,
    quantizers: List,
    cls
):
    output = _SmoothUnpermute.apply(permuted_tokens,
                                         row_id_map,
                                         row_id_indices,
                                         token_count_per_expert,
                                         splits,
                                         restore_shape,
                                         quantizers,
                                         cls)
    return output


# fp8 forward and bf16 backward
class _SmoothFusedPermute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, probs, routing_map, tokens_per_expert, splits, cls):
        """Forward function."""
        num_tokens, hidden_dim = tokens._rowwise_data.shape
        # num_experts = routing_map.shape[1]
        counts = routing_map.sum(-1)
        
        # num_out_tokens should including padding tokens
        row_id_map, row_id_indices = triton_make_row_id_map_and_index(routing_map, sum(splits))
        ctx.num_tokens = num_tokens
        ctx.hidden_dim = hidden_dim
        ctx.prob_shape = probs.shape
        ctx.shape = tokens.shape
        ctx.counts = counts
        ctx.row_id_map = row_id_map
        permuted_input_data,permuted_input_scales,permuted_probs = triton_permute_with_mask_map(
            tokens._rowwise_data,
            tokens._rowwise_scale_inv,
            probs,
            row_id_map,
            sum(splits))
        permuted_input = cls(
            shape=permuted_input_data.shape,
            dtype=tokens.dtype,
            fp8_dtype=tokens._fp8_dtype,
            rowwise_data=permuted_input_data,
            rowwise_scale_inv=permuted_input_scales,
            columnwise_data=None,
            columnwise_scale_inv=tokens._columnwise_scale_inv,
            quantizer=tokens._quantizer,
            requires_grad=tokens.requires_grad,
        )
        return permuted_input, permuted_probs, row_id_map, row_id_indices

    @staticmethod
    def backward(ctx, grad_output, grad_prob, grad_map, grad_indices):
        """Backward function."""
        output, prob_output = triton_unpermute_with_mask_map(grad_output, ctx.row_id_map, grad_prob)
        return output.view(ctx.shape), prob_output.view(ctx.prob_shape), None, None, None, None


def smooth_fused_permute(
    tokens,
    routing_map,
    probs: Optional[torch.Tensor] = None,
    tokens_per_expert: Optional[torch.Tensor] = None,
    splits: Optional[List[int]] = None,
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.
    When drop_and_pad=True, in routing_map, the number of non-zeros in each column equals to
    expert capacity. This function exploits this feature to use ops that support cuda graph.
    Args:
        tokens (torch.Tensor): The fp8 input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        num_out_tokens (int, optional): The number of output tokens. If None, it's set to
                                        the number of input tokens.
        fused (bool, optional): Whether use the fused permute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.
                                       If set to true, routing_map has a fixed number of non-zeros
                                       in each column.
    """
    (permuted_input,
     permuted_probs,
     row_id_map,
     row_id_indices) = _SmoothFusedPermute.apply(tokens,
                                                 probs,
                                                 routing_map,
                                                 tokens_per_expert,
                                                 splits)
    return permuted_input, permuted_probs, row_id_map, row_id_indices


# bf16 forward and fp8 backward
# fused unpermute and quant
class _SmoothFusedUnpermute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, permuted_tokens, row_id_map, row_id_indices, org_smooth_scale, token_count_per_expert, splits, quantizers, restore_shape, cls):
        """Forward function."""
        num_tokens, hidden_size = restore_shape
        num_out_tokens = permuted_tokens.shape[0]
        n_experts = row_id_map.size(1)
        ctx.save_for_backward(row_id_map, row_id_indices, org_smooth_scale, token_count_per_expert)
        ctx.input_requires_grad = permuted_tokens.requires_grad
        ctx.num_experts = n_experts
        ctx.restore_shape = restore_shape
        ctx.num_tokens = num_tokens
        ctx.num_out_tokens = num_out_tokens
        ctx.hidden_size = hidden_size
        ctx.splits = splits
        ctx.quantizers = quantizers
        ctx.cls = cls
        output, _ = triton_unpermute_with_mask_map(permuted_tokens, row_id_map, None)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function. grad_output is smooth quantized"""
        row_id_map, row_id_indices, org_smooth_scale, token_count_per_expert = ctx.saved_tensors
        quantizers = grad_output._quantizer
        # smooth_scale_inv will updated in every forward, it will cause error with PP
        smooth_scales = torch.stack([x.smooth_scale_inv for x in quantizers], 0)
        transpose_smooth_scales = torch.cat([x.transpose_smooth_scale_inv for x in quantizers], 0)
        #  todo(nanxiao): smooth
        round_scale = quantizers[0].force_pow_2_scales

        update_smooth_scales = smooth_scales * org_smooth_scale
        for i, quantizer in enumerate(quantizers):
            quantizer.smooth_scale_inv = update_smooth_scales[i]
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        permuted_grad_data, permuted_grad_scales = triton_batch_smooth_fused_permute_with_indices(
                                grad_output._rowwise_data,
                                grad_output._rowwise_scale_inv,
                                update_smooth_scales,
                                token_count_per_expert,
                                row_id_indices,
                                reverse=True,
                                round_scale=round_scale)
        permuted_grad_data_t, permuted_grad_scales_t = triton_batch_transpose_smooth_fused_permute_with_indices(
                                                grad_output._rowwise_data, 
                                                grad_output._rowwise_scale_inv, 
                                                grad_smooth_scale, 
                                                transpose_smooth_scales, 
                                                row_id_indices, 
                                                token_count_per_expert, 
                                                ctx.splits, 
                                                round_scale=round_scale)
    

        input_grad = ctx.cls(
            shape=permuted_grad_data.shape,
            dtype=grad_output.dtype,
            fp8_dtype=quantizers[0].dtype,
            rowwise_data=permuted_grad_data,
            rowwise_scale_inv=permuted_grad_scales,
            columnwise_data=permuted_grad_data_t,
            columnwise_scale_inv=permuted_grad_scales_t,
            quantizer=quantizers,
            requires_grad=ctx.input_requires_grad
        )
        return input_grad, None, None, None, None, None, None, None, None


def smooth_fused_unpermute(
    permuted_tokens: torch.Tensor,
    row_id_map: torch.Tensor,
    row_id_indices: torch.Tensor,
    org_smooth_scale: torch.Tensor,
    token_count_per_expert: torch.Tensor, 
    splits: List[int],
    quantizers: List,
    restore_shape: torch.Size,
    cls
):
    output = _SmoothFusedUnpermute.apply(permuted_tokens,
                                         row_id_map,
                                         row_id_indices,
                                         org_smooth_scale,
                                         token_count_per_expert,
                                         splits,
                                         quantizers,
                                         restore_shape,
                                         cls)
    return output
