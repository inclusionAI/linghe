from typing import Optional, List

import torch

from linghe.utils.gather import (
    triton_permute_with_mask_map,
    triton_make_row_id_map,
    triton_make_row_id_map_and_index,
    triton_batch_block_pad_permute_with_indices,
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
    ):
        """Forward function."""
        num_tokens, hidden_dim = tokens.shape

        row_id_map = triton_make_row_id_map(routing_map, multiple_of=16)
        num_out_tokens = sum([(x + 15) // 16 * 16 for x in tokens_per_expert_list])

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
        )


def padded_permute(
    tokens,
    routing_map,
    tokens_per_expert_cuda_tensor,
    tokens_per_expert_list,
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

    permuted_input, permuted_probs, row_id_map = _PaddedPermute.apply(
        tokens,
        probs,
        routing_map,
        tokens_per_expert_cuda_tensor,
        tokens_per_expert_list,
    )
    return permuted_input, permuted_probs, row_id_map


class _PaddedUnpermute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, permuted_tokens, row_id_map, tokens_per_expert, restore_shape):
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

        output, _ = triton_unpermute_with_mask_map(permuted_tokens, row_id_map, None)
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

        num_out_tokens = sum([(x + 15) // 16 * 16 for x in tokens_per_expert_list])
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

        output, _ = triton_unpermute_with_mask_map(permuted_tokens, row_id_map, None)
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
