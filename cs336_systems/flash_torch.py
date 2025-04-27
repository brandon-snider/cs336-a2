"""
This file contains the implementation of the FlashAttention-2 forward pass in pure PyTorch.

Test: uv run pytest -k test_flash_forward_pass_pytorch
"""

import torch
from einops import rearrange, einsum
import math

Q_TILE_SIZE = 128
K_TILE_SIZE = 128


class FlashTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
        """
        Implements FlashAttention-2 forward pass for a single head in pure PyTorch.
        Handles arbitrary batch dimensions.

        Args:
          - Q: Query tensor of shape (*batch_dims, seq_len, head_dim)
          - K: Key tensor of shape (*batch_dims, seq_len, head_dim)
          - V: Value tensor of shape (*batch_dims, seq_len, head_dim)
          - is_causal: Whether to apply causal masking

        Produces the output tensor O of shape (*batch_dims, seq_len, head_dim), and the logsumexp tensor L
        of shape (*batch_dims, seq_len). Saves L, Q, K, V, O to ctx for the backward pass, and returns O.

        Returns:
          - O: Output tensor of shape (*batch_dims, seq_len, head_dim)
        """
        seq_len, head_dim = Q.shape[-2:]
        original_shape = Q.shape
        Q_orig, K_orig, V_orig = Q, K, V

        q_tile_size = min(Q_TILE_SIZE, seq_len)
        k_tile_size = min(K_TILE_SIZE, seq_len)

        # Flatten batch-like dimensions and tile the query, key, and value tensors
        q_tiles = rearrange(Q, "... (t_q q_tile_size) h -> (...) t_q q_tile_size h", q_tile_size=q_tile_size)
        k_tiles = rearrange(K, "... (t_k k_tile_size) h -> (...) t_k k_tile_size h", k_tile_size=k_tile_size)
        v_tiles = rearrange(V, "... (t_v v_tile_size) h -> (...) t_v v_tile_size h", v_tile_size=k_tile_size)

        batch_size = q_tiles.shape[0]

        # Initialize O and L with the correct (i.e. original, unflattened) shapes
        O = torch.zeros((*original_shape[:-2], seq_len, head_dim), device=Q.device)
        L = torch.zeros((*original_shape[:-2], seq_len), device=Q.device)

        # Create tiled views of O and L that we can update in place
        O_tiles = rearrange(O, "... (t_q q_tile_size) h -> (...) t_q q_tile_size h", q_tile_size=q_tile_size)
        L_tiles = rearrange(L, "... (t_q q_tile_size) -> (...) t_q q_tile_size", q_tile_size=q_tile_size)

        num_q_tiles = q_tiles.shape[1]
        num_k_tiles = k_tiles.shape[1]

        scale = 1 / math.sqrt(head_dim)

        for i in range(num_q_tiles):
            queries = q_tiles[:, i]

            out_tile = torch.zeros_like(queries)  # Shape: (batch_size, q_tile_size, head_dim)
            row_maxes_so_far = torch.full((batch_size, q_tile_size), -torch.inf, device=Q.device)
            logsumexps_so_far = torch.zeros((batch_size, q_tile_size), device=Q.device)

            for j in range(num_k_tiles):
                keys = k_tiles[:, j]
                values = v_tiles[:, j]

                scores_tile = einsum(queries, keys, "b q h, b k h -> b q k") * scale

                current_tile_row_maxes = torch.max(scores_tile, dim=-1).values
                row_maxes_prev = row_maxes_so_far.clone()
                row_maxes_so_far = torch.maximum(row_maxes_so_far, current_tile_row_maxes)

                exp_scores_tile_hat = torch.exp(scores_tile - row_maxes_so_far[..., None])
                exp_max_diff = torch.exp(row_maxes_prev - row_maxes_so_far)

                logsumexps_so_far = exp_max_diff * logsumexps_so_far + torch.sum(exp_scores_tile_hat, dim=-1)

                weighted_values_tile = einsum(exp_scores_tile_hat, values, "b q k, b k h -> b q h")
                out_tile = torch.diag_embed(exp_max_diff) @ out_tile + weighted_values_tile

            # Normalise and write results back to the untiled view
            O_tiles[:, i] = torch.diag_embed(1.0 / logsumexps_so_far) @ out_tile
            L_tiles[:, i] = row_maxes_so_far + torch.log(logsumexps_so_far)

        ctx.save_for_backward(L, Q_orig, K_orig, V_orig, O)

        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        """
        Implements the backward pass for the FlashAttention-2 forward pass.

        Args:
          - dO: Gradient of the output tensor w.r.t. the output of the forward pass

        Returns:
          - dQ: Gradient of the query tensor w.r.t. the loss
          - dK: Gradient of the key tensor w.r.t. the loss
          - dV: Gradient of the value tensor w.r.t. the loss
        """
        L, Q, K, V, O = ctx.saved_tensors
        head_dim = Q.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)

        D = torch.sum(O * dO, dim=-1, keepdim=True)

        S = einsum(Q, K, "... q d, ... k d -> ... q k") * scale
        P = torch.exp(S - L.unsqueeze(-1))

        dV = einsum(P, dO, "... q k, ... q h -> ... k h")
        dP = einsum(dO, V, "... q h, ... k h -> ... q k")
        dS = P * (dP - D)
        dQ = einsum(dS, K, "... q k, ... k d -> ... q d") * scale
        dK = einsum(dS, Q, "... q k, ... q d -> ... k d") * scale

        return dQ, dK, dV, None
