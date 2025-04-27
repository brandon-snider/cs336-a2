import math

import torch
import triton
import triton.language as tl
from einops import rearrange


@torch.compile
def _flash_bwd_inner(dO, L, Q, K, V, O, S, scale, is_causal):
    """Compiled inner loop for flash attention backward pass."""
    D = (dO * O).sum(dim=-1)
    S_scores = Q @ K.transpose(-2, -1) * scale

    if is_causal:
        mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=S_scores.device))
        S_scores = S_scores.masked_fill(~mask, float("-inf"))

    P = torch.exp(S_scores - L.unsqueeze(-1))
    if is_causal:
        mask = torch.tril(torch.ones(S, S, dtype=P.dtype, device=P.device))
        P = P * mask

    dV = P.transpose(-2, -1) @ dO
    dP = dO @ V.transpose(-2, -1)
    dS = P * (dP - D.unsqueeze(-1))
    dQ = (dS @ K) * scale
    dK = (dS.transpose(-2, -1) @ Q) * scale

    return dQ, dK, dV


class FlashTriton(torch.autograd.Function):
    """torch.autograd.Function that calls the triton kernel.

    Notes:
    - Inputs are (*batch_dims, seq_len, d).
    - Batch-like dims are flattened so the kernel launch grid is *2‑D*: (Tq, B)
      where Tq is the number of query tiles and B is the product of the batch dims.
    - The kernel writes its tile of *O* and *L* into pre‑allocated output buffers.
    """

    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
        # Flatten batch dims & derive launch parameters
        *batch_dims, seq_len, d = Q.shape
        B = math.prod(batch_dims)  # total launch grid height

        Q_T = min(128, seq_len)  # query‑tile rows
        K_T = min(128, K.shape[-2])  # key‑tile cols
        Tq = triton.cdiv(seq_len, Q_T)  # number of query tiles

        # Flatten batch-like dims
        Qf = rearrange(Q, "... s h -> (...) s h")
        Kf = rearrange(K, "... s h -> (...) s h")
        Vf = rearrange(V, "... s h -> (...) s h")

        # Output / logsumexp buffers (keep original un‑flattened view for return)
        O = torch.empty_like(Q)
        L = torch.empty((*batch_dims, seq_len), dtype=torch.float32, device=Q.device)
        Of = rearrange(O, "... s h -> (...) s h")
        Lf = rearrange(L, "... s -> (...) s")

        scale = 1.0 / math.sqrt(d)

        # Launch grid (Tq × B)
        flash_fwd_kernel[(Tq, B)](
            Qf, Kf, Vf, Of, Lf,
            *Qf.stride(), *Kf.stride(), *Vf.stride(), *Of.stride(), *Lf.stride(),
            N_QUERIES=seq_len, N_KEYS=K.shape[-2], scale=scale,
            D=d, Q_TILE_SIZE=Q_T, K_TILE_SIZE=K_T, is_causal=is_causal,
        )  # fmt: skip

        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO: torch.Tensor):
        """
        Implements the backward pass for the FlashAttention-2 forward pass.

        Args:
          - dO: (*batch_dims, S, d) gradients w.r.t. forward output O.

        Returns:
          - dQ: gradients w.r.t. Q
          - dK: gradients w.r.t. K
          - dV: gradients w.r.t. V
          - None: is_causal has no grad
        """
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal

        S, d = Q.shape[-2:]
        scale = 1.0 / math.sqrt(d)

        dQ, dK, dV = _flash_bwd_inner(dO, L, Q, K, V, O, S, scale, is_causal)

        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):  # fmt: skip
    """Compute one query tile of O and L for one batch.

    Launch grid is (Tq, B), so program ID gives us query tile index and batch index.
    The kernel loops over all key tiles (inner loop).
    """

    q_tile_idx = tl.program_id(0)
    b_idx = tl.program_id(1)

    Q_bp = tl.make_block_ptr(
        Q_ptr + b_idx * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_bp = tl.make_block_ptr(
        K_ptr + b_idx * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_bp = tl.make_block_ptr(
        V_ptr + b_idx * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_bp = tl.make_block_ptr(
        O_ptr + b_idx * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(q_tile_idx * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_bp = tl.make_block_ptr(
        L_ptr + b_idx * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(q_tile_idx * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    # Load query tile into SRAM
    Q_tile = tl.load(Q_bp, boundary_check=(0, 1), padding_option="zero")

    # Initialise online‑softmax state
    acc_o = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    # Pre‑compute query indices once if causal
    if is_causal:
        q_global_start = q_tile_idx * Q_TILE_SIZE
        q_idx = q_global_start + tl.arange(0, Q_TILE_SIZE)

    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)  # number of key tiles

    for k_tile_idx in range(0, Tk):
        # Load K / V tile with boundary checks
        K_tile = tl.load(K_bp, boundary_check=(0, 1), padding_option="zero")
        V_tile = tl.load(V_bp, boundary_check=(0, 1), padding_option="zero")

        # Compute raw scores S
        S = tl.dot(Q_tile, tl.trans(K_tile)) * scale  # (Q_T, K_T)

        if is_causal:
            k_global_start = k_tile_idx * K_TILE_SIZE
            k_idx = k_global_start + tl.arange(0, K_TILE_SIZE)
            mask = q_idx[:, None] < k_idx[None, :]
            S = tl.where(mask, -1e6, S)

        # Online softmax update
        row_max = tl.max(S, axis=1)  # tile max per row
        m_i_new = tl.maximum(m_i, row_max)

        exp_S = tl.exp(S - m_i_new[:, None])
        exp_mdiff = tl.exp(m_i - m_i_new)
        l_i_new = exp_mdiff * l_i + tl.sum(exp_S, axis=1)

        # Accumulate un‑normalised output
        acc_o = exp_mdiff[:, None] * acc_o + tl.dot(exp_S.to(V_tile.dtype), V_tile)

        # Advance block pointers to next key tile
        K_bp = K_bp.advance((K_TILE_SIZE, 0))
        V_bp = V_bp.advance((K_TILE_SIZE, 0))
        m_i, l_i = m_i_new, l_i_new

    # Final normalisation & store
    acc_o = (acc_o / l_i[:, None]).to(O_bp.type.element_ty)  # cast to fp16/bf16
    L_out = (m_i + tl.log(l_i)).to(tl.float32)

    tl.store(O_bp, acc_o, boundary_check=(0, 1))
    tl.store(L_bp, L_out, boundary_check=(0,))
