import math
import triton
import triton.language as tl
import torch
from einops import rearrange


class FlashTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        *batch_dims, seq_len, d = Q.shape
        B = math.prod(batch_dims)

        Q_T = min(128, seq_len)
        K_T = min(128, K.shape[-2])
        Tq = triton.cdiv(seq_len, Q_T)

        # Handle arbitrary batch dimensions
        Qf = rearrange(Q, "... s h -> (...) s h")
        Kf = rearrange(K, "... s h -> (...) s h")
        Vf = rearrange(V, "... s h -> (...) s h")

        O = torch.empty_like(Q)
        L = torch.empty((*batch_dims, seq_len), dtype=torch.float32, device=Q.device)

        Of = rearrange(O, "... s h -> (...) s h")
        Lf = rearrange(L, "... s -> (...) s")

        scale = 1.0 / math.sqrt(d)

        flash_fwd_kernel[(Tq, B)](
            Qf, Kf, Vf,
            Of, Lf,
            *Qf.stride(),
            *Kf.stride(),
            *Vf.stride(),
            *Of.stride(),
            *Lf.stride(),
            N_QUERIES=seq_len, N_KEYS=K.shape[-2],
            scale=scale,
            D=d,
            Q_TILE_SIZE=Q_T,
            K_TILE_SIZE=K_T,
            is_causal=is_causal,
        )  # fmt: skip

        ctx.save_for_backward(L, Q, K, V, O)
        return O


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
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):  # fmt: skip
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    # Complete the kernel implementation as follows...
    # - Create the rest of the required block pointers
    # - Load the current Q-tile
    # - Initialize on-chip accumulators
    # - Iterate over the keys with `for j in range(Tk)`:
    #   - load K(j) / V(j) (block pointers, boundary-checked)
    #   - compute S = Q · Kᵀ * scale
    #   - update running m_i, l_i (online softmax)
    #   - accumulate unnormalised acc_o += P̃ @ V(j) (use tl.dot, keep acc in fp32)
    #   - Q, K, V stay in registers/shared SRAM for the tile
    # - After the loop, finish the normalization
    #   - acc_o = (acc_o / l_i[:, None]).to(out_dtype)
    # - Write the tile of `O` and `L = m_i + log(l_i)` back to global memory
    # (correctly handle boilerplate — boundary advances, dtypes, casts, etc. — and remember to use FP32 accumulators)
