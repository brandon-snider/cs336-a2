"""
Benchmarking script for vanilla PyTorch attention vs. Triton FlashAttention-2

run with:  uv run -m cs336_systems.flash_triton_benchmark  (≈ 2–3 min on a single H100)
"""

import itertools
import time
import torch
import triton.testing as ttesting
from tabulate import tabulate  # pip install tabulate
from cs336_systems.flash_triton import FlashTriton
from cs336_basics.model import scaled_dot_product_attention


def mk_causal_mask(S: int, device):
    """Pre-compute a [S,S] lower-triangular causal mask once per sequence length."""
    return torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))


def bench_forward(fn, *tensors):
    return ttesting.do_bench(lambda: fn(*tensors)) * 1e3  # ms


def bench_backward(fwd_fn, *inps):
    """
    Measure *only* backward: run fwd once outside the timed region to get the ctx,
    then backward inside `do_bench`.  This matches the spec's notion of "backward".
    """
    for x in inps:
        x.requires_grad_(True)
    out = fwd_fn(*inps)
    grad_out = torch.randn_like(out)

    def _backward():
        out.backward(grad_out, retain_graph=True)  # only bwd is timed
        for x in inps:
            x.grad = None  # clear for next iter

    return ttesting.do_bench(_backward) * 1e3


def bench_total(fwd_fn, *inps):
    """Full forward + backward in one call (loss = sum(O))."""
    for x in inps:
        x.requires_grad_(True)

    def _step():
        out = fwd_fn(*inps)
        out.sum().backward()
        for x in inps:
            x.grad = None

    return ttesting.do_bench(_step) * 1e3


def run_one(seq_len: int, d_model: int, dtype: torch.dtype, device: str):
    """
    Returns results for PyTorch and Triton separately.
    Each result is either a float (ms) or the string "OOM".
    (py_fwd, py_bwd, py_tot, tr_fwd, tr_bwd, tr_tot)
    """
    Q = torch.randn(seq_len, d_model, dtype=dtype, device=device)
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    py_fwd, py_bwd, py_tot = "OOM", "OOM", "OOM"
    tr_fwd, tr_bwd, tr_tot = "OOM", "OOM", "OOM"

    # ---- PyTorch reference ----
    try:
        mask = mk_causal_mask(seq_len, device)

        def _torch_fwd(Q_, K_, V_):
            return scaled_dot_product_attention(Q_, K_, V_, mask=mask)

        py_fwd = bench_forward(_torch_fwd, Q, K, V)
        py_bwd = bench_backward(_torch_fwd, Q.clone(), K.clone(), V.clone())
        py_tot = bench_total(_torch_fwd, Q.clone(), K.clone(), V.clone())
        torch.cuda.synchronize()  # Ensure operations finish before timing/clearing cache
    except torch.cuda.OutOfMemoryError:
        print(f"OOM (PyTorch) S={seq_len:<6}  d={d_model:<4}  {dtype}")
    finally:
        torch.cuda.empty_cache()

    # ---- Triton FlashAttention-2 ----
    try:

        def _flash_fwd(Q_, K_, V_):
            return FlashTriton.apply(Q_, K_, V_, True)

        tr_fwd = bench_forward(_flash_fwd, Q, K, V)
        tr_bwd = bench_backward(_flash_fwd, Q.clone(), K.clone(), V.clone())
        tr_tot = bench_total(_flash_fwd, Q.clone(), K.clone(), V.clone())
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        print(f"OOM (Triton)  S={seq_len:<6}  d={d_model:<4}  {dtype}")
    finally:
        torch.cuda.empty_cache()

    return py_fwd, py_bwd, py_tot, tr_fwd, tr_bwd, tr_tot


def main():
    device = "cuda"

    seq_lens = [2**p for p in range(7, 17)]  # 128 … 65 536
    d_models = [2**p for p in range(4, 8)]  #   16 …    128
    dtypes = [torch.bfloat16, torch.float32]
    dtype_n = {torch.bfloat16: "bf16", torch.float32: "fp32"}

    rows = []
    for S, D, dt in itertools.product(seq_lens, d_models, dtypes):
        # BF16 matmuls need shapes multiple of 16 on Ampere/Hopper – skip invalid combos
        if dt == torch.bfloat16 and D % 8:  # 8 is OK on Hopper, but keep simple
            continue

        print(f"Running S={S:<6}  d={D:<4}  {dtype_n[dt]}...")

        torch.manual_seed(0)
        t0 = time.time()

        pf, pb, pt, tf, tb, tt = run_one(S, D, dt, device)

        # Format results or keep "OOM"
        pf_s = f"{pf:6.1f}" if isinstance(pf, float) else pf
        pb_s = f"{pb:6.1f}" if isinstance(pb, float) else pb
        pt_s = f"{pt:6.1f}" if isinstance(pt, float) else pt
        tf_s = f"{tf:6.1f}" if isinstance(tf, float) else tf
        tb_s = f"{tb:6.1f}" if isinstance(tb, float) else tb
        tt_s = f"{tt:6.1f}" if isinstance(tt, float) else tt

        # Calculate speedup only if both ran successfully
        speedup_fwd_s = "N/A"
        if isinstance(pf, float) and isinstance(tf, float) and tf > 0:
            speedup_fwd_s = f"{(pf / tf):5.1f}×"
        elif isinstance(pf, float) and isinstance(tf, float) and tf == 0:
            speedup_fwd_s = "Inf"

        speedup_bwd_s = "N/A"
        if isinstance(pb, float) and isinstance(tb, float) and tb > 0:
            speedup_bwd_s = f"{(pb / tb):5.1f}×"
        elif isinstance(pb, float) and isinstance(tb, float) and tb == 0:
            speedup_bwd_s = "Inf"

        speedup_tot_s = "N/A"
        if isinstance(pt, float) and isinstance(tt, float) and tt > 0:
            speedup_tot_s = f"{(pt / tt):5.1f}×"
        elif isinstance(pt, float) and isinstance(tt, float) and tt == 0:
            speedup_tot_s = "Inf"

        rows.append(
            [
                S,
                D,
                dtype_n[dt],
                pf_s,
                pb_s,
                pt_s,
                tf_s,
                tb_s,
                tt_s,
                speedup_fwd_s,
                speedup_bwd_s,
                speedup_tot_s,
            ]
        )
        print(
            f"done  S={S:<6}  d={D:<4}  {dtype_n[dt]}   flash speed-up (f/b/t): {speedup_fwd_s}/{speedup_bwd_s}/{speedup_tot_s}  ({time.time() - t0:4.1f}s)"
        )

    header = [
        "seq",
        "d_model",
        "dtype",
        "Py fwd",
        "Py bwd",
        "Py tot",
        "Flash fwd",
        "Flash bwd",
        "Flash tot",
        "× fwd",
        "× bwd",
        "× tot",
    ]
    print("\n" + tabulate(rows, headers=header, tablefmt="github"))


if __name__ == "__main__":
    main()
