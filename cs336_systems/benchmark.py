#!/usr/bin/env python
"""
`timeit`-based benchmarking.

Usage:
- Forward only (inference mode):
    python -m cs336_systems.benchmark --size sm --no-backward
- Forward + backward (training mode):
    python -m cs336_systems.benchmark --size sm

Flags:
  --size: sm, md, lg, xl, 2.7b
  --d_model: int
  --d_ff: int
  --d_layers: int
  --num_heads: int
  --batch_size: int
  --seq_len: int
  --vocab_size: int
  --dtype: fp32, bf16
  --warmup: int
  --steps: int
"""

import argparse
import sys
import timeit
import torch

from cs336_basics.model import BasicsTransformerLM as Transformer
from cs336_basics.nn_utils import cross_entropy

_PRESETS: dict[str, dict[str, int]] = {
    "sm": {"d_model": 768, "d_ff": 3072, "d_layers": 12, "num_heads": 12},
    "md": {"d_model": 1024, "d_ff": 4096, "d_layers": 24, "num_heads": 16},
    "lg": {"d_model": 1280, "d_ff": 5120, "d_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "d_layers": 48, "num_heads": 25},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "d_layers": 32, "num_heads": 32},
}


def _parse_args() -> argparse.Namespace:
    """Build an ``argparse`` interface and return parsed arguments."""
    p = argparse.ArgumentParser()

    size_grp = p.add_mutually_exclusive_group(required=True)
    size_grp.add_argument("--size", choices=_PRESETS.keys())
    size_grp.add_argument("--d_model", type=int)

    p.add_argument("--d_ff", type=int)
    p.add_argument("--num-layers", dest="d_layers", type=int)
    p.add_argument("--num-heads", dest="num_heads", type=int)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=10_000)

    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)

    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")

    bw_grp = p.add_mutually_exclusive_group()
    bw_grp.add_argument("--no-backward", dest="do_backward", action="store_false")
    bw_grp.add_argument("--forward-backward", dest="do_backward", action="store_true")
    p.set_defaults(do_backward=True)

    return p.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This benchmark requires a CUDA-capable GPU.")
        sys.exit(1)

    args = _parse_args()

    if args.size:
        cfg = _PRESETS[args.size]
    else:
        required = ("d_model", "d_ff", "d_layers", "num_heads")
        if any(getattr(args, k) is None for k in required):
            raise ValueError("Must supply all custom h‑params when --size is omitted")
        cfg = dict(d_model=args.d_model, d_ff=args.d_ff, d_layers=args.d_layers, num_heads=args.num_heads)

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    torch.set_float32_matmul_precision("high")

    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.seq_len,
        d_model=cfg["d_model"],
        num_layers=cfg["d_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=10000.0,
    ).to(device="cuda", dtype=dtype)
    model.train(args.do_backward)

    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device="cuda")
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device="cuda")

    fw_samples, bw_samples = [], []
    for i in range(args.warmup + args.steps):
        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        logits = model(inputs)
        torch.cuda.synchronize()
        dt_fw = timeit.default_timer() - t0

        dt_bw = 0.0
        if args.do_backward:
            loss = cross_entropy(logits, targets)
            t1 = timeit.default_timer()
            loss.backward()
            torch.cuda.synchronize()
            dt_bw = timeit.default_timer() - t1

        if i >= args.warmup:
            fw_samples.append(dt_fw)
            bw_samples.append(dt_bw)
        if args.do_backward:
            model.zero_grad(set_to_none=True)

    fw_t = torch.tensor(fw_samples, device="cuda")
    bw_t = torch.tensor(bw_samples, device="cuda")

    mode_str = "forward-only" if not args.do_backward else "forward + backward"
    print(
        f"Timings for {args.size} model, batch size {args.batch_size}, seq len {args.seq_len}, {mode_str} over {args.steps} steps:"
    )
    print(f"Forward:  {fw_t.mean() * 1e3:.3f} ± {fw_t.std() * 1e3:.3f} ms")

    if args.do_backward:
        total = fw_t + bw_t
        print(f"Backward: {bw_t.mean() * 1e3:.3f} ± {bw_t.std() * 1e3:.3f} ms")
        print(f"Total:    {total.mean() * 1e3:.3f} ± {total.std() * 1e3:.3f} ms")


if __name__ == "__main__":
    main()
