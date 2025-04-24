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
  --warmup: int
  --steps: int
  --mixed: bool
  --compile: bool
  --mode: forward, forward_backward, train
  --memory: bool
"""

import argparse
from contextlib import nullcontext
import sys
import timeit
import torch

from cs336_basics.optimizer import AdamW
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

    p.add_argument("--mixed", action="store_true")
    p.add_argument("--compile", action="store_true")

    p.add_argument("--mode", choices=["forward", "forward_backward", "train"], default="forward_backward")

    p.add_argument("--memory", action="store_true")

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

    # torch.set_float32_matmul_precision("high")

    model = Transformer(
        vocab_size=args.vocab_size,
        context_length=args.seq_len,
        d_model=cfg["d_model"],
        num_layers=cfg["d_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        rope_theta=10000.0,
    ).to(device="cuda")
    if args.compile:
        model = torch.compile(model)
        print("Compiled model")

    model.train(args.mode in ("forward_backward", "train"))
    optimizer = AdamW(model.parameters(), lr=1e-3)

    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device="cuda")
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device="cuda")

    if args.mixed:
        print("Using mixed precision")

    f_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.mixed else nullcontext()

    fw_samples, bw_samples, opt_samples = [], [], []

    with f_ctx:
        for i in range(args.warmup + args.steps):
            if i == args.warmup and args.memory:
                torch.cuda.memory._record_memory_history(max_entries=1000000)

            torch.cuda.synchronize()
            t0 = timeit.default_timer()
            logits = model(inputs)
            torch.cuda.synchronize()
            dt_fw = timeit.default_timer() - t0

            dt_bw = 0.0
            if args.mode in ("forward_backward", "train"):
                loss = cross_entropy(logits, targets)
                t1 = timeit.default_timer()
                loss.backward()
                torch.cuda.synchronize()
                dt_bw = timeit.default_timer() - t1

            dt_opt = 0.0
            if args.mode == "train":
                torch.cuda.synchronize()
                t2 = timeit.default_timer()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                dt_opt = timeit.default_timer() - t2
            else:
                model.zero_grad(set_to_none=True)

            if i >= args.warmup:
                fw_samples.append(dt_fw)
                bw_samples.append(dt_bw)
                opt_samples.append(dt_opt)

    if args.memory:
        mem_name_map = {
            "forward": "f",
            "forward_backward": "fb",
            "train": "fbs",
        }

        m = mem_name_map[args.mode]
        d = "bf16" if args.mixed else "fp32"
        torch.cuda.memory._dump_snapshot(f"out/memory/mem_{d}_{args.size}_{m}_{args.seq_len}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    fw_t = torch.tensor(fw_samples, device="cuda")
    bw_t = torch.tensor(bw_samples, device="cuda")
    opt_t = torch.tensor(opt_samples, device="cuda")
    total = fw_t + bw_t + opt_t

    fw_t_mean = fw_t.mean()
    fw_t_std = fw_t.std() if len(fw_t) > 1 else 0.0
    fw_t_pct = (fw_t / total).mean() * 100

    bw_t_mean = bw_t.mean()
    bw_t_std = bw_t.std() if len(bw_t) > 1 else 0.0
    bw_t_pct = (bw_t / total).mean() * 100

    opt_t_mean = opt_t.mean()
    opt_t_std = opt_t.std() if len(opt_t) > 1 else 0.0
    opt_t_pct = (opt_t / total).mean() * 100

    total_mean = total.mean()
    total_std = total.std() if len(total) > 1 else 0.0
    total_pct = (total / total).mean() * 100

    print(
        f"Timings for {args.size} model, batch size {args.batch_size}, seq len {args.seq_len}, {args.mode} over {args.steps} steps:"
    )

    print(f"Forward:  {fw_t_mean * 1e3:.3f} ± {fw_t_std * 1e3:.3f} ms ({fw_t_pct:.1f}%)")

    if args.mode in ("forward_backward", "train"):
        print(f"Backward: {bw_t_mean * 1e3:.3f} ± {bw_t_std * 1e3:.3f} ms ({bw_t_pct:.1f}%)")

    if args.mode == "train":
        print(f"Optimizer: {opt_t_mean * 1e3:.3f} ± {opt_t_std * 1e3:.3f} ms ({opt_t_pct:.1f}%)")

    print(f"Total:    {total_mean * 1e3:.3f} ± {total_std * 1e3:.3f} ms ({total_pct:.1f}%)")

    print("-" * 80)


if __name__ == "__main__":
    main()
