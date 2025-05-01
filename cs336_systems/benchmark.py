#!/usr/bin/env python
"""
`timeit`-based benchmarking.

Example usage:
- Forward only (inference mode):
    python -m cs336_systems.benchmark --size sm --mode forward
- Forward + backward:
    python -m cs336_systems.benchmark --size sm --mode forward_backward
- Forward + backward + optimizer step (i.e. full training step):
    python -m cs336_systems.benchmark --size sm --mode train
"""

import argparse
from contextlib import nullcontext
import timeit
import torch
import os

from cs336_basics.optimizer import AdamW
from cs336_basics.model import BasicsTransformerLM as Transformer
from cs336_basics.nn_utils import cross_entropy

_PRESETS: dict[str, dict[str, int]] = {
    "sm": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "md": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "lg": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

_MODE_NAME_MAP = {
    "forward": "f",
    "forward_backward": "fb",
    "train": "fbs",
}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    size_grp = p.add_mutually_exclusive_group(required=True)
    size_grp.add_argument("--size", choices=_PRESETS.keys())
    size_grp.add_argument("--d_model", type=int)

    p.add_argument("--mode", choices=["forward", "forward_backward", "train"], default="forward")

    p.add_argument("--d_ff", type=int)
    p.add_argument("--num-layers", dest="num_layers", type=int)
    p.add_argument("--num-heads", dest="num_heads", type=int)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=10_000)

    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)

    p.add_argument("--mixed", action="store_true")
    p.add_argument("--compile", dest="compile_model", action="store_true")
    p.add_argument("--memory", dest="profile_memory", action="store_true")

    return p.parse_args()


def run_benchmark(
    model_config: dict[str, any],
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    warmup: int,
    steps: int,
    mixed: bool,
    compile_model: bool,
    mode: str,
    profile_memory: bool,
    size_name: str | None = None,
    device: str = "cuda",
) -> dict[str, float]:
    """Runs the benchmark with the given configuration.

    Returns:
        A dictionary containing timing statistics (mean, std, percentage) for
        forward, backward (if applicable), optimizer (if applicable), and total
        (forward [+ backward] [+ optimizer]).
    """
    if not torch.cuda.is_available() and device == "cuda":
        raise RuntimeError("CUDA is not available, but device='cuda' was requested.")

    if "d_layers" in model_config:
        model_config["num_layers"] = model_config.pop("d_layers")

    model: torch.nn.Module = Transformer(vocab_size=vocab_size, context_length=seq_len, **model_config).to(
        device=device
    )

    if compile_model:
        model = torch.compile(model)

    model.train(mode in ("forward_backward", "train"))
    optimizer = AdamW(model.parameters(), lr=1e-3) if mode == "train" else None

    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    mixed_or_null_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if mixed else nullcontext()

    fw_samples, bw_samples, opt_samples = [], [], []

    with mixed_or_null_ctx:
        for i in range(warmup + steps):
            if i == warmup and profile_memory:
                torch.cuda.memory._record_memory_history(max_entries=1000000)

            torch.cuda.synchronize(device=device)
            t0 = timeit.default_timer()
            logits = model(inputs)
            torch.cuda.synchronize(device=device)
            dt_fw = timeit.default_timer() - t0

            dt_bw = 0.0
            if mode in ("forward_backward", "train"):
                loss = cross_entropy(logits, targets)
                t1 = timeit.default_timer()
                loss.backward()
                torch.cuda.synchronize(device=device)
                dt_bw = timeit.default_timer() - t1

            dt_opt = 0.0
            if mode == "train":
                t2 = timeit.default_timer()
                optimizer.step()
                torch.cuda.synchronize(device=device)
                dt_opt = timeit.default_timer() - t2
                optimizer.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)

            if i >= warmup:
                fw_samples.append(dt_fw)
                bw_samples.append(dt_bw)
                opt_samples.append(dt_opt)

    if profile_memory:
        m = _MODE_NAME_MAP[mode]
        d = "bf16" if mixed else "fp32"

        size_str = size_name if size_name else f"custom_d{model_config['d_model']}"
        output_dir = "out/memory"
        os.makedirs(output_dir, exist_ok=True)
        torch.cuda.memory._dump_snapshot(f"{output_dir}/mem_{d}_{size_str}_{m}_{seq_len}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    fw_t = torch.tensor(fw_samples, device=device)
    bw_t = torch.tensor(bw_samples, device=device)
    opt_t = torch.tensor(opt_samples, device=device)
    total_t = fw_t + bw_t + opt_t

    results = {}

    results["fw_mean_ms"] = fw_t.mean().item() * 1e3
    results["fw_std_ms"] = fw_t.std().item() * 1e3 if len(fw_t) > 1 else 0.0
    results["fw_pct"] = (fw_t / total_t).mean().item() * 100 if total_t.mean() > 0 else 0.0

    results["bw_mean_ms"] = bw_t.mean().item() * 1e3 if mode in ("forward_backward", "train") else 0.0
    results["bw_std_ms"] = bw_t.std().item() * 1e3 if len(bw_t) > 1 and mode in ("forward_backward", "train") else 0.0
    results["bw_pct"] = (
        (bw_t / total_t).mean().item() * 100 if total_t.mean() > 0 and mode in ("forward_backward", "train") else 0.0
    )

    results["opt_mean_ms"] = opt_t.mean().item() * 1e3 if mode == "train" else 0.0
    results["opt_std_ms"] = opt_t.std().item() * 1e3 if len(opt_t) > 1 and mode == "train" else 0.0
    results["opt_pct"] = (opt_t / total_t).mean().item() * 100 if total_t.mean() > 0 and mode == "train" else 0.0

    results["total_mean_ms"] = total_t.mean().item() * 1e3
    results["total_std_ms"] = total_t.std().item() * 1e3 if len(total_t) > 1 else 0.0
    results["total_pct"] = 100.0

    return results


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    args = _parse_args()
    size_name = args.size

    if args.size:
        cfg = _PRESETS[args.size].copy()
    else:
        required = ("d_model", "d_ff", "num_layers", "num_heads")
        if any(getattr(args, k) is None for k in required):
            raise ValueError("Must supply all of --d_model, --d_ff, --num-layers, --num-heads when --size is omitted.")

        cfg = dict(d_model=args.d_model, d_ff=args.d_ff, num_layers=args.num_layers, num_heads=args.num_heads)

    results = run_benchmark(
        model_config=cfg,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        warmup=args.warmup,
        steps=args.steps,
        mixed=args.mixed,
        compile_model=args.compile_model,
        mode=args.mode,
        profile_memory=args.profile_memory,
        size_name=size_name,
        device="cuda",
    )

    size_str = size_name if size_name else f"custom_d{cfg['d_model']}"

    print("size\tbsz\tseq\tmode\tcompile\tmixed\tfw_ms\tbw_ms\topt_ms\ttotal_ms")

    print(
        f"{size_str}\t{args.batch_size}\t{args.seq_len}\t{_MODE_NAME_MAP[args.mode]}\t{args.compile_model}\t{args.mixed}\t"
        f"{results['fw_mean_ms']:.3f}\t{results['bw_mean_ms']:.3f}\t{results['opt_mean_ms']:.3f}\t{results['total_mean_ms']:.3f}"
    )


if __name__ == "__main__":
    main()
