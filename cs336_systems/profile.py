#!/usr/bin/env python

"""
NVTX-based profiling harness for nsys.

Example usage:
- Forward only:
    uv run nsys profile -o out/profiling/forward-sm-128 python -m cs336_systems.profile --size sm --seq-len 128 --mode forward
- Forward + backward:
    uv run nsys profile -o out/profiling/forward-backward-md-128 python -m cs336_systems.profile --size md --seq-len 128 --mode forward_backward
- Train (forward + backward + optimizer step):
    uv run nsys profile -o out/profiling/train-md-128 python -m cs336_systems.profile --size md --seq-len 128 --mode train
- Also annotate scaled_dot_product_attention:
    uv run nsys profile -o out/profiling/sdpa-md-128 python -m cs336_systems.profile --size md --seq-len 128 --mode forward --annotate-sdpa
"""

import argparse
import math
import sys
from contextlib import nullcontext

from einops import einsum
import torch
import torch.cuda.nvtx as nvtx

import cs336_basics.model as _m
from cs336_basics.model import BasicsTransformerLM as Transformer
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import softmax, cross_entropy

# Preset model configurations (same as benchmark.py)
_PRESETS: dict[str, dict[str, int]] = {
    "sm": {"d_model": 768, "d_ff": 3072, "d_layers": 12, "num_heads": 12},
    "md": {"d_model": 1024, "d_ff": 4096, "d_layers": 24, "num_heads": 16},
    "lg": {"d_model": 1280, "d_ff": 5120, "d_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "d_layers": 48, "num_heads": 25},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "d_layers": 32, "num_heads": 32},
}


# Scaled‑dot‑product attention annotated with fine‑grained NVTX ranges
def _sdpa_annotated(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None):  # noqa: N802
    with nvtx.range("sdpa"):
        with nvtx.range("sdpa.compute_scores"):
            d_k = K.shape[-1]
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

        with nvtx.range("sdpa.softmax"):
            attention_weights = softmax(attention_scores, dim=-1)
        with nvtx.range("sdpa.value_matmul"):
            return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


# Parse CLI arguments
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    size = p.add_mutually_exclusive_group(required=True)
    size.add_argument("--size", choices=_PRESETS.keys())
    size.add_argument("--d_model", type=int)

    p.add_argument("--d_ff", type=int)
    p.add_argument("--num-layers", dest="d_layers", type=int)
    p.add_argument("--num-heads", dest="num_heads", type=int)

    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=10_000)

    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=1)

    p.add_argument("--mixed", action="store_true")
    p.add_argument("--compile", action="store_true")

    p.add_argument(
        "--annotate-sdpa", action="store_true", help="Whether to annotate scaled_dot_product_attention with NVTX ranges"
    )
    p.add_argument("--mode", choices=["forward", "forward_backward", "train"], default="forward_backward")

    return p.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This benchmark requires a CUDA-capable GPU.")
        sys.exit(1)

    args = _parse()

    if args.size:
        cfg = _PRESETS[args.size]
    else:
        required = ("d_model", "d_ff", "d_layers", "num_heads")
        if any(getattr(args, k) is None for k in required):
            raise ValueError("Must supply all custom h‑params when --size is omitted")
        cfg = dict(d_model=args.d_model, d_ff=args.d_ff, d_layers=args.d_layers, num_heads=args.num_heads)

    if args.annotate_sdpa:
        _m.scaled_dot_product_attention = _sdpa_annotated

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

    mixed_or_null_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.mixed else nullcontext()

    with mixed_or_null_ctx:
        for _ in range(args.warmup + args.steps):
            mode_str = "warmup." if _ < args.warmup else "profiling."

            with nvtx.range(mode_str + "forward"):
                logits = model(inputs)

            if args.mode in ("forward_backward", "train"):
                with nvtx.range(mode_str + "ce_loss"):
                    loss = cross_entropy(logits, targets)

                with nvtx.range(mode_str + "backward"):
                    loss.backward()

            if args.mode == "train":
                with nvtx.range(mode_str + "optimizer.step"):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)


if __name__ == "__main__":
    main()
