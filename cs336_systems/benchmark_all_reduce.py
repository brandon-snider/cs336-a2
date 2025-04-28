#!/usr/bin/env python3
"""
Benchmark all-reduce communication in PyTorch.
"""

import os
import argparse
import itertools
import timeit

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DEFAULT_BACKENDS = ["gloo"] if not torch.cuda.is_available() else ["nccl", "gloo"]
MB = 1024 * 1024
DEFAULT_BYTE_SIZES = [int(x * MB) for x in (1, 10, 100, 1000)]  # 1 MB … 1 GB
DEFAULT_WORLD_SIZES = [2, 4, 6]

WARMUP_ITERS = 5
TIMED_ITERS = 10


def _run_bench(
    rank: int,
    world_size: int,
    backend: str,
    tensor_bytes: int,
    warmup: int,
    iters: int,
) -> None:
    """Per‑process benchmark routine."""

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12356")
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # Select device
    if backend == "nccl":
        gpu_count = torch.cuda.device_count()
        torch.cuda.set_device(rank % gpu_count)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Allocate tensor
    numel = tensor_bytes // 4  # fp32
    x = torch.randn(numel, dtype=torch.float32, device=device)

    def _sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        dist.all_reduce(x, async_op=False)
        _sync()

    # Timed loop
    latencies: list[float] = []
    for _ in range(iters):
        _sync()
        t0 = timeit.default_timer()
        dist.all_reduce(x, async_op=False)
        _sync()
        latencies.append(timeit.default_timer() - t0)

    # Gather timings
    gathered: list[list[float]] = [None] * world_size
    dist.all_gather_object(gathered, latencies)

    if rank == 0:
        flat = list(itertools.chain.from_iterable(gathered))
        size_mb = tensor_bytes / MB
        print(
            f"{backend.upper():<4} | {device.type:<4} | {world_size} proc | "
            f"{size_mb:>4.0f} MB  ->  "
            f"min {min(flat) * 1e3:6.2f}  mean {np.mean(flat) * 1e3:7.2f}  "
            f"max {max(flat) * 1e3:6.2f}  ms"
        )

    dist.destroy_process_group()


def run_sweep(
    backends: list[str] = DEFAULT_BACKENDS,
    sizes: list[int] = DEFAULT_BYTE_SIZES,
    world_sizes: list[int] = DEFAULT_WORLD_SIZES,
    warmup: int = WARMUP_ITERS,
    iters: int = TIMED_ITERS,
) -> None:
    """Iterate over all requested combinations sequentially."""
    for backend, tensor_bytes, world_size in itertools.product(backends, sizes, world_sizes):
        # Skip NCCL configs that exceed number of visible GPUs
        if backend == "nccl" and torch.cuda.device_count() < world_size:
            print(
                f"[skip] NCCL with world_size={world_size} requires "
                f"≥{world_size} GPUs (found {torch.cuda.device_count()})."
            )
            continue

        mp.spawn(
            _run_bench,
            args=(world_size, backend, tensor_bytes, warmup, iters),
            nprocs=world_size,
            join=True,
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backends", nargs="*", choices=["gloo", "nccl"])
    p.add_argument("--sizes", nargs="*", type=float)
    p.add_argument("--world-sizes", nargs="*", type=int)
    p.add_argument("--iters", type=int, default=TIMED_ITERS)
    p.add_argument("--warmup", type=int, default=WARMUP_ITERS)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    backends = args.backends if args.backends else DEFAULT_BACKENDS
    sizes = [int(x * MB) for x in args.sizes] if args.sizes else DEFAULT_BYTE_SIZES
    world_sizes = args.world_sizes or DEFAULT_WORLD_SIZES

    run_sweep(
        backends=backends,
        sizes=sizes,
        world_sizes=world_sizes,
        warmup=args.warmup,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
