"""
Benchmark all_reduce for different backends, data sizes, and world sizes.

Usage:
uv run -m cs336_systems.benchmark_all_reduce
"""

import argparse
import itertools
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import timeit

BACKEND_DEVICE_MAP = {
    "gloo": "cpu",
    "nccl": "cuda" if torch.cuda.is_available() else "cpu",
}

BACKENDS = ["gloo"] if not torch.cuda.is_available() else ["nccl", "gloo"]
DATA_SIZES = [int(x * 1024**2) for x in [1e0, 1e1, 1e2, 1e3]]  # 1MB, 10MB, 100MB, 1GB
TENSOR_SIZES = [x // 4 for x in DATA_SIZES]  # number of elements to create a float32 tensor of each size
WORLD_SIZES = [2, 4, 6]
N_WARMUP = 5
N_BENCHMARK = 10


def setup(rank, world_size, backend):
    """
    Setup the process group for distributed training.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def all_reduce_test(rank, world_size, backend, tensor_size, n_warmup, n_benchmark):
    """
    Setup the process group, run warmup and benchmark iterations for dist.all_reduce,
    and gather timing results across ranks.
    """
    setup(rank, world_size, backend)
    device = torch.device(BACKEND_DEVICE_MAP[backend])

    # tensor_size is the number of float32 elements
    data = torch.randn(int(tensor_size), device=device, dtype=torch.float32)

    # Warmup iterations
    for _ in range(n_warmup):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        if backend == "nccl":
            torch.cuda.synchronize()

    # Benchmark iterations
    timings = []
    for _ in range(n_benchmark):
        if backend == "nccl":
            torch.cuda.synchronize()
        start_time = timeit.default_timer()

        dist.all_reduce(data, op=dist.ReduceOp.SUM)

        if backend == "nccl":
            torch.cuda.synchronize()
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)

    # Aggregate timings across ranks
    local_avg_time = torch.tensor(np.mean(timings), device=device)
    all_times = [torch.zeros_like(local_avg_time) for _ in range(world_size)]
    dist.all_gather(all_times, local_avg_time)

    if rank == 0:
        all_times_cpu = [t.cpu().item() for t in all_times]
        avg_time = np.mean(all_times_cpu)
        std_time = np.std(all_times_cpu)

        # Calculate data size in MB (float32 = 4 bytes)
        data_size_mb = tensor_size * 4 / 1024**2

        print(
            f"Backend: {backend}, World Size: {world_size}, Data Size: {data_size_mb:.0f}MB, "
            f"Avg Time (ms): {avg_time * 1000:.3f} ± {std_time * 1000:.3f}"
        )

    dist.destroy_process_group()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-sizes", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.world_sizes:
        world_sizes = [int(x) for x in args.world_sizes.split(",")]
    else:
        world_sizes = WORLD_SIZES

    # Iterate over tensor sizes (number of elements) instead of data sizes (bytes)
    for backend, tensor_size, world_size in itertools.product(BACKENDS, TENSOR_SIZES, world_sizes):
        mp.spawn(
            fn=all_reduce_test,
            args=(world_size, backend, tensor_size, N_WARMUP, N_BENCHMARK),
            nprocs=world_size,
            join=True,
        )
