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

BACKENDS = ["gloo"] if not torch.cuda.is_available() else ["gloo", "nccl"]
DATA_SIZES = [int(x * 1024**2) for x in [1e0, 1e1, 1e2, 1e3]]  # 1MB, 10MB, 100MB, 1GB
WORLD_SIZES = [2, 4, 6]
N_WARMUP = 5
N_BENCHMARK = 10


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def all_reduce_test(rank, world_size, backend, data_size, n_warmup, n_benchmark):
    """
    Setup the process group, run warmup and benchmark iterations for dist.all_reduce,
    and gather timing results across ranks.
    """
    setup(rank, world_size, backend)
    device = torch.device(BACKEND_DEVICE_MAP[backend])
    data = torch.randn(data_size, device=device)

    # Warmup iterations
    for _ in range(n_warmup):
        dist.all_reduce(data, op=dist.ReduceOp.SUM)
        if backend == "nccl":
            torch.cuda.synchronize(device=device)

    # Benchmark iterations
    timings = []
    for _ in range(n_benchmark):
        if backend == "nccl":
            torch.cuda.synchronize(device=device)
        start_time = timeit.default_timer()

        dist.all_reduce(data, op=dist.ReduceOp.SUM)

        if backend == "nccl":
            torch.cuda.synchronize(device=device)
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
        print(
            f"Backend: {backend}, World Size: {world_size}, Data Size: {data_size / 1024**2:.0f}MB, "
            f"Avg Time (ms): {avg_time * 1000:.3f} ± {std_time * 1000:.3f}"
        )

    cleanup()


if __name__ == "__main__":
    for backend, data_size, world_size in itertools.product(BACKENDS, DATA_SIZES, WORLD_SIZES):
        mp.spawn(
            fn=all_reduce_test,
            args=(world_size, backend, data_size, N_WARMUP, N_BENCHMARK),
            nprocs=world_size,
        )
