from contextlib import nullcontext
import os
import argparse
import timeit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import itertools
from cs336_basics.model import BasicsTransformerLM as Transformer
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.benchmark import _PRESETS
from cs336_systems.ddp_overlap_bucketed import DDPBucketedParameters
from cs336_systems.ddp_overlap_individual import DDPIndividualParameters
from cs336_systems.optimizer_state_sharding import ShardedOptimizer

VOCAB_SIZE = 10000


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> str:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12357")

    if torch.cuda.is_available() and backend == "nccl":
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)
        else:
            raise ValueError("Unable to find CUDA devices.")
        device = f"cuda:{local_rank}"
    else:
        device = "cpu"

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def cleanup_ddp() -> None:
    dist.barrier()
    dist.destroy_process_group()


# Training loop
def run(
    rank: int,
    world_size: int,
    warmup: int = 5,
    steps: int = 5,
    batch_size: int = 32,
    seq_len: int = 256,
    backend: str = "nccl",
    mixed_precision: bool = False,
    jit: bool = False,
    flat: bool = False,
    overlap_individual: bool = False,
    overlap_bucketed: bool = False,
    bucket_size_mb: float = 100,
    shard_optimizer: bool = False,
    report_memory: bool = False,
):
    print(
        f"Running DDP with rank {rank}, world size {world_size}, batch size {batch_size}, seq len {seq_len}, and backend {backend}"
    )

    device = setup_ddp(rank, world_size, backend)
    dist.barrier()

    def _sync() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize(device=device)

    assert batch_size % world_size == 0, "Batch size must be divisible by world size"
    local_batch_size = batch_size // world_size

    # Reset peak memory stats before any allocation on this rank
    torch.cuda.reset_peak_memory_stats()

    preset = _PRESETS["xl"]
    model = Transformer(
        vocab_size=10000,
        context_length=seq_len,
        d_model=preset["d_model"],
        num_layers=preset["num_layers"],
        num_heads=preset["num_heads"],
        d_ff=preset["d_ff"],
    ).to(device)

    if jit:
        model = torch.compile(model)

    overlap = True if overlap_individual or overlap_bucketed else False

    if overlap_individual:
        model = DDPIndividualParameters(model)
    elif overlap_bucketed:
        model = DDPBucketedParameters(model, bucket_size_mb=bucket_size_mb)

    # Measure peak memory after model initialization
    mem_after_init = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params}")

    if shard_optimizer:
        optim = ShardedOptimizer(model.parameters(), AdamW, lr=1e-3)
    else:
        optim = AdamW(model.parameters(), lr=1e-3)

    loss_fn = cross_entropy

    if not overlap:
        # Broadcast initial parameters so all ranks start identically
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

    times_total = []
    times_comm = []

    mems_before_step = []
    mems_after_step = []

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if mixed_precision else nullcontext()

    for i in range(warmup + steps):
        x = torch.randint(0, VOCAB_SIZE, (local_batch_size, seq_len), device=device)
        y = torch.randint(0, VOCAB_SIZE, (local_batch_size, seq_len), device=device)

        _sync()
        tt0 = timeit.default_timer()

        optim.zero_grad(set_to_none=True)
        with ctx:
            logits = model(x)
            loss = loss_fn(logits, y)
        loss.backward()

        if not overlap:
            _sync()
            tc0 = timeit.default_timer()

            if flat:
                params_with_grads = [p for p in model.parameters() if p.grad is not None]
                if not params_with_grads:
                    continue

                grads = [p.grad for p in params_with_grads]
                flattened_grads = torch._utils._flatten_dense_tensors(grads)

                dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM)

                flattened_grads.mul_(1.0 / world_size)
                updated_grads = torch._utils._unflatten_dense_tensors(flattened_grads, grads)

                for p, updated_grad in zip(params_with_grads, updated_grads):
                    p.grad = updated_grad
            else:
                # Naive gradient averaging: one all‑reduce per parameter
                for p in model.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                        p.grad.mul_(1.0 / world_size)

            _sync()
            tc1 = timeit.default_timer()
        else:
            tc1 = 0
            tc0 = 0

        if overlap:
            model.finish_gradient_synchronization()

        # Measure peak memory just before optimizer step (captures peak memory during fwd, bwd, and grad sync)
        mem_b_step = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        optim.step()

        # Measure peak memory just after optimizer step (captures peak memory during the optimizer step itself)
        mem_a_step = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        _sync()
        tt1 = timeit.default_timer()

        if i >= warmup:
            times_total.append(tt1 - tt0)
            times_comm.append(tc1 - tc0)
            mems_before_step.append(mem_b_step)
            mems_after_step.append(mem_a_step)

    # Gather timings and memory stats
    gathered_totals: list[list[float]] | None = [None] * world_size if rank == 0 else None
    gathered_comms: list[list[float]] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(times_total, gathered_totals, dst=0)
    dist.gather_object(times_comm, gathered_comms, dst=0)

    # Gather single init memory value by putting it in a list
    gathered_mem_after_init: list[list[float]] | None = [None] * world_size if rank == 0 else None
    dist.gather_object([mem_after_init], gathered_mem_after_init, dst=0)  # Note: [mem_after_init]

    # Gather lists of step memory values
    gathered_mems_before_step: list[list[float]] | None = [None] * world_size if rank == 0 else None
    gathered_mems_after_step: list[list[float]] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(mems_before_step, gathered_mems_before_step, dst=0)
    dist.gather_object(mems_after_step, gathered_mems_after_step, dst=0)

    # On rank 0, print results
    if rank == 0:
        all_totals = np.array(list(itertools.chain.from_iterable(gathered_totals)))
        all_comms = np.array(list(itertools.chain.from_iterable(gathered_comms)))

        total_mean_ms = np.mean(all_totals) * 1000
        total_std_ms = np.std(all_totals) * 1000
        comm_mean_ms = np.mean(all_comms) * 1000
        comm_std_ms = np.std(all_comms) * 1000
        comm_proportion = comm_mean_ms / total_mean_ms if total_mean_ms > 0 else 0

        # Convert bytes to MB
        bytes_to_mb = 1 / (1024 * 1024)
        all_mem_after_init = np.array(gathered_mem_after_init).flatten() * bytes_to_mb
        mem_after_init_mean_mb = np.mean(all_mem_after_init)
        mem_after_init_std_mb = np.std(all_mem_after_init)

        all_mems_before_step = np.array(list(itertools.chain.from_iterable(gathered_mems_before_step))) * bytes_to_mb
        all_mems_after_step = np.array(list(itertools.chain.from_iterable(gathered_mems_after_step))) * bytes_to_mb

        mems_before_step_mean_mb = np.mean(all_mems_before_step)
        mems_before_step_std_mb = np.std(all_mems_before_step)
        mems_after_step_mean_mb = np.mean(all_mems_after_step)
        mems_after_step_std_mb = np.std(all_mems_after_step)

        print(
            f"--- Results for Batch Size: {batch_size}, Seq Len: {seq_len}, Backend: {backend}, Warmup: {warmup}, Steps: {steps} ---"
        )
        print(
            f"--- Using: {'JIT, ' if jit else ''}{'Mixed Precision, ' if mixed_precision else ''}{'Flat, ' if flat else ''}{'Overlap Individual, ' if overlap_individual else ''}{'Overlap Bucketed with bucket size: ' + str(bucket_size_mb) + ' MB, ' if overlap_bucketed else ''}{'Sharded Optimizer' if shard_optimizer else ''} ---"
        )

        print(f"Avg total time / step : {total_mean_ms:.2f} ± {total_std_ms:.2f} ms")

        if not overlap:
            print(f"Avg comm time / step  : {comm_mean_ms:.2f} ± {comm_std_ms:.2f} ms")
            print(f"Comm proportion       : {comm_proportion:.2%}")

        if report_memory:
            print("-" * 20 + " Memory Stats (MB) " + "-" * 20)
            print(
                f"Avg peak mem (across ranks) after init  : {mem_after_init_mean_mb:.2f} ± {mem_after_init_std_mb:.2f} MB"
            )
            print(
                f"Avg peak mem (across ranks) before step : {mems_before_step_mean_mb:.2f} ± {mems_before_step_std_mb:.2f} MB"
            )
            print(
                f"Avg peak mem (across ranks) after step  : {mems_after_step_mean_mb:.2f} ± {mems_after_step_std_mb:.2f} MB"
            )

            print("-" * 50)

    cleanup_ddp()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[128])
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--overlap-individual", action="store_true")
    parser.add_argument("--overlap-bucketed", action="store_true")
    parser.add_argument("--bucket-sizes-mb", type=float, nargs="+", default=[100])
    parser.add_argument("--shard-optimizer", action="store_true")
    parser.add_argument("--report-memory", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    WORLD_SIZE = args.world_size
    BACKEND = args.backend or "nccl" if torch.cuda.is_available() else "gloo"

    if args.tf32:
        torch.set_float32_matmul_precision("high")

    for bucket_size_mb in args.bucket_sizes_mb:
        for batch_size, seq_len in itertools.product(args.batch_sizes, args.seq_lens):
            assert batch_size % WORLD_SIZE == 0, (
                f"Batch size ({batch_size}) must be divisible by world size ({WORLD_SIZE})"
            )

            mp.spawn(
                run,
                args=(
                    WORLD_SIZE,
                    args.warmup,
                    args.steps,
                    batch_size,
                    seq_len,
                    BACKEND,
                    args.mixed_precision,
                    args.jit,
                    args.flat,
                    args.overlap_individual,
                    args.overlap_bucketed,
                    bucket_size_mb,
                    args.shard_optimizer,
                    args.report_memory,
                ),
                nprocs=WORLD_SIZE,
                join=True,
            )
