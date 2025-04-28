import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn


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


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


# Training loop
def run(rank: int, world_size: int, steps: int = 20, batch_size: int = 32, backend: str = "nccl"):
    print(f"Running DDP with rank {rank} and world size {world_size} and backend {backend}")

    device = setup_ddp(rank, world_size, backend)
    dist.barrier()
    # torch.manual_seed(rank)

    assert batch_size % world_size == 0, "Batch size must be divisible by world size"
    local_batch_size = batch_size // world_size

    model = ToyModel().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    # Broadcast initial parameters so all ranks start identically
    for p in model.parameters():
        dist.broadcast(p.data, src=0)

    # Baseline setup for rank 0
    baseline_model = None
    baseline_optim = None
    if rank == 0:
        baseline_model = ToyModel().to(device)
        # Ensure baseline starts with identical parameters as the DDP model after broadcast
        baseline_model.load_state_dict(model.state_dict())
        baseline_optim = torch.optim.SGD(baseline_model.parameters(), lr=1e-2)

    for step in range(steps):
        # Deterministic Data Generation
        data_seed = step
        torch.manual_seed(data_seed)
        all_x = torch.randn(batch_size, 32, device=device)
        all_y = torch.randint(0, 10, (batch_size,), device=device)

        # Restore rank-specific seed if other random operations depended on it
        # torch.manual_seed(rank * steps + step)

        # DDP Training Step with partition for the current rank
        start_idx = rank * local_batch_size
        end_idx = start_idx + local_batch_size
        x = all_x[start_idx:end_idx]
        y = all_y[start_idx:end_idx]

        optim.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()

        # Naive gradient averaging: one all‑reduce per parameter
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad.mul_(1.0 / world_size)

        optim.step()

        # Baseline Training Step on rank 0 only
        if rank == 0:
            baseline_optim.zero_grad(set_to_none=True)
            baseline_logits = baseline_model(all_x)  # Use full batch
            baseline_loss = loss_fn(baseline_logits, all_y)
            baseline_loss.backward()
            baseline_optim.step()

        if device.startswith("cuda"):
            torch.cuda.synchronize(device=device)

    # Verification vs single-process baseline on rank 0
    if rank == 0:
        print("Rank 0: Verifying DDP model parameters against single-process baseline...")
        for (name, param), (base_name, base_param) in zip(model.named_parameters(), baseline_model.named_parameters()):
            assert name == base_name
            assert torch.allclose(param.data, base_param.data), (
                f"Mismatch found in parameter: {name}. Max diff: {torch.abs(param.data - base_param.data).max().item()}"
            )
        print("Rank 0: Verification successful! DDP parameters match baseline.")

    cleanup_ddp()


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--backend", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    WORLD_SIZE = args.world_size
    # Ensure batch size is divisible by world size before spawning processes
    assert args.batch_size % WORLD_SIZE == 0, (
        f"Batch size ({args.batch_size}) must be divisible by world size ({WORLD_SIZE})"
    )
    BACKEND = args.backend or "nccl" if torch.cuda.is_available() else "gloo"
    mp.spawn(run, args=(WORLD_SIZE, args.steps, args.batch_size, BACKEND), nprocs=WORLD_SIZE, join=True)
