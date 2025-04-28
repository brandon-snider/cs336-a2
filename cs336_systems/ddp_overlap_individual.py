import torch
import torch.distributed as dist
from torch import nn


class DDPIndividualParameters(nn.Module):
    """
    A minimal Distributed Data‑Parallel wrapper that overlaps gradient communication with the
    computation of the backward pass by immediately launching an asynch `all_reduce` on each
    parameter’s gradient as soon as it is produced.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        self.handles: list[tuple[dist.Work, torch.nn.Parameter]] = []
        self.world_size = dist.get_world_size()

        # Broadcast parameters from rank 0 to all other ranks
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0, async_op=False)

        # Register post-accumulate gradient hook for each parameter
        def make_hook(param: torch.nn.Parameter):
            def hook(*_: torch.Tensor):
                handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
                self.handles.append((handle, param))

            return hook

        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(make_hook(p))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Block until all outstanding gradient all‑reduces have completed."""
        for work, param in self.handles:
            work.wait()
            param.grad.div_(self.world_size)
        self.handles.clear()
