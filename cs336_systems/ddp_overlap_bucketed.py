import torch
import torch.distributed as dist
from torch import nn
from dataclasses import dataclass, field
from typing import NamedTuple


@dataclass
class Bucket:
    params: list[nn.Parameter] = field(default_factory=list)
    num_params: int = 0
    num_params_ready: int = 0


class PendingSync(NamedTuple):
    work: dist.Work
    flattened_grads: torch.Tensor
    grads: list[torch.Tensor]
    params_with_grads: list[nn.Parameter]


class DDPBucketedParameters(nn.Module):
    """
    A minimal Distributed Data Parallel wrapper that overlaps gradient communication with the
    computation of the backward pass by bucketing parameters and all-reducing buckets when
    each of their constituent tensors' gradients are ready.
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb

        # Determine parameters per bucket based on the data type used for the parameters
        param_dtype = next(module.parameters()).dtype
        bytes_per_param = param_dtype.itemsize
        self.bucket_size_params = int(bucket_size_mb * 1024**2 / bytes_per_param)

        # Buckets of parameters, each of which will be all-reduced together
        self.buckets: list[Bucket] = []
        self._init_buckets()

        self.handles: list[PendingSync] = []
        self.world_size = dist.get_world_size()

        # Broadcast parameters from rank 0 to all other ranks
        for p in self.module.parameters():
            dist.broadcast(p.data, src=0, async_op=False)

        self._register_hooks()

    def _init_buckets(self):
        """Assign parameters to buckets, respecting the maximum bucket size."""
        this_bucket = Bucket()
        this_bucket_idx = 0

        for p in reversed(list(self.module.parameters())):
            if not p.requires_grad:
                continue

            if this_bucket.num_params + p.numel() > self.bucket_size_params and this_bucket.num_params > 0:
                self.buckets.append(this_bucket)
                this_bucket = Bucket()
                this_bucket_idx += 1

            # Store bucket index directly on the parameter object (for hook lookup)
            p.bucket_idx = this_bucket_idx

            this_bucket.params.append(p)
            this_bucket.num_params += p.numel()

        if this_bucket.num_params > 0:
            self.buckets.append(this_bucket)

    def _register_hooks(self):
        """Register post-accumulate gradient hook for each parameter"""

        def make_hook(param: torch.nn.Parameter):
            def hook(*_: torch.Tensor):
                bucket_idx = param.bucket_idx
                bucket = self.buckets[bucket_idx]
                bucket.num_params_ready += param.numel()

                # If all parameters in the bucket have accumulated gradients, all-reduce them
                if bucket.num_params_ready == bucket.num_params:
                    params_with_grads = [p for p in bucket.params if p.grad is not None]
                    if not params_with_grads:
                        bucket.num_params_ready = 0
                        return

                    grads = [p.grad for p in params_with_grads]
                    flattened_grads = torch._utils._flatten_dense_tensors(grads)
                    handle = dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM, async_op=True)
                    self.handles.append(PendingSync(handle, flattened_grads, grads, params_with_grads))
                    bucket.num_params_ready = 0

            return hook

        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(make_hook(p))

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        """Block until all outstanding gradient all‑reduces have completed."""
        for handle in self.handles:
            handle.work.wait()

            handle.flattened_grads.div_(self.world_size)
            unflattened_grads = torch._utils._unflatten_dense_tensors(handle.flattened_grads, handle.grads)
            for p, updated_grad in zip(handle.params_with_grads, unflattened_grads):
                p.grad.copy_(updated_grad)
        self.handles.clear()
