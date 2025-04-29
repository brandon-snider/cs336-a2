import torch
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs: any):
        self.world_size: int = dist.get_world_size()
        self.rank: int = dist.get_rank()
        self.total_params: int = 0
        self.params_per_rank: list[int] = [0] * self.world_size

        # Maps parameter to the rank responsible for optimizing it
        self.param_rank_map: dict[torch.nn.Parameter, int] = {}

        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs

        # The underlying optimizer instance, initialized lazily (in `add_param_group`)
        self.optimizer: torch.optim.Optimizer | None = None

        # Call the superclass constructor, which calls `add_param_group` for the initial parameters
        super().__init__(params, defaults={})

    def step(self, closure=None, **kwargs):
        if self.optimizer is not None:
            self.optimizer.step(closure, **kwargs)

        # Synchronize all parameters across ranks after the step
        for p in self.param_rank_map:
            dist.broadcast(p.data, src=self.param_rank_map[p])

    def add_param_group(self, param_group: dict[str, any]):
        """Assigns parameters to ranks and potentially initializes or adds to the local optimizer."""
        params_to_add_locally: list[torch.nn.Parameter] = []

        for p in param_group["params"]:
            num_params = p.numel()
            self.total_params += num_params

            # Assign parameter to the rank with the fewest parameters currently
            min_params_on_any_rank = min(self.params_per_rank)
            target_rank = self.params_per_rank.index(min_params_on_any_rank)

            self.param_rank_map[p] = target_rank
            self.params_per_rank[target_rank] += num_params

            if self.rank == target_rank:
                params_to_add_locally.append(p)

        # If this rank is responsible for any parameters in the new group, update the local optimizer
        if params_to_add_locally:
            # Extract non-parameter settings from the group
            base_cfg = {k: v for k, v in param_group.items() if k != "params"}
            new_local_group = {**base_cfg, "params": params_to_add_locally}

            if self.optimizer is None:
                # Initialize the optimizer if this is the first group assigned to this rank
                self.optimizer = self.optimizer_cls([new_local_group], **self.optimizer_kwargs)
            else:
                # Add the new group to the existing optimizer
                self.optimizer.add_param_group(new_local_group)

        # Call the superclass method to maintain the parameter group structure internally
        super().add_param_group(param_group)
