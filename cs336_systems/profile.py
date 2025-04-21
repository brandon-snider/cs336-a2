import argparse

import torch
from cs336_basics.model import BasicsTransformerLM as Transformer
from cs336_systems.common import MODEL_SIZES, BATCH_SIZE, VOCAB_SIZE, DEFAULT_DEVICE
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
import torch.cuda.nvtx as nvtx


def profile(
    model: Transformer,
    optimizer: torch.optim.Optimizer | None,
    batch_size: int = BATCH_SIZE,
    seq_len: int = 512,
    warmup_steps: int = 5,
    timed_steps: int = 10,
    device: str = DEFAULT_DEVICE,
    range_prefix: str = "",
    include_backward: bool = False,
    include_step: bool = False,
    include_zero_grad: bool = False,
):
    if not include_backward:
        model.eval()
    else:
        model.train()

    inputs = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device=device)

    for i in range(warmup_steps + timed_steps):
        rp = range_prefix + ("[warmup] " if i < warmup_steps else "")

        # Forward pass
        with nvtx.range(rp + "forward pass"):
            logits = model(inputs)

        # Backward pass
        if include_backward:
            with nvtx.range(rp + "backward pass"):
                logits.mean().backward()

        # Optimizer step (requires optimizer and typically follows backward)
        if include_step:
            with nvtx.range(rp + "step"):
                optimizer.step()

        # Zero grad (requires optimizer and typically follows step or backward)
        if include_zero_grad:
            with nvtx.range(rp + "zero_grad"):
                optimizer.zero_grad()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=str, default="small")
    parser.add_argument("--seq-lens", type=str, default="512")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=10)
    parser.add_argument("--include-backward", action="store_true", help="Include backward pass")
    parser.add_argument("--include-step", action="store_true", help="Include optimizer step")
    parser.add_argument("--include-zero-grad", action="store_true", help="Include zero_grad")
    args = parser.parse_args()

    sizes = args.sizes.split(",")
    if sizes == ["all"]:
        sizes = MODEL_SIZES.keys()

    seq_lens = args.seq_lens.split(",")
    if seq_lens == ["all"]:
        seq_lens = [128, 256, 512, 1024]
    else:
        seq_lens = [int(seq_len) for seq_len in seq_lens]

    for size in sizes:
        for seq_len in seq_lens:
            print(f"Profiling size={size}, seq_len={seq_len} on {args.device}")

            model = Transformer(
                vocab_size=VOCAB_SIZE,
                context_length=seq_len,
                **MODEL_SIZES[size],
            )
            model.to(device=args.device)

            optimizer = None
            if args.include_step or args.include_zero_grad:
                # Optimizer is only needed if step or zero_grad are included
                optimizer = AdamW(model.parameters(), lr=1e-3)

            range_prefix = f"[size={size}, seq_len={seq_len}] "

            profile(
                model,
                optimizer,
                batch_size=args.batch_size,
                seq_len=seq_len,
                warmup_steps=args.warmup_steps,
                timed_steps=args.timed_steps,
                device=args.device,
                range_prefix=range_prefix,
                include_backward=args.include_backward,
                include_step=args.include_step,
                include_zero_grad=args.include_zero_grad,
            )


if __name__ == "__main__":
    main()
