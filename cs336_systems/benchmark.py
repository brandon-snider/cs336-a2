import argparse
import timeit
import torch

from cs336_basics.model import BasicsTransformerLM as Transformer
from cs336_basics.nn_utils import cross_entropy

MODEL_SIZES = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 1600,
        "d_ff": 6400,
        "num_layers": 48,
        "num_heads": 25,
    },
    "2.7B": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
}


def benchmark(
    model: Transformer,
    batch_size: int = 4,
    seq_len: int = 512,
    warmup_steps: int = 5,
    timed_steps: int = 10,
    include_backward: bool = True,
    device: str = "cpu",
) -> dict:
    if not include_backward:
        model.eval()
    else:
        model.train()

    inputs = torch.randint(0, 10000, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 10000, (batch_size, seq_len), device=device)

    forward_times = []
    backward_times = []
    total_times = []

    if device == "cuda":
        torch.cuda.synchronize()

    for i in range(warmup_steps + timed_steps):
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = timeit.default_timer()

        # Forward pass
        logits = model(inputs)
        loss = cross_entropy(logits, targets)

        if device == "cuda":
            torch.cuda.synchronize()
        forward_end_time = timeit.default_timer()

        # Backward pass
        if include_backward:
            loss.backward()
            if device == "cuda":
                torch.cuda.synchronize()
        backward_end_time = timeit.default_timer()

        if i >= warmup_steps:
            forward_duration = forward_end_time - start_time
            backward_duration = backward_end_time - forward_end_time if include_backward else 0
            total_duration = backward_end_time - start_time

            forward_times.append(forward_duration)
            if include_backward:
                backward_times.append(backward_duration)
            total_times.append(total_duration)

    forward_times_t = torch.tensor(forward_times)
    backward_times_t = torch.tensor(backward_times) if include_backward else torch.tensor([])
    total_times_t = torch.tensor(total_times)

    results = {
        "forward_times": forward_times,
        "forward_mean": forward_times_t.mean().item(),
        "forward_std": forward_times_t.std().item(),
        "total_times": total_times,
        "total_mean": total_times_t.mean().item(),
        "total_std": total_times_t.std().item(),
    }
    if include_backward:
        results.update(
            {
                "backward_times": backward_times,
                "backward_mean": backward_times_t.mean().item(),
                "backward_std": backward_times_t.std().item(),
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark a transformer model")
    parser.add_argument("--size", type=str, help="Model size", default="small")
    parser.add_argument("--d_model", type=int, help="Model dimension")
    parser.add_argument("--d_ff", type=int, help="Feed-forward dimension")
    parser.add_argument("--num_layers", type=int, help="Number of layers")
    parser.add_argument("--num_heads", type=int, help="Number of heads")
    parser.add_argument("--batch-size", type=int, help="Batch size", default=4)
    parser.add_argument("--seq-len", type=int, help="Sequence length", default=512)
    parser.add_argument("--warmup-steps", type=int, help="Warmup steps", default=5)
    parser.add_argument("--timed-steps", type=int, help="Timed steps", default=10)
    parser.add_argument("--include-backward", type=bool, help="Include backward", default=True)
    parser.add_argument("--device", type=str, help="Device (default: cuda, mps, cpu)")
    parser.add_argument("--dtype", type=str, help="Data type (default: bfloat16, float32)")
    args = parser.parse_args()

    device = (
        args.device or "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    if args.dtype:
        dtype = getattr(torch, args.dtype) if hasattr(torch, args.dtype) else dtype

    # Get defaults from size, then override with any explicitly provided args
    model_config = MODEL_SIZES.get(args.size, {})
    d_model = args.d_model if args.d_model is not None else model_config["d_model"]
    d_ff = args.d_ff if args.d_ff is not None else model_config["d_ff"]
    num_layers = args.num_layers if args.num_layers is not None else model_config["num_layers"]
    num_heads = args.num_heads if args.num_heads is not None else model_config["num_heads"]

    model = Transformer(
        vocab_size=10000,
        context_length=args.seq_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=10000.0,
    )
    model.to(device=device)

    results = benchmark(
        model, args.batch_size, args.seq_len, args.warmup_steps, args.timed_steps, args.include_backward, device=device
    )

    print(
        f"Benchmarking results for {args.size} model ({d_model=}, {num_layers=}, {num_heads=}, {d_ff=}) on {device} with {dtype}:"
    )
    print(f"Batch size: {args.batch_size}, Sequence length: {args.seq_len}")
    print(f"Forward pass:  mean={results['forward_mean']:.4f}s, std={results['forward_std']:.4f}s")
    if args.include_backward:
        print(f"Backward pass: mean={results['backward_mean']:.4f}s, std={results['backward_std']:.4f}s")
    print(f"Total time:    mean={results['total_mean']:.4f}s, std={results['total_std']:.4f}s")


if __name__ == "__main__":
    main()
