import torch
import triton

BATCH_SIZE = 1
SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
D_MODELS = [16, 32, 64, 128]
D_TYPES = [torch.float16, torch.float32]
WARMUP = 5
REP = 10


def benchmark_forward(
    impl: torch.autograd.Function,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    is_causal=False,
    warmup=WARMUP,
    rep=REP,
):
    def forward_fn():
        return impl.apply(Q, K, V, is_causal)

    torch.cuda.synchronize()
    return triton.testing.do_bench(forward_fn, warmup=warmup, rep=rep)
