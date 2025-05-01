#!/usr/bin/env python3

"""Minimal benchmark for `scaled_dot_product_attention`"""

import timeit
import torch
import argparse
from cs336_basics.model import scaled_dot_product_attention

BATCH = 8
D_LIST = [16, 32, 64, 128]
SEQ_LIST = [256, 1024, 4096, 8192, 16384]
WARMUP = 5
STEPS = 100

print("d_model\tseq_len\tfwd(ms)\tbwd(ms)\tmem_before(MB)")


def main(compile=False):
    attn_fn = torch.compile(scaled_dot_product_attention) if compile else scaled_dot_product_attention

    for d in D_LIST:
        for n in SEQ_LIST:
            try:
                # fresh tensors per configuration
                q = torch.randn(BATCH, n, d, device="cuda", requires_grad=True)
                k = torch.randn_like(q)
                v = torch.randn_like(q)

                # warm-up forward
                with torch.no_grad():
                    for _ in range(WARMUP):
                        attn_fn(q, k, v)
                        torch.cuda.synchronize()

                # timed forward
                with torch.no_grad():
                    torch.cuda.synchronize()
                    t0 = timeit.default_timer()
                    for _ in range(STEPS):
                        attn_fn(q, k, v)
                        torch.cuda.synchronize()
                    fwd_ms = (timeit.default_timer() - t0) * 1e3 / STEPS

                # warm-up backward
                for _ in range(WARMUP):
                    out = attn_fn(q, k, v)
                    out.sum().backward()
                    torch.cuda.synchronize()
                    q.grad = k.grad = v.grad = None

                # timed backward & mem
                mem_before_total = 0.0
                torch.cuda.synchronize()
                t0 = timeit.default_timer()
                for _ in range(STEPS):
                    out = attn_fn(q, k, v)
                    torch.cuda.synchronize()
                    mem_before_total += torch.cuda.memory_allocated()
                    out.sum().backward()
                    torch.cuda.synchronize()
                    q.grad = k.grad = v.grad = None
                bwd_ms = (timeit.default_timer() - t0) * 1e3 / STEPS
                mem_before_mb = mem_before_total / STEPS / 1e6

                print(f"{d}\t{n}\t{fwd_ms:.2f}\t{bwd_ms:.2f}\t{mem_before_mb:.0f}")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"{d}\t{n}\tOOM\tOOM\tOOM")
                else:
                    raise


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.compile)
