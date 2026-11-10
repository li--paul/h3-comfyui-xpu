#!/usr/bin/env python3
"""
Stress-test XPU all_reduce with sustained large-tensor pressure.

Reproduces GPU hangs by applying the same stress factors as stress_comm.py
but using only dist.all_reduce (no gather/scatter, no model patterns).

Usage:
    export CCL_SYCL_COPY_ENGINE=1

    export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7
    export ZE_AFFINITY_MASK=8,9,10,11,12,13,14,15
    export ZE_AFFINITY_MASK=16,17,18,19,20,21,22,23
    export ZE_AFFINITY_MASK=24,25,26,27,28,29,30,31

    torchrun --nproc-per-node=4 dist_allreduce.py
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])


def init_dist(backend: str = "xccl"):
    dist.init_process_group(backend=backend)
    torch.xpu.set_device(LOCAL_RANK)
    torch.xpu.empty_cache()


@torch.no_grad()
def run_step(h: int, b: int, n_calls: int):
    for _ in range(n_calls):
        t = torch.randn(b, h, dtype=torch.bfloat16, device="xpu")
        dist.all_reduce(t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--batch-tokens", type=int, default=8192)
    parser.add_argument("--calls-per-step", type=int, default=28)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--backend", type=str, default="xccl")
    args = parser.parse_args()

    if WORLD_SIZE != 8:
        print(f"WARNING: expected 8 ranks, got {WORLD_SIZE}", flush=True)

    init_dist(args.backend)

    torch.xpu.synchronize()
    dist.barrier()

    # ---- Correctness verification ----
    if LOCAL_RANK == 0:
        print("Verifying all_reduce correctness...", flush=True)
    row = torch.arange(args.batch_tokens, dtype=torch.float32, device="xpu") % 7
    col = torch.arange(args.hidden_size, dtype=torch.float32, device="xpu") % 6
    v = (row[:, None] + col[None, :] + LOCAL_RANK).to(torch.bfloat16)
    del row, col
    dist.all_reduce(v)
    ranks_sum = WORLD_SIZE * (WORLD_SIZE - 1) // 2
    row2 = torch.arange(args.batch_tokens, dtype=torch.float32, device="xpu") % 7
    col2 = torch.arange(args.hidden_size, dtype=torch.float32, device="xpu") % 6
    expected = ((row2[:, None] + col2[None, :]) * WORLD_SIZE + ranks_sum).to(torch.bfloat16)
    del row2, col2
    if not (v == expected).all():
        mismatch = (v != expected).sum().item()
        raise RuntimeError(
            f"all_reduce correctness check FAILED on rank {LOCAL_RANK}: "
            f"{mismatch}/{v.numel()} elements mismatch"
        )
    if LOCAL_RANK == 0:
        print("all_reduce correctness check PASSED.\n", flush=True)
    torch.xpu.synchronize()
    dist.barrier()

    # Warmup
    for _ in range(args.warmup):
        t = torch.randn(args.batch_tokens, args.hidden_size,
                        dtype=torch.bfloat16, device="xpu")
        dist.all_reduce(t)
    torch.xpu.synchronize()
    dist.barrier()

    if LOCAL_RANK == 0:
        print(f"\n{'='*60}")
        print(f"Configuration:")
        print(f"  hidden_size={args.hidden_size}")
        print(f"  batch_tokens={args.batch_tokens}")
        print(f"  calls_per_step={args.calls_per_step}")
        print(f"  steps={args.steps}, warmup={args.warmup}")
        print(f"  world_size={WORLD_SIZE}, backend={args.backend}")
        print(f"{'='*60}\n", flush=True)

    # Dry-run for JIT compilation
    run_step(args.hidden_size, args.batch_tokens, n_calls=1)
    torch.xpu.synchronize()
    dist.barrier()

    # Timed steps
    laps = []
    for i in range(args.steps):
        t0 = time.perf_counter()
        run_step(args.hidden_size, args.batch_tokens,
                 n_calls=args.calls_per_step)
        torch.xpu.synchronize()
        t1 = time.perf_counter()
        laps.append(t1 - t0)

        if LOCAL_RANK == 0 and (i + 1) % 50 == 0:
            avg = sum(laps[-50:]) / 50
            print(f"  step {i+1:>4d}  avg {avg*1000:.1f} ms/step, "
                  f"{(args.hidden_size * args.batch_tokens * args.calls_per_step * 4 / avg * 1e-9):.1f} gbps", flush=True)

    torch.xpu.synchronize()
    dist.barrier()

    if LOCAL_RANK == 0:
        avg = sum(laps) / len(laps)
        p50 = sorted(laps)[len(laps) // 2]
        p99 = sorted(laps)[int(len(laps) * 0.99)]
        print(f"\n{'='*60}")
        print(f"Results ({args.steps} steps):")
        print(f"  avg: {avg*1000:.1f} ms/step")
        print(f"  p50: {p50*1000:.1f} ms/step")
        print(f"  p99: {p99*1000:.1f} ms/step")
        print(f"  total: {sum(laps):.1f}s")
        print(f"{'='*60}\n", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
