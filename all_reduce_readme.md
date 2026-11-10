# run_dist_allreduce.sh — XPU `all_reduce` stress test

`run_dist_allreduce.sh` drives `dist_allreduce.py`: a sustained, multi-rank
`all_reduce` loop over large tensors. It exists to reproduce the failure mode
where an XPU collective stops making progress and the GPU needs a reset — the
same shape of load that a tensor-parallel inference server puts on the fabric.

The wrapper exists because `dist_allreduce.py` cannot start on this host as-is:
two library-resolution problems kill it before it reaches a single collective.
Both are handled here (see [The environment the wrapper sets](#the-environment-the-wrapper-sets)),
and both are silent enough to look like "torch is broken".

```bash
./run_dist_allreduce.sh                                        # 4 ranks, XPU 16-19
./run_dist_allreduce.sh --ranks 8                              # 8 ranks, XPU 16-23
./run_dist_allreduce.sh --gpus 24,25,26,27 --ranks 4
./run_dist_allreduce.sh --timeout 600 -- --steps 50 --calls-per-step 8
```

## What actually runs

`dist_allreduce.py` runs five phases per rank, in order:

| Phase | What it does | Why |
|---|---|---|
| init | `init_process_group(backend="xccl")`, `torch.xpu.set_device(LOCAL_RANK)` | oneCCL over the XPU fabric |
| correctness | builds `row + col + rank` from index patterns, `all_reduce`s it, compares against the analytic sum `(row+col)*W + W(W-1)/2` | catches a silent partial or corrupt reduction, which a timing run alone would not |
| warmup | 50 × (`randn` + `all_reduce`) | compile/JIT warmup |
| dry run | one call, then `synchronize` + `barrier` | makes the first timed step comparable to the rest |
| timed | `steps × calls_per_step` reductions, `synchronize` per step, record each lap | the measurement |

The tensor is `batch_tokens × hidden_size` bf16 — 8192 × 4096 = **64 MiB** by
default, allocated fresh on every call so allocator pressure is part of the test.

The correctness check is exact rather than approximate, and that is safe: with
the defaults every element is an integer ≤ 160, and bf16's 8-bit mantissa
represents integers up to 256 exactly.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--ranks N` | `4` | Number of processes (`--nproc-per-node`) |
| `--gpus LIST` | `16..(16+N-1)` | Device ids, comma list or range (`16-19`); count must equal `--ranks` |
| `--timeout S` | `3600` | Wall clock limit; a timeout is reported as a hang |
| `--port P` | `29500` | torchrun rendezvous port (`MASTER_PORT`) |
| `--log PATH` | `h3_output/dist_allreduce_<timestamp>.log` | Also teed to the terminal |
| `--venv PATH` | `.venv` | Environment to run from (`STRESS_VENV` works too) |
| `--` | — | Everything after this goes to `dist_allreduce.py` |

The python script's own knobs, passed after `--`:

| Flag | Default | Meaning |
|---|---|---|
| `--hidden-size` | `4096` | Tensor columns |
| `--batch-tokens` | `8192` | Tensor rows |
| `--calls-per-step` | `28` | `all_reduce` calls per timed step |
| `--steps` | `500` | Timed steps |
| `--warmup` | `50` | Untimed warmup iterations |
| `--backend` | `xccl` | Leave alone on XPU |

So the default run is 500 × 28 = **14,000 reductions of 64 MiB each**.

## The environment the wrapper sets

Set explicitly, every run, rather than inherited:

| Variable | Value | Why it is load-bearing |
|---|---|---|
| `LD_LIBRARY_PATH` | `<oneAPI libfabric>:<venv>/lib` | *Replaced*, not extended. The venv's `libsycl.so.9` (`intel-sycl-rt` 2026.1.0) needs the `libur_loader` that ships beside it; a newer oneAPI puts its own first and the import dies with `undefined symbol: urKernelSetArgLocal, version LIBUR_LOADER_0.12` |
| `FI_PROVIDER_PATH` | `<oneAPI libfabric>/libfabric` | The venv ships libfabric's *core* library only, and its build differs from oneAPI's. Without the plugin path oneCCL reports `fi_getinfo error: ret -61, providers 0` |
| `CCL_ROOT` | `<venv>` | The wheel keeps the real `libccl.so` in `<venv>/lib`; the "documented" plugin path only has the static archive |
| `CCL_SYCL_COPY_ENGINE` | `0` | Forced, not overridable. `0` is oneCCL's own default, so the run takes the plain data path; `dist_allreduce.py`'s header suggests `1`, and this wrapper overrides that on purpose |
| `ZE_AFFINITY_MASK` | the resolved id list | Pins the cards; `torch.xpu.set_device(LOCAL_RANK)` then maps rank *n* to the *n*-th masked device |

`CCL_ATL_TRANSPORT` is unset first, so a stray `=mpi` cannot break a `torchrun`
launch. `FI_LIB_DIR` overrides the oneAPI libfabric location if it moves.

If you run `dist_allreduce.py` by hand instead of through the wrapper, you need
the first three variables yourself — that is the whole reason this script exists.

## Reading the output

```
Results (500 steps):
  avg: 812.4 ms/step
  p50: 810.1 ms/step
  p99: 980.2 ms/step
  total: 406.2s
```

- **`avg` / `p50`** are the per-step cost of `calls_per_step` reductions. Compare
  runs against each other, not against an absolute expectation.
- **`p99`** is the one that matters for a hang hunt: a step count where p99 is
  many multiples of p50 is a fabric stall, not noise.
- The `gbps` figure printed every 50 steps uses a **4-byte** element size while
  the tensor is bf16 (2 bytes). Either it counts send+recv (2 × 2) or it
  overstates by 2×. Treat it as a relative trend only, not an absolute
  bandwidth claim.

## Exit codes and hangs

| Exit | Meaning |
|---|---|
| `0` | All phases completed |
| `124` | Hit `--timeout`. Reported as **a hang** — this is the failure the test hunts, so it gets its own message and the last log line |
| other | `torchrun`'s own failure; the log has the traceback |

`dist_allreduce.py` has no internal watchdog: if the collective stops making
progress, the process waits forever. Always run it through this wrapper (or an
outer `timeout`) rather than directly.

## Card policy

**XPU 0–15 are refused.** Ids default to 16 and up, and any id below 16 exits
before a server or process starts. On a host with fewer cards, pass `--gpus`
explicitly with ids that exist.

The wrapper also warns (and continues) when a requested card is already held,
listing the pids: another workload distorts the numbers but does not invalidate
the hang check.

## Safety

8-rank sustained `all_reduce` is the exact shape that once dropped a PCIe root
port to **Gen1 permanently**, recoverable only by a power cycle — see
[`vllm_tp8_pcie_downgrade.md`](vllm_tp8_pcie_downgrade.md). That is why
`--ranks` defaults to 4. Raise it deliberately, and check the link state
afterwards if you do.

## Kernel log

After every run the wrapper prints the kernel-log lines produced **during that
run** (it records the log length before starting), matching:

```
DEVICE_LOST | exec queue reset | xe ... wedged/reset | guc timeout/fail
```

`dmesg` is restricted on this host, so it tries a cached `sudo -n` and otherwise
prints the command to run by hand. The wrapper never prompts and never stores a
password.

## Known rough edges in `dist_allreduce.py`

Not fixed here, because changing them would change what the test measures:

- **Device is set after `init_process_group`**, so every rank prints
  `device used by this process is currently unknown ... can potentially cause a
  hang if this rank to GPU mapping is incorrect`. With `ZE_AFFINITY_MASK` the
  mapping is right, but the warning is legitimate.
- **No timeout** (see above).
- **Bandwidth formula uses 4 bytes** for a bf16 tensor (see above).
- Its header says `--nproc-per-node=4` but the code warns unless `WORLD_SIZE == 8`,
  and it shows four separate `export ZE_AFFINITY_MASK=...` lines, which is the
  4-node × 8-rank layout rather than a single-host one.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `undefined symbol: urKernelSetArgLocal, version LIBUR_LOADER_0.12` | A newer oneAPI's `libur_loader` outranks the venv's | Run through the wrapper; by hand, export `LD_LIBRARY_PATH=<venv>/lib` |
| `fi_getinfo error: ret -61, providers 0` then `onecclCommInitRank failed with code 11` | libfabric found no provider plugins | Export `FI_PROVIDER_PATH=<oneAPI libfabric>/libfabric` |
| `torch.xpu.is_available() is False` on a healthy driver | Same loader problem as the first row | Same fix |
| `--gpus lists 2 device(s) but --ranks is 4` | The two must agree | Make the list length equal `--ranks` |
| `XPU N is not allowed` | Id below 16 | Use 16 and up |
| Runs but reports `xccl backend unavailable` | oneCCL not loadable from the venv | Check `CCL_ROOT` and that `<venv>/lib/libccl.so` exists |
| Timeout with no progress and a clean kernel log | The collective stalled without a driver fault | That is a valid reproduction; keep the log |

## Measured on this host

32× Arc Pro B60. Smoke runs only — these check the plumbing, they are not
benchmarks:

| Setup | Result |
|---|---|
| 4 ranks, XPU 16-19, 2048×2048, 5 steps × 2 calls | correctness PASSED, avg 1.5 ms/step |
| 4 ranks, XPU 16-19, 1024×512, 2 steps × 1 call | avg 0.2 ms/step |
| `--timeout 4` on a longer run | exit 124, hang message printed |
| kernel log after the runs | no `xe` / `GuC` events |

For reference, the five `exec queue reset` lines present in the kernel log came
from `0000:10:00.0` (renderD128 = **XPU 0**) at 06:14, hours before these runs
and on a card this test never touches.
