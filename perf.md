# MiniMax-H3 performance & quality log

Measured 2026-10-22 on the Intel XPU host. Two independent runtimes, same model,
same prompt, same reference image, same seed — so the tables isolate runtime,
precision and step-count effects.

- GPU: Intel Arc Pro B60, 23.9 GiB each (`xpu:0` for ComfyUI, `xpu:0-1` for vLLM-Omni)
- oneAPI 2026.1, Level Zero, driver as shipped on Ubuntu 26.04
- ComfyUI 0.42.0 in `.venv` (torch 2.14.0+xpu) — see `h3-test.sh`
- vLLM-Omni 0.32.0rc1 / vLLM 0.32.0 (torch 2.14.0+xpu) — see `vllm-omni-run.sh`
- diffusers 0.41.0 / torch 2.14.1+xpu in the 5-stage harness at
  `~/work/simon/h3/minimax-h3-diffuser` (layer-streamed, one card)
- Weights: Ref2VA partition, `~/work/models/MiniMax-H3/Ref2VA` (135 GiB, BF16)
- Turbo LoRA: `minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors` (rank 128)

## Test conditions

| parameter | value |
|---|---|
| task | `ref2va` (reference image → video + audio) |
| prompt | "An empty village road in the low morning sun before anyone comes. Photorealistic live-action film look, natural daylight." |
| reference image | `h3_input/assets/03-村口-实景.png` |
| seed | 1001 |
| frames | 124 @ 24 fps (≈5.17 s) |
| audio | 32 kHz stereo AAC, produced by both runtimes |

## Summary: what each runtime actually loaded

One row per runtime, measured on the same box, prompt, reference image and seed:

| runtime | weights loaded | quantization | GPUs | VRAM/card | GPU freq% | power mean/peak | output |
|---|---|---|---|---:|---:|---:|---|
| ComfyUI 0.42 | `minimax_h3_ref2va_pruned_int8_convrot` (21 GB) + `qwen3vl_32b_..._int8_convrot` (27 GB) | **INT8 + ConvRot**, pre-quantized single-file (ComfyUI format) | 1 | 21.4 GB | 97.1 % | 139 / 164 W | **usable** |
| diffusers 0.41 | full BF16 release, root modular layout (`transformer_ref` 62 GB + `text_encoder` 63 GB) | **BF16**, no quantization | 1 | 17.6 GiB | 97.2 % | 127 / 160 W | **usable** |
| vLLM-Omni 0.32 | `Ref2VA/` partition, BF16 on disk (62 GB DiT + 63 GB encoder) | **FP8 online** on DiT + text-decoder linears; vision tower, VAEs and norms stay BF16 | 2 (TP2) | 14.7 GiB/rank | 87.5 / 90.2 % | 128 / 178 W per card | **usable** |

Idle draw on these cards is **34 W**. Means include the model-load phase, not just denoising.

How to read the utilization columns:

- **GPU freq% is a weak signal.** The `gtidle` idle-residency counter is frozen on this host, so
  `gpu_monitor.py` falls back to `cur_freq` against its min/max window. It reads ~97 % for any
  card that is clocked up, and cannot distinguish real EU occupancy from waiting on memory.
- **Power is the discriminating metric.** ComfyUI sustains the highest average draw (139 W on one
  card), diffusers 127 W on one card, and vLLM-Omni 128/125 W on each of two cards. Diffusers and
  vLLM-Omni both stream weights over PCIe every step (layer streaming / DLO), so part of each
  step is spent transferring rather than computing; that is consistent with their lower average
  draw despite higher peak (178 W) on the vLLM-Omni cards.
- Neither proxy is a hardware counter. `xpu-smi` cannot read telemetry on this host, so true EU
  utilization is **not measured** — treat these as activity indicators, not utilisation figures.

Extra configurations measured without power sampling: vLLM-Omni BF16 uses 18.7 GiB/rank (vs
14.7 for FP8) at 375 s vs 338–349 s, and the base model at 50 steps uses 12.4 GiB/rank.

## ComfyUI

Single card, INT8 + ConvRot single-file checkpoints (DiT 20.97 GB, text encoder
27.14 GB), 8-step turbo LoRA, `res_multistep` / `beta`, CFG 1.0.

**On XPU, INT8 is a storage format only — the matmul runs in BF16.** `comfy_kitchen`
reports no native quantized kernels on Intel XPU (`comfy/model_management.py:2101`
hardcodes `supports_int8_compute() -> False` for `is_intel_xpu()`, so
`get_disabled_quant_formats()` disables every format and `comfy/ops.py:1253` sets
`_full_precision_mm = True` per layer). Each forward therefore dequantizes the
weight — `q.float() * scale`, then the ConvRot Hadamard is rotated back to the
original basis — before a BF16 `F.linear`; VAE and text encoder run FP16
(`server.log`: `Native ops:  , emulated ops: ... int8_tensorwise ...`). The gain is
21 GB instead of ~42 GB of weights and half the per-step PCIe/disk traffic under
`fast_disk=True` layer streaming; there is **no** compute gain, and the dequant is
recomputed every step. Note `torch._int_mm` does work on XPU (verified: int8
matmul → int32, correct), so this is a conservative policy in ComfyUI, not a
missing operator. This is not comparable to vLLM-Omni's `--quantization fp8`,
which is a real FP8 inference path.

| # | ComfyUI | quant | steps | shape | sampling | wall | peak VRAM | frame mean (std) | verdict |
|---|---|---|---:|---|---:|---:|---:|---|---|
| 1 | 0.39.0 (delivery package) | INT8 ConvRot | 8 | 1376×768 | 70.1 s/step | ~660 s | ~21.4 GB | 109.4 (69.6) | **correct** |
| 2 | 0.42.0 + `h3-test.sh` | INT8 ConvRot | 8 | 1376×768 | 70.1 s/step | 652 s | ~21.4 GB | 109.4 (69.6) | **correct** |
| 3 | 0.42.0, after trimming assets | INT8 ConvRot | 8 | 1376×768 | 70.1 s/step | 653 s | ~21.4 GB | 109.4 (69.6) | **correct** |

Run 2 vs run 1 are **bit-identical** (mean absolute pixel difference 0.00 across
all 124 frames), so ComfyUI 0.42 + torch 2.14.0 reproduces the 0.39 + torch
2.14.1 reference exactly.

VRAM phases from the ComfyUI log: text encoder ~15.5 GB usable, DiT ~21.4 GB,
VAE decode ~21.0 GB — each boundary returns to ~24.2 GB free, i.e. one large
model resident at a time.

## vLLM-Omni

Two cards, TP2 + distributed layerwise offload (12 resident DiT layers) + tiled
VAE patch-parallel 2. Server start-up 70–120 s, excluded from the numbers below.

### Valid configurations

| # | precision | LoRA in request | steps | flow | shape | E2E | peak/rank | mean (std) |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | FP8 online | **yes** | 8 | 6 | 1376x768 | 338-349 s | 14.7 GiB | **114.6 (68.4)** |
| 2 | BF16 | **yes** | 8 | 6 | 1376x768 | 375 s | 18.7 GiB | **114.7 (68.9)** |
| 3 | FP8 online | none (base model) | 50 | 12 | 1024x576 | 943 s | 12.4 GiB | 71.8 (71.8) |
| 4 | BF16, **one** GPU (DLO 2 layers) | **yes** | 8 | 6 | 1376x768 | 700 s | 19.7 GiB | **114.7 (69.0)** |

Row 4 is the single-card capacity path: `--num-gpus 1` with DLO at only 2 resident layers,
no `--quantization fp8`, so the DiT stays 62 GB and streams over PCIe every step. It fits
(19.7 GiB of 24) and the frame statistics are identical to the two-card BF16 row
(114.7/69.0 against 114.7/68.9), but E2E is 1.87× slower — half the compute and ~59 GB of
streaming per step.

**`--diffusion-offload-config '{"mode":"layer", ...}'` cannot be combined with the turbo
LoRA.** The recipe's own "Single GPU: blockwise capacity path" uses it, and any request that
binds an adapter then fails with `MiniMax-H3 Turbo dynamic LoRA does not support layerwise
offload` (`minimax_h3/lora.py:298`). The guard rejects `MODEL_LEVEL` and `LAYER_WISE` but not
`DISTRIBUTED_LAYER_WISE`, so single-card + LoRA must use
`--enable-distributed-layerwise-offload --dlo-resident-layers N` instead.

All three produce correct output. Rows 1-2 use the turbo adapter and its mandated
schedule; row 3 is the undistilled base model on its own schedule, so it needs about
six times the steps and comes out dimmer but structurally right.

### Harness-bug rows (adapter never bound)

| # | precision | attention | steps | flow | shape | E2E | peak/rank | mean (std) |
|---|---|---|---:|---:|---:|---:|---:|---|
| 4 | FP8 online | FLASH_ATTN | 9 | 6 | 1376x768 | 360-370 s | 14.8 GiB | 40.8 (28.7) |
| 5 | BF16 | FLASH_ATTN | 9 | 6 | 1376x768 | 371 s | 18.7 GiB | 43.0 (32.9) |
| 6 | BF16 | TORCH_SDPA | 9 | 6 | 1376x768 | 450 s | 18.4 GiB | 43.4 (32.6) |
| 7 | FP8 online | FLASH_ATTN | 2 | 12 | 1024x576 | 78-98 s | 12.8 GiB | 71.7 (70.5) |

Rows 4-6 ran the **base model on the turbo schedule** (the adapter was requested
but never activated), and row 7 is simply 2 steps of anything. All four are
under-denoised rather than broken; they are kept because "dark and blurry" is the
signature worth recognising.

Every request returned HTTP 200 with a well-formed 124-frame H.264 mp4 carrying
32 kHz stereo AAC. No `DEVICE_LOST`, no xe/GuC faults, no OOM.

> Rows marked *under-denoised* are dark and blurred. That is the signature of running the
> base model on the turbo schedule, not a container or audio failure — see the correction
> below.

## Sparse attention (SOL) on XPU

Measured 2026-10-27. Sparse attention needs `omni_xpu_kernel` plus a Kitchen build with an
`xpu` backend; the stock ComfyUI install in `.venv` has neither, so it cannot run there at
all (three CUDA-hardcoded gates: `nodes_sparse_attention.py:156`/`:232` test
`device.type != "cuda"`, and `comfy_kitchen.sol_attn_is_available()` returns False for any
non-CUDA device; the eager reference is also unusable — it materialises scores densely,
O(T²), and hard-errors past 4 GiB, while our sequence is 41,042 rows). Use Intel's published
image instead.

Environment: `intel/llm-scaler-omni:0.2.0-b2`, one Arc Pro B60 (24 GB, XPU 16), ComfyUI
0.39.0, torch 2.14.0+xpu, `omni_xpu_kernel 0.2.0b2+torch214.bmg`, `comfy-kitchen 0.2.37` +
`comfy-kitchen-xpu-runtime`, `ComfyUI-OmniXPU`. Providers activate from the node's
`prestartup_script.apply_runtime_providers()` (before torch import); a bare
`import comfy_kitchen` shows no `xpu` backend and looks like a failure when it is not.

Workflow: our 6-node JZL graph with `BlockSparseAttention` inserted between the LoRA and
`JZL_MiniMaxAssetManager` (the node's `model` input is optional, so it accepts the patched
model). In the API JSON `selection` is a `COMFY_DYNAMICCOMBO_V3` and takes **flat** keys —
`"selection": "sol-attn"` plus `"selection.tau": 1.3`; a nested dict silently drops the
input and fails with `missing 1 required positional argument: 'selection'`.
`sink_conditioning=exact_kv_and_rows` is the H3-specific mode that keeps packed
text/audio/reference rows and the target-audio query rows dense.

8-step turbo LoRA, 1376×768, 124 frames, seed 1001, 1.0 MP, `extra_tokens=256`,
`min_tokens=12288` (sequence is 41,042 rows, so sparsity engages):

| config | E2E | vs dense | frame mean | std | motion | PSNR vs dense |
|---|---:|---:|---:|---:|---:|---:|
| dense | 470.2 s | 1.00× | 109.3 | 69.7 | 11.47 | — |
| SOL `tau=1.0`, start 0.2 | 410.2 s | 1.15× | 108.8 | 69.9 | 11.10 | 20.99 dB |
| SOL `tau=1.3`, start 0.2 | 410.4 s | 1.15× | 109.2 | 69.9 | 11.18 | 19.88 dB |
| SOL `tau=2.0`, start 0.2 | 390.2 s | 1.21× | 109.2 | 69.8 | 11.15 | 18.26 dB |
| SOL `tau=1.3`, **start 0.0** | **370.2 s** | **1.27×** | 108.4 | 70.1 | 11.50 | 18.46 dB |

Sparse engages only after `start_percent`: with the default `0.2` the first three steps
(sigma 1 / 0.995 / 0.983) stay dense at ~46.2 s and the rest run at ~38 s, giving a measured
40.53 s/it; `start_percent=0.0` runs all eight steps sparse at a measured **36.97 s/it**
(4:55 sampling, 295 s) and is what lifts E2E to 1.20–1.27×. Identical configs vary by ~5%
run to run (370.2 s versus 390.2 s for the same `start=0.0` job), so treat the E2E figure as
a range and compare per-step numbers instead. `verbose=true` is the only reliable check — it
prints `sparse producer path: 41042 tokens, sinks (0, 45)/(38, 45)` when routing is live
(at `start=0.0` from the very first step) and `dense (41042 tokens): sigma … outside the
start/end window` otherwise. Note `sol_attn_chunked` is **absent** from this image's Kitchen
yet the H3 producer path still runs, so that capability name is not a usable readiness check.

Quality: every variant is a different sampling trajectory (0/124 frames identical,
18–21 dB PSNR) but the same shot — identical composition, mean/std within 1%, no visible
artefacts even at `tau=2.0`. Judge it on the clip, not on PSNR.

**The image is faster than our own venv even without sparsity**: dense 46.4 s/step /
470.2 s versus 70.1 s/step / 652 s in `.venv` at identical frame statistics
(109.3/69.7 vs 109.4/69.6), from native INT8 GEMM plus the CUTE attention adapter. Single
card here versus vLLM-Omni's TP2 — one-card ComfyUI with sparsity matches two-card
vLLM-Omni per step (40.5 vs 40.1 s) at 410 vs 375 s E2E.

vLLM-Omni has **no** Sol/SLA backend at all: `DiffusionAttentionBackendEnum` lists only
FLASH_ATTN, TORCH_SDPA, SAGE_ATTN(_3), CUDNN_ATTN, FLASHINFER_ATTN, FASTVIDEO_VSA,
VDNH3_ATTN, FLASH_ATTN_HUB(_3), TRTLLM_ATTN and RAINFUSION_ATTN, and the one sparse entry
falls back to dense on XPU — `fastvideo_vsa.py:437` `forward_xpu()` returns
`self.sdpa_fallback.forward_xpu(...)`, with `validate_available()` demanding CUDA-only
`fastvideo_kernel`/`flashinfer`.

### Two cards: two instances, not tensor parallel

ComfyUI has no TP, so the way to use a second card is a second instance. Two containers,
one `renderD144` (XPU 16) and one `renderD145` (XPU 17), each exposing exactly one device
(`Device: xpu:0` in both), each given the same SOL `tau=1.3` / `start_percent=0.0` job:

| | solo | dual |
|---|---:|---:|
| wall clock | 370.2 s for 1 clip | **390 s for 2 clips** |
| per job | 370.2 s | 380.3 s (XPU 16) / 390.2 s (XPU 17) |
| throughput | 1.00× | **1.90×** |

Each job slows by only 3–5% under concurrency, and both outputs are bit-identical to the
solo run (same 9,096,609 B, mean 108.4 / std 70.1 / motion 11.50), so the instances do not
interfere numerically. Two cards buy 1.9× throughput for 5% latency — far better than
vLLM-Omni's TP2+DLO, which needs both cards to run a single request at roughly the same
per-step cost as one card here.

### SOL vs SLA vs VSA

Measured 2026-10-29, one Arc Pro B60 per run in the `llm-scaler-omni:0.2.0-b2` image. Each
method is paired with a dense run of the *identical* graph, weights, step count and seed, so
the per-step delta is attributable to the sparse patch alone.

| method | task | steps | per-step dense → sparse | speedup | sparse steps |
|---|---|---:|---|---:|---:|
| SOL `tau=1.3`, start 0.0 | ref2va | 8 | 46.37 → 36.97 s | **1.25×** | 8/8 |
| SOL `tau=1.3`, start 0.0 | **t2va** | 8 | 42.04 → 32.27 s | **1.30×** | 8/8 |
| SLA `keep_percent=15` | fl2v | 4 | 45.92 → 37.06 s | **1.24×** | 4/4 |
| VSA `keep_percent=10`, start 0.2 | fl2v | 8 | 44.87 → 42.48 s | **1.06×** | 6/8 |

Packed sequence length and sink rows per task, from the node's own verbose output:
ref2va 41,042 rows with sinks `(0, 45)/(38, 45)`; fl2v 40,694 with `(0, 40)/(32, 40)`; t2va
38,622 with `(0, 7)/(0, 7)` — no reference rows, so only 7 prefix rows. t2va is the cheapest
per step *and* the best sparse speedup, which says the gain does not depend on the
conditioning layout.

The three are **not** cross-comparable on quality: SOL ran on ref2va with the turbo LoRA,
SLA on fl2v with the Turbo-SLA LoRA (the task it was trained for), VSA on fl2v with the
FastH3 V2 checkpoint. Only the per-step ratio is a fair comparison. Each against its own
dense control: SOL 18.5 dB, SOL-t2va 12.2 dB, SLA 24.08 dB, VSA 15.74 dB PSNR, with frame
mean/std within a few percent throughout — judge the clips, not the PSNR. On t2va the sparse
run came out brighter and higher contrast than dense (93.1/74.0 vs 85.7/65.1) with a
different composition; both clips are clean and on-prompt.

VSA's gain is much smaller, and its own log says why: `VSA tiles (44864 padded rows, 41
prefix tiles)` — the 40,694 real rows are padded to 44,864 (+10% work) and the learned
coarse branch costs more than the exact blocks it saves. Its `keep_percent=10` is also the
value the checkpoint was trained at, so unlike SOL's `tau` it cannot be tuned down without
leaving the trained policy.

#### Traps hit while measuring (each cost real runs)

- **`fasth3_omniref_8step_pruned_int8_convrot` cannot be loaded by ComfyUI's H3 nodes.** Its
  metadata reads `source: transformer_ref` — it is the reference branch's *second* DiT, with
  a different AdaLN basis and a PDD 32-step schedule. Both the JZL ref2va graph and a native
  r2v graph fail identically with `The size of tensor a (32) must match the size of tensor b
  (1024)`. Use `fastvideo_fasth3_8step_v2_pruned_int8_convrot` from
  `FastVideo/FastVideo-FastH3-Comfy`, which is the file the image's own FastH3 template names.
- **The template's settings beat the recipe table for a given checkpoint.** The FastH3
  template uses shift **10/3**, 8 steps and `res_multistep` for the 8-step V2 checkpoint,
  while `docs/SPARSE_ATTENTION.md`'s VSA row (4 steps, euler, 12/3) describes a different,
  4-step artifact.
- **`fuser` saying a card is busy proves nothing.** On this host one process holds open
  handles on all 32 render nodes while running on exactly one. Four consecutive OOMs came
  from another tenant's `h3_gpu` job on XPU 16; the giveaway is an OOM signature that does
  not move — 12.69 GiB allocated, 51 MiB free, 126 MiB requested — however far you cut the
  resolution. Look for an active workload, not an open fd.
- **The JZL node swallows per-segment exceptions** and returns an empty bus, so ComfyUI
  reports `status=success` for a run that produced nothing. Always confirm the mp4 exists and
  the container log shows `采样完成` / `✅ 解码完成`.

### Fewer steps: the 4-step turbo LoRA on ref2va

`minimax_h3_ref2v_turbo_4step_v1.0_768p_comfyui_bf16` on the same JZL ref2va graph,
same sampler (`res_multistep`/`beta`), same seed, 4 steps instead of 8:

| config | per-step | E2E | frame mean | std | motion |
|---|---:|---:|---:|---:|---:|
| 8-step dense | 46.37 s | 470.2 s | 109.3 | 69.7 | 11.47 |
| 8-step SOL | 36.97 s | 370.2 s | 108.4 | 70.1 | 11.50 |
| 4-step dense | 46.31 s | 270.1 s | 98.3 | 69.8 | 7.10 |
| **4-step SOL** | **37.91 s** | **240.1 s** | 102.8 | 87.9 | 7.66 |

4-step SOL is the fastest ref2va configuration measured (240 s against 370 s for 8-step
SOL, 1.54× end to end), and the sparse patch still contributes 1.22× per step. The cost is
visible: **temporal motion drops from ~11.5 to ~7.1–7.7** (−35%) and the mean level drops
~5%, which is the "4 steps loses fast motion" effect the turbo authors warn about. The clip
itself stays sharp and on-prompt; the higher std is the scene's contrast, not artefacts. Both
clips carry AAC audio.

### Fewer steps on t2va

The same 4-step fl2v artifact also serves T2VA (the recipe table lists it as "T2VA / FL2VA"),
and the t2v graph is the i2v subgraph with no first_frame, so only the adapter and step count
change:

| config | per-step | E2E | frame mean | std | motion |
|---|---:|---:|---:|---:|---:|
| 8-step dense | 42.04 s | 410.2 s | 85.7 | 65.1 | 9.15 |
| 8-step SOL | 32.27 s | 330.1 s | 93.1 | 74.0 | 9.17 |
| 4-step dense | 41.94 s | 240.1 s | 94.2 | 80.8 | 4.56 |
| **4-step SOL** | **33.11 s** | **210.1 s** | 93.6 | 84.3 | 7.15 |

Per-step cost is identical at 4 and 8 steps, so the step count is a pure end-to-end lever:
330 → 210 s (1.57×), with the sparse patch still worth 1.27× per step. The cost is larger than
on ref2va: **t2va dense loses half its motion** (9.15 → 4.56) and prompt adherence loosens —
an "empty village road" prompt comes back with a car and power lines at 4 steps but not at 8.
Interestingly the 4-step *sparse* run keeps more motion than the 4-step dense one (7.15 vs
4.56), so at this step count the sparse patch changes the trajectory noticeably; judge by
watching, not by the numbers.

## Where the time goes, and what each optimisation changes

Everything below is measured on this host. The point of the layering is that each optimisation
attacks a *different* term, so their speedups multiply while their costs do not overlap.

```
total   = fixed overhead (TE encode + VAE decode + mux, 40-100 s)  +  steps x per-step
per-step = attention + FFN/linear + norm/RoPE + weight movement (DLO / offload)
```

| optimisation | term it changes |
|---|---|
| JZL staged offloading | makes the run fit at all — neither the multiplier nor the per-step |
| OmniXPU native kernels | the constant factor inside a step |
| turbo LoRA | **the step count** |
| SOL / SLA / VSA | **attention**, inside a step |
| int8 / FP8 / bf16 | **FFN and weight bandwidth**, inside a step |

### Layer 1 — JZL staged offloading: it buys capacity, not speed

The three components are individually large and never needed at the same time, but ComfyUI
holds them together by default:

| component | int8 | bf16 |
|---|---:|---:|
| DiT | 19.5 GiB | 37.5 GiB |
| text encoder (Qwen3-VL 32B) | 25.3 GiB | 48 GiB |
| video + audio VAE | 4.8 + 0.6 GiB | same |

The JZL node owns the whole graph, so it knows the phase boundaries and unloads between them:

```
1 encode : CLIP + video VAE -> encode -> unload
2 sample : DiT + video VAE + audio VAE -> N steps
3 decode : video VAE + audio VAE -> video + audio
```

Its log shows it waiting for VRAM to settle before each phase
(`[JZL阶段] TE编码前 显存稳定：可用 24480 MB`). A hand-built native graph doing the same job
holds the TE and the DiT together and dies:

```
XPU out of memory. Tried to allocate 126 MiB. 23.91 GiB total, 51 MiB free
12.69 GiB allocated by PyTorch
```

The price is a reload per phase — that is the 40–100 s fixed overhead. It does **not** change
the per-step cost: JZL dense 46.37 s against a native graph's 45.92 s on fl2v.

### Layer 2 — OmniXPU native kernels: same maths, faster operators

This is the whole of the 1.51× gap between our `.venv` and Intel's image, with identical output:

| | per-step | frame mean/std |
|---|---:|---|
| `.venv`: int8 weights, dequantise to BF16, BF16 GEMM | **70.1 s** | 109.4 / 69.6 |
| image: native int8 GEMM + CUTE attention | **46.37 s** | 109.3 / 69.7 |

Stock ComfyUI hardcodes `supports_int8_compute() -> False` for Intel XPU, so every linear
dequantises (`q.float() * scale`, then the ConvRot rotation is undone) before a BF16 matmul.
The image's Kitchen `xpu` backend runs the int8 GEMM directly with int32 accumulation, routes
our attention shape through a CUTE kernel (`attention CUTE_BHLD_D128: heads=56 q=41042`), and
adds native RMS-RoPE / GroupNorm / AdaLN plus an H3-specific segmented RMS modulation patch.

### Layer 3 — turbo LoRA: fewer steps, nothing else

The adapter is distilled to land on the *teacher's converged output* rather than reproduce its
trajectory, so 8 steps (or 4) reach roughly where 50 Euler steps would. It carries its own
sampling contract in the filename, which the server enforces:

| artifact | task | steps | flow_shift |
|---|---|---:|---:|
| `ref2v_turbo_8step_v1.0_768p` | Ref2VA | 8 | 6 |
| `ref2v_turbo_4step_v1.0_768p` | Ref2VA | 4 | 6 |
| `fl2v_turbo_4step_v0.1` | T2VA / FL2VA | 4 | 12 |

Its effective strength is only 1/16: every artifact is rank 128 with `alpha=8`, and the delta is
applied at `scale × alpha / rank`. It nudges the base weights rather than replacing the model.

Measured on ref2va with graph, sampler and seed held fixed:

| steps | per-step | E2E | motion |
|---|---:|---:|---:|
| 8 (turbo) | 46.37 s | 470.2 s | 11.47 |
| 4 (turbo) | 46.31 s | 270.1 s | **7.10** |

Per-step is unchanged; only the multiplier moves. The cost is motion: −35% on ref2va and −50%
on t2va, where prompt adherence also loosens.

### Layer 4 — SOL / SLA / VSA: cheaper attention inside a step

All three split the sequence into 64-token blocks and attend only a routed subset of key blocks
exactly.

| method | selection | needs | measured |
|---|---|---|---:|
| SOL | training-free, per-block σ threshold `tau`; unselected blocks each contribute one pooled term and **stay in the softmax denominator** | nothing beyond base + turbo | **1.25–1.30×** |
| SLA | fixed top-k% of key blocks | a LoRA distilled against that pattern | **1.24×** |
| VSA | 3D video-cube tiling + a learned coarse-gate branch | a FastH3 checkpoint with `to_gate_compress` | 1.06× |

SOL's design is why it survives aggressive settings: it approximates rather than truncates, so
`tau=2.0` (≈2.7% of blocks exact) still leaves the frame statistics intact.

The reason the ceiling is ~1.3× rather than 5× is that attention is only part of a step. From
the SOL delta on ref2va, attention is at most 9.4 s of the 46.37 s step (~20%); the remaining
~37 s is FFN/linear, normalisation and weight movement.

VSA's 1.06× is explained by its own log — `VSA tiles (44864 padded rows, 41 prefix tiles)`:
40,694 real rows are padded to 44,864 (+10% work) and the learned coarse branch costs about
what the exact blocks save.

### Layer 5 — precision: FFN and bandwidth

| configuration | per-step | cards |
|---|---:|---:|
| int8 native (image) | 46.37 s | 1 |
| int8 storage + BF16 compute (`.venv`) | 70.1 s | 1 |
| FP8 online (vLLM-Omni) | 37.06 s | 2 |
| BF16 full partition (vLLM-Omni) | 40.1 s | 2 |

int8 is not only a memory saving here: with a native GEMM it also removes the dequantise step,
which is why it beats the dequantising `.venv` path by 1.51×. Whether native int8 beats native
bf16 GEMM is still open — that is what the `ref2va_pruned_bf16` comparison is for.

### The layers multiply

The fastest ref2va configuration is exactly this stack:

```
4-step turbo + SOL, one card:
  ~89 s fixed overhead + 4 x 37.91 s = 240 s
  turbo   : 2x   (8 -> 4 steps)
  SOL     : 1.22x per step
  int8    : 1.51x per step
  JZL     : makes it fit
```

Against 8-step dense on `.venv` (652 s) that is 2.7×, not the 3.7× the factors suggest, because
fixed overhead does not shrink with the optimisations and the 4-step run pays its motion cost.

### Not measured

- No per-operator profiler breakdown. The "attention ≤20% of a step" figure is inferred from the
  SOL delta, not from `torch.profiler`.
- No same-graph bf16 control yet (the 40.2 GB `ref2va_pruned_bf16` download is in flight).
- No 6-step run: the turbo artifacts are distilled at 8 and 4, so 6 would be extrapolation.

## All measured configurations on B60

Everything measured on this host, one row per configuration. `step` is the per-denoise-step
time from the runtime's own progress bar (ⓜ) or derived from E2E minus fixed overhead (ⓓ);
`E2E` is the wall time for one 1376×768 / 124-frame clip at seed 1001. Fixed overhead
(TE encode + VAE decode + mux) is 40–100 s per run and is why E2E never equals steps × step.

### ComfyUI + `llm-scaler-omni:0.2.0-b2`, single card, pruned int8_convrot DiT

| # | task | graph | optimisation | steps | step | sampling | E2E | frame mean/std | note |
|---:|---|---|---|---:|---:|---:|---:|---|---|
| 1 | ref2va | JZL | turbo 8-step | 8 | **46.37** ⓜ | 371 s | 470.2 | 109.3 / 69.7 | dense baseline |
| 2 | ref2va | JZL | SOL τ1.0, start 0.2 | 8 | — | — | 410.2 | 108.8 / 69.9 | step not sampled |
| 3 | ref2va | JZL | SOL τ1.3, start 0.2 | 8 | 40.53 ⓜ | 324 s | 410.4 | 109.2 / 69.9 | first 3 steps dense |
| 4 | ref2va | JZL | SOL τ2.0, start 0.2 | 8 | — | — | 390.2 | 109.2 / 69.8 | step not sampled |
| 5 | ref2va | JZL | **SOL τ1.3, start 0.0** | 8 | **36.97** ⓜ | 296 s | 370.2 | 108.4 / 70.1 | 8/8 steps sparse |
| 6 | ref2va | JZL | turbo **4-step** | 4 | 46.31 ⓜ | 185 s | 270.1 | 98.3 / 69.8 | motion 7.10 |
| 7 | ref2va | JZL | **turbo 4-step + SOL** | 4 | **37.91** ⓜ | 152 s | **240.1** | 102.8 / 87.9 | motion 7.66 |
| 8 | ref2va | JZL | SLA LoRA (off-spec) | 4 | 46.67 ⓜ | 187 s | 290.1 | 116.4 / 73.0 | fl2v LoRA on ref2va |
| 9 | ref2va | JZL | SLA keep 15 (off-spec) | 4 | 38.22 ⓜ | 153 s | 250.1 | 113.8 / 68.3 | timing only |
| 10 | fl2v | native i2v | SLA 4-step, dense | 4 | 45.92 ⓜ | 184 s | 260.1 | 128.4 / 74.7 | in-spec |
| 11 | fl2v | native i2v | **SLA keep 15** | 4 | **37.06** ⓜ | 148 s | **190.1** | 130.3 / 73.6 | PSNR 24.1 dB |
| 12 | fl2v | native i2v | FastH3 V2, dense | 8 | 44.87 ⓜ | 359 s | 420.2 | 122.5 / 78.4 | V2 checkpoint |
| 13 | fl2v | native i2v | VSA keep 10, start 0.2 | 8 | 42.48 ⓜ | 340 s | 430.2 | 128.9 / 76.1 | +10% tiling padding |
| 14 | t2va | native t2v | turbo 8-step | 8 | 42.04 ⓜ | 336 s | 410.2 | 85.7 / 65.1 | |
| 15 | t2va | native t2v | **SOL τ1.3, start 0.0** | 8 | **32.27** ⓜ | 258 s | 330.1 | 93.1 / 74.0 | best speedup, 1.30× |
| 16 | t2va | native t2v | turbo 4-step | 4 | 41.94 ⓜ | 168 s | 240.1 | 94.2 / 80.8 | motion 4.56 |
| 17 | t2va | native t2v | **turbo 4-step + SOL** | 4 | **33.11** ⓜ | 132 s | **210.1** | 93.6 / 84.3 | motion 7.15 |

### Our own `.venv` ComfyUI 0.42.0, single card

| # | task | quantisation | optimisation | steps | step | E2E | frame mean/std |
|---:|---|---|---|---:|---:|---:|---|
| 18 | ref2va | int8 weights, **BF16 compute** | turbo 8-step | 8 | **70.1** ⓜ | 652.0 | 109.4 / 69.6 |

Same weights and same frame statistics as row 1 but **1.51× slower per step** — the gap is the
missing native int8 GEMM and CUTE attention, not the model.

### vLLM-Omni, **two** cards (TP2 + DLO 12 layers)

| # | task | weights | optimisation | steps | step | E2E | frame mean/std |
|---:|---|---|---|---:|---:|---:|---|
| 19 | ref2va | BF16 partition | turbo 8-step | 8 | 40.1 ⓓ | 375 | 114.7 / 68.9 |
| 20 | ref2va | FP8 online | turbo 8-step | 8 | **37.06** ⓜ | 349.8 | 114.6 / 68.4 |
| 21 | ref2va | BF16 | base, euler | 20 | 38.5 ⓜ | 824.0 | 55.9 / 30.5 ✗ |
| 22 | ref2va | BF16 | base, res_multistep | 20 | 36.94 ⓜ | 789.9 | 61.1 / 34.9 ✗ |
| 23 | ref2va | BF16 | base, euler | 50 | 38.0 ⓓ | 1902.0 | 71.9 / 35.7 |
| 24 | ref2va | FP8 | base @1024×576 | 50 | 17.8 ⓓ | 942.7 | 71.8 / 73.3 |

### What the table says

- **Per-step cost does not depend on the step count.** The same graph and weights give
  41.94 vs 42.04 s at 4 and 8 steps (t2va) and 37.91 vs 36.97 s (ref2va). Steps are purely an
  end-to-end multiplier; quality is what they buy.
- **Sparse attention is worth 1.22–1.30× per step** on SOL and SLA, independent of task, and
  only 1.06× on VSA (its tiling pads 40,694 rows to 44,864).
- **One card of ComfyUI matches two cards of vLLM-Omni**: 37.06 s/step both, but vLLM-Omni
  needs TP2 + DLO to get there.
- The fastest end-to-end clips are fl2v + SLA (190 s) and t2va 4-step + SOL (210 s); the
  fastest ref2va is 4-step + SOL (240 s).

## diffusers

Single card, BF16, **root modular layout** (`~/work/models/MiniMax-H3`, not the
`Ref2VA/` partition). Weights are streamed layer by layer: the DiT is built on
`meta` and each of its 50 blocks is uploaded just before use, so peak VRAM stays
at 17.6 GiB while RSS holds the 62 GiB checkpoint.

Run through `~/work/simon/h3/minimax-h3-diffuser` (5 stages: te → dit →
vae_video → vae_audio → media), `--ops official`, seed 1001, 1376×768, 124 frames:

| stage | wall | note |
|---|---:|---|
| te | 28.2 s | 51 layers streamed, 534 tokens → `(1, 7710, 5120)` bf16 |
| dit | **839.7 s** | 7 evals at 116.9 s/eval, 50 blocks streamed per eval |
| vae_video | 45.2 s | |
| vae_audio | 4.7 s | |
| media | 1.9 s | x264 + AAC mux via system `ffmpeg` |
| **total** | **~920 s** | peak VRAM 17.6 GiB, RSS 66.5 GB |

Output: 124 frames, 1376×768, 5.17 s, AAC 32 kHz stereo, frame mean **121.2**,
std **65.7** — **correct**.

## Quantization comparison

Same shape, same 9-step LoRA schedule, only precision changed:

| precision | DiT weights | E2E | peak/rank | Δ vs FP8 | frame mean (std) |
|---|---|---:|---:|---:|---|
| FP8 online | ~half, quantized at load | 338–349 s | 14.7 GiB | — | **114.6 (68.4)** |
| BF16 | full checkpoint | 375 s | 18.7 GiB | +27 % memory, +7–11 % time | **114.7 (68.9)** |
| BF16 (diffusers) | full checkpoint, layer-streamed | ~920 s | 17.6 GiB on 1 card | n/a, different runtime | 121.2 (65.7) |
| INT8 ConvRot (ComfyUI) | 21 GB single file | 652 s | ~21.4 GB on 1 card | n/a, different runtime | 109.4 (69.6) |

Reading this:

- FP8 is the better choice: **~27 % less per-rank memory, 7–11 % faster**, and
  visually indistinguishable from BF16 at the same schedule.
- All four runtimes now agree on quality once configured correctly (109–121 mean).
- INT8 ConvRot has no diffusers or vLLM-Omni equivalent. Diffusers can only do
  BF16 here: `torchao`/`bitsandbytes`/`quanto`/`sdnq` are all absent, GGUF has no
  H3 file, and diffusers' `AutoRoundQuantizer` refuses on-the-fly quantization
  (`requires_calibration = True`). The harness supports `int8_convrot` when the
  checkpoint's `config.json` declares it — ours does not (full BF16 release), so
  INT8 stays ComfyUI-only.

## Step comparison

| runtime | steps | shape | E2E | per-step |
|---|---:|---|---:|---|
| ComfyUI | 8 | 1376×768 | 652 s | **70.1 s/step** (from tqdm, 561 s of sampling) |
| vLLM-Omni FP8 | 2 | 1024×576 | 78–98 s | not separable (fixed encode/decode dominates) |
| vLLM-Omni FP8 | 9 | 1376×768 | 360–370 s | not separable |

Only ComfyUI gives a clean per-step figure, because both of its runs share one
shape and one step count. A proper vLLM-Omni step sweep (same shape, 2/4/9 steps)
was **not run** — see *Not measured* below; do not derive a slope from rows 5–6
against rows 1–2, because the shape differs.

The turbo artifacts pin the schedule: 8-step LoRA ⇒ 9 sigma points,
`flow_shift=6`; 4-step LoRA ⇒ 5 sigma points, `flow_shift=6`. Requesting
anything else is rejected by the server, so "steps" is not a free knob while a
LoRA is loaded.

## Speed summary

- **vLLM-Omni is ~1.8× faster end-to-end** than ComfyUI for the same 5 s shot
  (360–370 s vs 652 s) — while running *one more* sampling step.
- That speed comes from **two cards**, so per-card efficiency is ~0.9× of
  linear: layerwise offload streams non-resident DiT blocks over PCIe every step,
  which is the expected bottleneck on this topology.
- **ComfyUI needs 21.4 GB on one card**; vLLM-Omni needs 14.8 GiB/rank on two
  (FP8) or 18.7 GiB/rank (BF16). For fitting a 135 GiB partition onto 24 GiB
  cards, the vLLM-Omni configuration has more headroom per card.
- Raising `--dlo-resident-layers` above 12 is the obvious latency knob at FP8:
  it halves DiT residency and the run peaks at 14.8 GiB of 23.9 GiB, leaving
  ~9 GiB unused. **Not yet measured.**

## Quality verdict

| path | usable output? | evidence |
|---|---|---|
| ComfyUI 0.39 / 0.42, INT8 ConvRot, 8 steps | **yes** | correct village-road clip; frame std 69.6; bit-identical across versions |
| vLLM-Omni, FP8 or BF16, **8 steps with the turbo LoRA activated in the request** | **yes** | corrected: frame mean 114.6, std 68.6 |
| vLLM-Omni, base model, 50 steps, `flow_shift=12` | **yes** | frame mean 71.8, std 71.8 (dimmer grade, correct structure) |
| diffusers 0.41, BF16, layer-streamed, 1 card | **yes** | frame mean 121.2, std 65.7 |
| vLLM-Omni, **8–9 steps without the adapter actually bound** | **no** | dark blurred crop; frame mean ~41, std ~30 |

### Correction: vLLM-Omni is not broken

An earlier version of this document concluded that vLLM-Omni's XPU diffusion path was at
fault. That was wrong, and the error was in the harness, not in vLLM-Omni.

`--lora-path` on the server only **preloads** an adapter. Activation is per request, through
the request's own `lora` field:

```bash
-F 'lora={"name":"h3-8step","path":"/abs/path/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors"}'
```

Without that field every request silently ran the **base model** — but with the turbo
schedule (8 steps, `flow_shift=6`), which is far too few steps for the undistilled model.
Dark, blurred video was the expected result. That is why precision, attention backend, and
2-vs-9 steps all appeared to make no difference: none of them was the variable.

Two facts that would have caught it earlier:

- No `DiffusionLoRAManager` / `FastH3 adapter active` line appears in the server log unless
  an adapter is actually bound.
- The artifact also enforces its own step count: `num_inference_steps` must equal the
  artifact's `denoise_steps` (8 for the 8-step file). The recipe table's "9 sigma points"
  is the implementation's 8 evaluations.

`vllm-omni-run.sh` now sends the `lora` field on every request, uses 8 steps, and raises
`VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` so runs longer than 600 s do not die with
`RPC call to execute_model timed out`.

### Operational hazard found the hard way

Loading the whole 135 GiB partition across 16 cards at once with
`device_map="balanced"` **wedged three GPUs** (`xe ... device wedged, needs recovery`, GuC
queue timeouts) and left the stack unusable until a reboot. That matches the delivery
manifest's `UR_RESULT_ERROR_DEVICE_LOST` warning for concurrent ~20 GB loads. Streaming one
block at a time on a single card (the diffusers path above) peaked at 17.6 GiB with no driver
events at all — prefer that shape.

## Reproduce

```bash
# ComfyUI (working path) — ~11 min on one card
./h3-test.sh 0

# diffusers (working path) — ~15 min on one card, layer-streamed
cd ~/work/simon/h3/minimax-h3-diffuser
H3_WEIGHTS_ROOT=~/work/models/MiniMax-H3 \
H3_RUNS_ROOT=/home/lm/work/paul/uv_scaffolding_xpu/diffusers-output/runs \
  .venv/bin/python run.py \
    --config-dir /home/lm/work/paul/uv_scaffolding_xpu/diffusers-output/cfg \
    shot01-diffusers xpu all --ops official

# vLLM-Omni — ~6 min on two cards (adapter sent in the request)
./vllm-omni-run.sh run --gpus 2 --precision fp8 --lora 8step
./vllm-omni-run.sh run --gpus 2 --precision bf16 --lora 8step
./vllm-omni-run.sh run --profile smoke            # 2 steps, no LoRA

# raw numbers for a run
grep -iE 'x-inference-time-s|x-peak-memory-mb' vllm-omni-output/last_headers.txt
```

Frame statistics in the tables come from decoding every frame and taking the
mean and standard deviation of pixel values: a well-exposed image with structure
lands around mean 100–120 / std 65–75, a flat or dark failure well below that.

## Not measured

- **vLLM-Omni step sweep** with the adapter bound at a fixed shape (4 / 8 / 16) —
  only 8 steps is mandated by the artifact, so the useful sweep is base-model
  steps at `flow_shift=12` (e.g. 20 / 35 / 50).
- **`--dlo-resident-layers` 16–20 at FP8** — the most promising latency knob,
  given ~9 GiB/rank sits unused.
- **`fl2va` / T2VA** — the other 135 GiB partition is on disk but never served.
- **ComfyUI at other step counts** — only the 8-step turbo LoRA is available;
  the 4-step artifact was not fetched.
- **Concurrent requests** — all runtimes served one request at a time.

## 2026-10-31 — single Intel Arc Pro B70 (this host)

Different box from the tables above: **one** B70 (`0xe223`, 30.3 GiB, `xpu:0`;
`/dev/dri/renderD129` — 128 is the iGPU), Ubuntu 26.04, oneAPI 2026.2,
torch 2.14.0+xpu. ComfyUI 0.42.0 + `omni_xpu_kernel` 0.2.0b2 (`int8 enabled: True`),
vLLM-Omni 0.32.0rc1 built from source into `.venv-omni`. Same prompt, seed 1001.

| runtime | mode | config | wall | per step | artefact |
|---|---|---|---:|---:|---|
| ComfyUI + XPU kernel | t2v | `fl2va_pruned_fp8_scaled` DiT + nvfp4 encoder, 1376×768, 124 f, 8 steps | **300 s** | 37.5 s | 124 frames, AAC 32 kHz, frame mean 82.3–95.2 (std 57.6–65.5) |
| ComfyUI + XPU kernel | ref2v | `ref2va_pruned_int8_convrot` DiT, same shape | **280 s** | 35 s | 124 frames, AAC 32 kHz, frame mean 106.7–111.6 (std 61) — needs fix (1) |
| vLLM-Omni, 1 card BF16 + DLO 2 | t2va | `FL2VA/` partition | — | — | aborts (2) |
| vLLM-Omni, 1 card BF16 + DLO 2 | ref2va | `Ref2VA/` partition | — | — | aborts (1) |

Peak VRAM was not captured for these runs.

**What the numbers say.** 37.5 s/step on the B70 beats both earlier points at the
same shape: the eager-int8 B70 baseline in `~/work/nv_h3` (40.3 s/step) and the
stock B60 figure above (42 s/step). The clip is well exposed, so the int8 path
through `omni_xpu_kernel` is working, not just fast.

**(1) `torch.nonzero` returns wrong indices on this runtime, which is what broke every
reference-image run.** The symptom is an abort far away from the cause:

```
torch-xpu-ops .../ATen/native/xpu/sycl/Indexing.h:622:
Assertion `index >= -sizes_[i] && index < sizes_[i] && "index out of bounds"` failed
```

The minimal repro needs no ComfyUI and no weights:

```python
m = torch.zeros(176, dtype=torch.bool, device="xpu"); m[7:151] = True
m.nonzero(as_tuple=True)[0].cpu()   # 144 values, but the tail is garbage: [..., 1, 1, 1]
m.cpu().nonzero(as_tuple=True)[0]   # the correct, sorted list
```

`mask.sum()` is right (`144`), the mask round-trips correctly, and the index list starts
correctly — only the tail is wrong and unsorted. Length-dependent: 64 is fine, 65–1000
are broken, 4096 and 41042 are fine. Boolean-mask **writes** convert the mask through
`nonzero` and therefore abort in the index kernel; boolean-mask **reads** convert the
same way and fail *silently*, returning the wrong rows.

That is why the failure looked so arbitrary:

- ComfyUI ref2v does three such mask writes — the Qwen3-VL DeepStack inject
  (`comfy/text_encoders/llama.py:1066`) and the two `_embed_and_pack` scatters
  (`comfy/ldm/minimax/model.py:577`, `:582`) — and dies on the first one.
- t2v never builds a boolean mask, so it was unaffected.
- vLLM-Omni's ref2va dies on the **same** assertion from its own mask use, which is why
  the suspicion moved to the runtime rather than either harness.
- Ruled out along the way: the OmniXPU node (`OMNIXPU_ENABLE=0` identical), reference
  image size (2752×1568 and 1376×784 both die), graph capture (`enable_graph` is gated on
  `is_device_cuda`), and upstream code drift (ComfyUI master has the same line).

Fix: `setup_xpu_comfy.sh` now patches those three call sites to compute the index list on
the CPU (`mask.cpu().nonzero(...)`) and use `index_select`/`index_copy_`. A boolean mask is
tiny, so the cost is not measurable; with it, ref2v runs at 35 s/step and the clip is
correct. The patch is idempotent and lives in the script, not in `vendor/`.

**(2) vLLM-Omni text-only (`t2va`) fails earlier and differently**: the text encoder
receives zero token ids, so `_get_rope_index` calls `torch.cat` on an empty list at
1376×768 (`pipeline_minimax_h3.py:1509` → `encoder.py:1077`), and the same request at
512×288 tries to allocate ~4.3e9 GiB. The prompt does reach the server (`prompt` is the
right form field) and the 8-step artifact guard rejects 2-step runs by design, so this
is inside the pipeline rather than the harness.

**Mode/partition pairing is enforced.** A Ref2VA adapter requested from an `fl2va`
server returns HTTP 500: *"…is a Ref2VA Turbo artifact; start the server with
--task-type ref2va"*. Switching modes does restart the server, as the top-level docs say.
