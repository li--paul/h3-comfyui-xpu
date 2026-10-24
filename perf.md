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
