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

| # | precision | attention | LoRA | steps | flow shift | shape | E2E | peak/rank | frame mean (std) | verdict |
|---|---|---|---|---|---:|---:|---|---:|---:|---|---|
| 1 | FP8 online | FLASH_ATTN | 8-step | 9 | 6 | 1376×768 | 360 s | 14.8 GiB | 40.8 (28.7) | broken |
| 2 | FP8 online | FLASH_ATTN | 8-step | 9 | 6 | 1376×768 | 370 s | 14.8 GiB | 39.8 (30.6) | broken |
| 3 | BF16 | FLASH_ATTN | 8-step | 9 | 6 | 1376×768 | 371 s | 18.7 GiB | 43.0 (32.9) | broken |
| 4 | BF16 | TORCH_SDPA | 8-step | 9 | 6 | 1376×768 | 450 s | 18.4 GiB | 43.4 (32.6) | broken |
| 5 | FP8 online | FLASH_ATTN | none | 2 | 12 | 1024×576 | 98 s | 12.8 GiB | 71.7 (70.5) | broken |
| 6 | FP8 online | FLASH_ATTN | none | 2 | 12 | 1024×576 | 78 s | 11.8 GiB | 74.9 (67.8) | broken |

Every request returned HTTP 200 with a well-formed 124-frame H.264 mp4 carrying
32 kHz stereo AAC. No `DEVICE_LOST`, no xe/GuC faults, no OOM.

> **Broken** means the picture is degenerate: dark, blurred, and zoomed into a
> fragment of the reference image instead of the described road. It is a content
> failure, not a container or audio failure.

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
| FP8 online | ~half, quantized at load | 360–370 s | 14.8 GiB | — | 40.8 / 39.8 (28.7 / 30.6) |
| BF16 | full checkpoint | 371 s | 18.7 GiB | +26 % memory, +0–3 % time | 43.0 (32.9) |
| BF16 (diffusers) | full checkpoint, layer-streamed | ~920 s | 17.6 GiB on 1 card | n/a, different runtime | **121.2 (65.7)** |
| INT8 ConvRot (ComfyUI) | 21 GB single file | 652 s | ~21.4 GB on 1 card | n/a, different runtime | **109.4 (69.6)** |

Reading this:

- FP8 does what it advertises — **~26 % less per-rank memory**, essentially free
  in wall time.
- **Precision is not what breaks the vLLM-Omni picture.** BF16 there is just as
  bad as FP8 (43.0 vs 40.8 mean).
- **INT8 ConvRot has no diffusers or vLLM-Omni equivalent.** Diffusers can only
  do BF16 here: `torchao`/`bitsandbytes`/`quanto`/`sdnq` are all absent, GGUF has
  no H3 file, and diffusers' `AutoRoundQuantizer` refuses on-the-fly quantization
  (`requires_calibration = True`). The harness supports `int8_convrot` when the
  checkpoint's `config.json` declares it — ours does not (it is the full BF16
  release), so INT8 stays ComfyUI-only.
- **Two independent runtimes produce correct video at BF16-with-offload**
  (diffusers 121.2, ComfyUI 109.4) while vLLM-Omni's BF16 produces 43.0. The
  difference is the runtime, not the precision or the weights.

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
| diffusers 0.41, BF16, layer-streamed, 1 card | **yes** | correct village-road clip; frame mean 121.2, std 65.7 |
| vLLM-Omni, FP8 or BF16, 9 steps | **no** | dark blurred crop of the reference image; frame std 28.7–32.9; identical across precision and attention backend |
| vLLM-Omni, 2 steps, no LoRA | **no** | blurry smear, no scene structure |

**The fault is in vLLM-Omni's XPU diffusion execution path, not in the hardware,
the weights, or the precision.** Two independent implementations running BF16
with layer/layerwise offload on the same card and the same checkpoint produce a
correct shot; vLLM-Omni does not.

Isolation performed on vLLM-Omni (all negative): precision (FP8 vs BF16),
attention backend (FLASH_ATTN vs TORCH_SDPA), LoRA and step schedule (9-step vs
2-step no-LoRA), VRAM (18.7 of 23.9 GiB peak), and start-up (healthy, 70–120 s).
No MiniMax-H3 recipe upstream covers XPU (CUDA and NPU only), which is consistent
with `platforms/xpu` providing its own diffusion attention backend.

### Operational hazard found the hard way

Loading the whole 135 GiB partition across 16 cards at once with
`device_map="balanced"` **wedged three GPUs** (`xe ... device wedged, needs
recovery`, GuC queue timeouts) and left the stack unusable until a reboot. That
matches the delivery manifest's `UR_RESULT_ERROR_DEVICE_LOST` warning for
concurrent ~20 GB loads. Streaming one block at a time on a single card (the
diffusers path above) peaked at 17.6 GiB with no driver events at all — prefer
that shape.

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

# vLLM-Omni (broken path) — ~8 min on two cards
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

- **vLLM-Omni step sweep** at a fixed shape (2 / 4 / 9 steps) — needed for a real
  per-step slope.
- **`--dlo-resident-layers` 16–20 at FP8** — the most promising latency knob,
  given ~9 GiB/rank sits unused.
- **`fl2va` / T2VA** — the other 135 GiB partition is on disk but never served.
- **Single-GPU FP8 without DLO** — would remove both sharding and offload from
  the picture, separating "TP is wrong" from "offload staging is wrong".
- **ComfyUI at other step counts** — only the 8-step turbo LoRA is available;
  the 4-step artifact was not fetched.
- **Concurrent requests** — both runtimes served one request at a time.
