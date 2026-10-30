# diffusers-h3

Intel XPU AI stack in a `uv` virtual environment (CPython 3.12), with optional
standalone environments for the two video paths:

- **PyTorch** `2.14.0+xpu` (from `https://download.pytorch.org/whl/xpu`)
- **vLLM-Omni** `0.32.0rc1` (built from source, `VLLM_OMNI_TARGET_DEVICE=xpu`)
- **Diffusers** `0.40.0`
- **ComfyUI** (run from source, native XPU support)

Target hardware: 32x Intel Arc Pro B60 (Battlemage) / oneAPI 2026.1 / Level Zero.

## Setup

The whole environment is declared in `pyproject.toml` (packages, XPU wheel
index, vllm/vllm-omni path sources, static metadata and XPU build settings),
so rebuilding from scratch is just:

```bash
bash setup.sh
```

System prerequisite (not handled by `setup.sh`): FFmpeg **shared** libraries, which
`torchcodec` dlopens at import time:

```bash
sudo apt install -y ffmpeg     # Ubuntu 26.04 ships FFmpeg 8 (libavutil.so.60)
```

`setup.sh` only loads the Intel oneAPI toolchain (needed to build vllm for
XPU) and then runs `uv sync`. On hosts where oneAPI is already sourced
globally, `uv sync` alone suffices (as long as the `vendor/` checkouts exist;
`bash setup.sh` creates them if missing):

```bash
source /opt/intel/oneapi/setvars.sh --force            # skip if already global
source /opt/intel/oneapi/ccl/2022.1/env/vars.sh --force
uv sync
```

To start completely from scratch:

```bash
rm -rf .venv && uv sync
```

### Standalone environments

`setup.sh` builds the combined `.venv`. If you only need one of the two video
paths, build just that one instead — each is self-contained, so a fresh host
never pulls the other path's dependencies:

| script | venv | installs |
|---|---|---|
| `./setup_comfyui_env.sh` | `.venv-comfy` | torch + ComfyUI 0.42.0 + Intel's XPU kernel stack + 64 GB of H3 weights |
| `./setup_vllm_omni_env.sh` | `.venv-omni` | `envs/vllm-omni/pyproject.toml`: torch + vllm + vllm-omni, both built from source |

```bash
./setup_comfyui_env.sh                  # env in a few minutes, then 64 GB of weights
./setup_comfyui_env.sh --skip-models    # env only, reuse an existing model dir
./setup_comfyui_env.sh --check          # verify only, change nothing

./setup_vllm_omni_env.sh                # 30-60 min: uv sync compiles vllm
./setup_vllm_omni_env.sh --check
./setup_vllm_omni_env.sh --download     # also fetch the 270 GB of checkpoint partitions
```

Both are opt-in. The generation scripts read `COMFY_VENV` / `OMNI_VENV` and fall
back to `.venv-xpu` / `.venv`, so an existing tree keeps working untouched.

`setup_comfyui_env.sh` does not reimplement the XPU port: it calls
`setup_xpu_comfy.sh` with `XPU_NO_COPY=1` and installs in place, so that logic
exists once. `setup_vllm_omni_env.sh` keeps `envs/vllm-omni/pyproject.toml` in
lockstep with the root `pyproject.toml` — **when you bump the vllm pins, update
both**, or the two resolutions drift apart.

## How it works

- **XPU wheels** (`torch==2.14.0`, torchvision/audio, `triton-xpu`, `torchcodec`)
  are resolved from the `pytorch-xpu` index via `[tool.uv.index]` + `[tool.uv.sources]`.
  `torchaudio` stays at `2.11.0`: it is the newest XPU build published, although
  `torch` is already at 2.14.
- **vllm / vllm-omni** are `path` sources (`vendor/`). Their `setup.py` must
  run against an installed XPU torch (vllm imports torch at module level), so:
  - `dependency-metadata` tells the resolver their requirements without
    executing `setup.py`;
  - `no-build-isolation-package` makes uv build them in the target venv (torch
    is installed first as a dependency);
  - `extra-build-dependencies` pre-installs their build tools;
  - `extra-build-variables` sets `VLLM_TARGET_DEVICE=xpu` /
    `VLLM_OMNI_TARGET_DEVICE=xpu` during their builds.
- **Stock `triton`** is excluded (`exclude-dependencies`) because `openai-whisper`
  and `xgrammar` pull it in, and on XPU the `triton` import must come from
  `triton-xpu`.
- **ComfyUI** has no buildable `pyproject.toml`, so it is not installed as a
  package — its `requirements.txt` entries are project dependencies and it is
  run from `vendor/ComfyUI`.

## Verify

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.xpu.device_count())"
# 2.14.0+xpu 32

.venv/bin/vllm serve --omni --help
PYTHONPATH=vendor/ComfyUI .venv/bin/python -c "import comfy.model_management as mm; print(mm.get_torch_device())"   # xpu:0
```

## Peak FLOPS benchmark

`bench_matmul.py` measures achieved peak compute (TFLOPS) of a square
`N x N` matrix multiply for several dtypes via `torch.matmul`:

```bash
.venv/bin/python bench_matmul.py --n 8192 --iters 10
```

Results on the target GPU (Intel Arc B70, 256 EUs, ~30 GiB):

| dtype | time (N=8192) | TFLOPS |
|-------|---------------|--------|
| fp64  | 773.7 ms | 1.4 |
| fp32  | 48.7 ms | 22.6 |
| fp16  | 6.9 ms | 160.5 |
| bf16  | 6.9 ms | 159.4 |
| int8  | 3.2 ms | 342.4 |

fp16/bf16/int8 use the XMX/DPAS matrix engines; fp32 runs on the vector
lanes and fp64 is emulated. Native int4 (`torch.quint4x2`) has no matmul
kernel on XPU, so 4-bit GEMMs must be emulated as two int8 GEMMs
(nibble-split), capping effective peak at roughly half of int8.

`xpu-smi` (Intel) cannot read telemetry on this host (Sysman fails for the
B70 in the current compute runtime), so `gpu_monitor.py` reads the Xe driver
sysfs counters instead (busy %, frequency, power, temperature). `gt0` is the
B70's compute engine and `gt1` its media engine:

```bash
.venv/bin/python gpu_monitor.py --interval 2
# or via the wrapper:
./gpu-monitor.sh --interval 2
```

Options:

- `--interval <sec>` — sampling period (default 2 s)
- `--count <n>` — number of samples, `0` = run forever (default 0)
- `--device-id <hex>` — PCI device id to monitor (default `0xe223`, the B70)

## Usage

### vLLM-Omni (OpenAI-compatible server, XPU)

```bash
.venv/bin/vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091
```

That default assumes enough VRAM for all three stages (thinker, talker, code2wav)
on one device. On 24 GiB cards the bundled deploy config does not fit, so place
the stages across two GPUs and give the thinker less headroom:

```bash
.venv/bin/vllm serve Qwen/Qwen2.5-Omni-7B --omni --port 8091 \
  --deploy-config deploy/qwen2_5_omni_2xb60.yaml
```

### Diffusers

```python
import torch, diffusers
pipe = diffusers.StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", dtype=torch.float16
).to("xpu")
pipe("a photo of a cat on the moon").images[0].save("out.png")
```

(`stabilityai/stable-diffusion-2-1` is no longer publicly downloadable, and
`torch_dtype=` is deprecated in diffusers 0.40 — use `dtype=`.)

### ComfyUI

```bash
.venv/bin/python vendor/ComfyUI/main.py --listen 0.0.0.0 --port 8188
```

For MiniMax-H3 the launcher pins four things that silently produce the wrong
video if omitted (card mask, input directory, output directory, offload policy):

```bash
./run_xpu_comfy.sh --gpu 16 --port 8288 --out h3_output/xpu_comfy
```

Before `exec` it probes `omni_xpu_kernel` (AOT target, torch minor, `cute`, and
the Sol-Attn op) and refuses to start when anything is off — a degraded kernel
raises no error, the same clip just takes 70.1 s/step instead of 47.2.

The XPU kernel stack (`omni_xpu_kernel` + Kitchen's `xpu` provider + the
`ComfyUI-OmniXPU` node) is installed by `setup_xpu_comfy.sh`, which has two
sources for the kernel wheel:

```bash
./setup_xpu_comfy.sh                    # default: extract it from intel/llm-scaler-omni:0.2.0-b2
./setup_xpu_comfy.sh --kernel-from-source   # compile it from llm-scaler (no docker, ~10 min)
./setup_xpu_comfy.sh --check            # verify only
./run_xpu_comfy.sh --build-kernel ...   # compile inline if the probe fails
```

The source path drives Intel's own release harness
(`comfy-kitchen-xpu/packaging/omni_xpu_kernel/build_uv_wheel_matrix.sh`) rather
than a hand-rolled compile, and pins the llm-scaler revision to the window that
both accepts torch 2.14 and still bundles Sol-Attn. Two ways to get that wrong,
both silent: an older revision fails to build, a newer one drops Sol-Attn into a
separate wheel and `--sol` quietly stops being sparse.

### MiniMax-H3 video generation

`h3-test.sh` runs one H3 shot on a single card and writes to `h3_output/`:

```bash
./h3-test.sh                        # GPU 0, reference prompt, manifest defaults
./h3-test.sh 4                      # bind to GPU 4
./h3-test.sh 0 "a lighthouse in a storm"   # custom prompt, single segment
```

Needs the H3 weights at `~/work/models/ComfyUI-MiniMax-H3` (`H3_MODELS=` to
override); everything else — custom nodes, workflow, reference images — is
vendored in `h3_input/`. Expect ~11 min on one Arc Pro B60 (8 steps at ~70 s/step).

### MiniMax-H3 via vLLM-Omni

`vllm-omni-run.sh` drives the same model through vLLM-Omni instead (1 GPU with
distributed layerwise offload, or 2+ GPUs with tensor parallel), writing to
`vllm-omni-output/`:

```bash
./vllm-omni-run.sh download --task ref2va        # checkpoint + turbo LoRA
./vllm-omni-run.sh status
./vllm-omni-run.sh run --gpus 2 --precision fp8 --lora 8step
./vllm-omni-run.sh run --gpus 1 --precision bf16 --task t2va --lora 8step
./vllm-omni-run.sh run --profile smoke           # 1024x576, 2 steps, no LoRA
./vllm-omni-run.sh serve --gpus 2 --precision bf16 --keep-server
./vllm-omni-run.sh generate --prompt "a fox in falling snow"
./vllm-omni-run.sh stop
```

`--precision` is `fp8` or `bf16`; `--task` is `ref2va` (image references),
`fl2va` (first/last frame) or `t2va` (text only — same `FL2VA` partition, the
request field differs); `--lora 8step|4step|none` pins the turbo schedule (the
server rejects a step/flow_shift mismatch, so the script sets them for you).
`--attn TORCH_SDPA`, `--no-dlo` and `--resident-layers N` expose the tuning
knobs; on `--gpus 1` the resident-layer default drops to 2 by itself, because 12
needs ~15 GB and OOMs on a 24 GB card.

> **Configuration gotcha:** the turbo adapter must be sent in each request, not just
> preloaded. Without it the server runs the base model at 8 steps and the clip comes out
> dark and blurred. `vllm-omni-run.sh` handles this; see `perf.md` for the before/after.

### Generating from a project directory

Both paths have an entry point that takes a directory instead of flags. Same
layout for either:

```
<project>/
  input/prompt.txt     required; the whole file is the prompt
  input/img/*.png      0 files -> text-to-video; 1..9 -> text+reference-to-video
  output/              created if missing; the mp4 lands here
```

```bash
./gen_video.sh <project>/                  # ComfyUI, both modes from one server
./gen_video.sh <project>/ --sol --steps 8  # + sparse attention
./gen_video.sh <project>/ --restart        # force a server restart

./gen_video_omni.sh <project>/             # vLLM-Omni, single card BF16 + DLO 2 layers
./gen_video_omni.sh <project>/ --restart   # required when switching modes
```

Both default to 720p, 5 seconds (1376x768 / 124 frames @ 24 fps), 8 steps with
the turbo adapter, seed 1001, and verify the artefact afterwards (size, frame
count, audio track) rather than trusting the HTTP status.

Measured on one Arc Pro B60, same shot and seed:

| entry point | mode | per step | wall | peak VRAM | frame mean |
|---|---|---:|---:|---:|---:|
| `gen_video.sh --sol` | ref2v | 37.6 s | 380 s | — | 110.1 |
| `gen_video.sh` | t2v | 42.0 s | 410 s | — | 82.7 |
| `gen_video_omni.sh` | t2v | 55.1 s | 503 s | 18.0 GiB | 81.3 |
| `gen_video_omni.sh` | ref2v | 77.3 s | 696 s | 19.7 GiB | 114.7 |

ComfyUI is roughly 2x faster because its pruned int8_convrot DiT stays resident
(21 GB); vLLM-Omni streams the 62 GB BF16 DiT over PCIe every step.

What is **not** interchangeable between the two:

- ComfyUI serves both modes from one server. vLLM-Omni needs a different
  checkpoint partition (`FL2VA` vs `Ref2VA`), so switching modes restarts the
  server. `gen_video_omni.sh` records the running task in
  `vllm-omni-output/server.task` and refuses to reuse the wrong one.
- ComfyUI reference images go through `POST /upload/image`, so switching
  projects never needs `--input-directory` changed or the server restarted.
  Leaving that flag off a fresh server is what silently drops the reference rows
  (token count 41042 -> 38970, frame mean 109 -> 87).
- Single-card vLLM-Omni must use BF16 plus
  `--enable-distributed-layerwise-offload --dlo-resident-layers 2`. The recipe's
  `--diffusion-offload-config` cannot be combined with the turbo LoRA at all
  (`lora.py:298` rejects `MODEL_LEVEL` and `LAYER_WISE`, but not
  `DISTRIBUTED_LAYER_WISE`).

### Performance

Full measurements — quantization, step counts, memory, and the quality verdict
per configuration — live in **[perf.md](perf.md)**. Headline numbers for the same
5-second 1376×768 reference-to-video shot (seed 1001):

| runtime | precision | steps | cards | wall | peak VRAM | output |
|---|---|---|---:|---:|---:|---|
| ComfyUI (stock `.venv`) | INT8 ConvRot | 8 | 1 | 652 s | 21.4 GB | correct |
| ComfyUI + XPU kernel | INT8 ConvRot | 8 | 1 | 470 s | 21.4 GB | correct |
| ComfyUI + XPU kernel + SOL | INT8 ConvRot | 8 | 1 | **380 s** | 21.4 GB | correct |
| vLLM-Omni | FP8 online + turbo LoRA | 8 | 2 | 338–349 s | 14.7 GiB/rank | correct |
| vLLM-Omni | BF16 + turbo LoRA | 8 | 2 | 375 s | 18.7 GiB/rank | correct |
| vLLM-Omni | BF16 + turbo LoRA | 8 | 1 | 696 s | 19.7 GiB | correct |
| diffusers | BF16, layer-streamed | 8 | 1 | ~920 s | 17.6 GiB | correct |

All of them produce a correct clip once configured properly. For vLLM-Omni, FP8 is
the sweet spot on two cards: ~27 % less per-rank memory than BF16 and 7–11 % faster.
Single-card vLLM-Omni cannot use it — the measured profile there is BF16 with
distributed layerwise offload, because the 62 GB DiT has to stream over PCIe every step.

> **Gotcha:** the turbo adapter must be named **in the request**
> (`-F 'lora={"name":…,"path":…}'`). A server-side `--lora-path` only preloads it; without
> the request field the server silently runs the base model, and 8 steps on the base model
> with the turbo schedule yields a dark, blurred clip. `vllm-omni-run.sh` now sends it for you.

## Layout

- `.venv/` — combined environment (uv); `.venv-comfy/` and `.venv-omni/` are the optional standalone ones
- `pyproject.toml` + `uv.lock` — declarative environment for `.venv`
- `setup.sh` — one-command setup for `.venv` (oneAPI env + `uv sync`)

ComfyUI / XPU path:

- `setup_comfyui_env.sh` — build `.venv-comfy` from scratch (ComfyUI + XPU stack + weights)
- `setup_xpu_comfy.sh` — install the XPU kernel stack; `--kernel-from-source` compiles it instead of extracting it from the image
- `run_xpu_comfy.sh` — start ComfyUI pinned to one card, with the kernel preflight gate
- `gen_video.sh` — project-directory entry point (text-to-video / reference-to-video)
- `comfyui_h3_gen.py` — API graph builder and submitter behind `gen_video.sh`

vLLM-Omni path:

- `envs/vllm-omni/pyproject.toml` + `uv.lock` — standalone project for `.venv-omni`
- `setup_vllm_omni_env.sh` — build `.venv-omni` from scratch (compiles vllm)
- `gen_video_omni.sh` — project-directory entry point for the same two modes
- `vllm-omni-run.sh` — lower-level driver (`download` / `serve` / `generate` / `run` / `stop` / `status`)

Shared:

- `vendor/vllm/` — vllm `v0.32.0` source checkout (path source)
- `vendor/vllm-omni/` — vllm-omni `v0.32.0rc1` source checkout (path source)
- `vendor/ComfyUI/` — ComfyUI source checkout (run in place)
- `h3_input/` — MiniMax-H3 custom nodes, API workflow, reference assets, turbo LoRAs
- `h3_output/` — H3 test renders (sample output committed; `server.log` ignored)
- `vllm-omni-output/` — H3 renders, server logs, and the `server.task` marker
- `perf.md` — measured H3 performance/quality across runtimes, precisions, steps
- `性能分析.md` — the same analysis in Chinese, with the reasoning behind each number
