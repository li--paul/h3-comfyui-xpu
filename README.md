# diffusers-h3

Intel XPU AI stack in a single `uv` virtual environment (CPython 3.12):

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

`vllm-omni-run.sh` drives the same model through vLLM-Omni instead (2+ GPUs,
tensor parallel + layerwise offload), writing to `vllm-omni-output/`:

```bash
./vllm-omni-run.sh download --task ref2va        # checkpoint + turbo LoRA
./vllm-omni-run.sh status
./vllm-omni-run.sh run --gpus 2 --precision fp8 --lora 8step
./vllm-omni-run.sh run --profile smoke           # 1024x576, 2 steps, no LoRA
./vllm-omni-run.sh serve --gpus 2 --precision bf16 --keep-server
./vllm-omni-run.sh generate --prompt "a fox in falling snow"
./vllm-omni-run.sh stop
```

`--precision` is `fp8` or `bf16`; `--task` is `ref2va` (image references) or
`fl2va` (text/first-frame only); `--lora 8step|4step|none` pins the turbo
schedule (the server rejects a step/flow_shift mismatch, so the script sets them
for you). `--attn TORCH_SDPA`, `--no-dlo` and `--resident-layers N` expose the
tuning knobs.

> **Known issue:** on XPU this path serves correctly and returns a valid mp4 with
> audio, but the picture is degenerate (dark/blurred). FP8, BF16 and both
> attention backends were tested and all fail identically — it looks like the XPU
> diffusion path. ComfyUI remains the working H3 path; see `AGENTS.md`.

### Performance

Full measurements — quantization, step counts, memory, and the quality verdict
per configuration — live in **[perf.md](perf.md)**. Headline numbers for the same
5-second 1376×768 shot (seed 1001):

| runtime | precision | steps | cards | wall | peak VRAM | output |
|---|---|---|---:|---:|---:|---|
| ComfyUI | INT8 ConvRot | 8 | 1 | 652 s | 21.4 GB | correct |
| diffusers | BF16, layer-streamed | 8 | 1 | ~920 s | 17.6 GiB | correct |
| vLLM-Omni | FP8 online | 9 | 2 | 360–370 s | 14.8 GiB/rank | broken |
| vLLM-Omni | BF16 | 9 | 2 | 371 s | 18.7 GiB/rank | broken |

**Two runtimes produce correct video; vLLM-Omni does not.** Same checkpoint, same
seed, same prompt — so the fault is in vLLM-Omni's XPU diffusion path, not the
hardware, weights, or precision. FP8 is vLLM-Omni-only: diffusers can run BF16
here but has no usable INT8 backend, and INT8 ConvRot remains ComfyUI-only.

## Layout

- `.venv/` — virtual environment (uv)
- `vendor/vllm/` — vllm `v0.32.0` source checkout (path source)
- `vendor/vllm-omni/` — vllm-omni `v0.32.0rc1` source checkout (path source)
- `vendor/ComfyUI/` — ComfyUI source checkout (run in place)
- `h3_input/` — MiniMax-H3 custom nodes, API workflow, and reference assets
- `h3_output/` — H3 test renders (sample output committed; `server.log` ignored)
- `vllm-omni-output/` — H3 renders and server logs from `vllm-omni-run.sh`
- `perf.md` — measured H3 performance/quality across runtimes, precisions, steps
- `pyproject.toml` — declarative environment + `uv.lock`
- `setup.sh` — one-command setup (oneAPI env + `uv sync`)
