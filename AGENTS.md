# AGENTS.md

Intel XPU scaffolding for MiniMax-H3 ("神笔马良") and LTX-2 video generation on Intel
Arc B60/B70. The env is declared in `pyproject.toml`; everything else is thin bash
wrappers around `vendor/` source checkouts. No tests, linter, formatter, typecheck or
CI — verification is the smoke commands below, so run the one thing you touched
instead of hunting for a suite.

## Environments — five venvs, and the defaults do not line up

| venv | built by | used by |
|---|---|---|
| `.venv` | `bash setup.sh` (`uv sync`) | `h3-test.sh`, `gpu-monitor.sh`, the job submitter inside `gen_video.sh` |
| `.venv-xpu` | `./setup_xpu_comfy.sh` (copies `.venv`, 11 GiB) | ComfyUI server via `run_xpu_comfy.sh` |
| `.venv-comfy` | `./setup_comfyui_env.sh` (its default) | same role as `.venv-xpu` |
| `.venv-omni` | `./setup_vllm_omni_env.sh` | vLLM-Omni, when `OMNI_VENV=.venv-omni` |
| `vendor/LTX-2/.venv` | `setup.sh` (torch **2.12.0**) | `ltx-gen.sh` only |

The ComfyUI venv name does **not** propagate: `setup_comfyui_env.sh` honors
`COMFY_VENV` (default `.venv-comfy`), `run_xpu_comfy.sh` reads a *different* variable
`XPU_VENV` (default `.venv-xpu`), and `gen_video.sh` forwards neither — it reads
`COMFY_VENV` for its own sanity check and JSON parsing but shells out to
`run_xpu_comfy.sh` unchanged. Build straight into the default instead of fighting it:

```bash
COMFY_VENV=$PWD/.venv-xpu ./setup_comfyui_env.sh
```

`.venv` must exist on every path: `gen_video.sh` hardcodes
`$ROOT/.venv/bin/python comfyui_h3_gen.py` (stdlib-only, but the interpreter is
required), and `gpu-monitor.sh` hardcodes `.venv/bin/python` too. The vLLM-Omni side
is consistent — one `OMNI_VENV` is read by both `gen_video_omni.sh` and
`vllm-omni-run.sh`; `setup_vllm_omni_env.sh` prints the exact line to re-run with.

The LTX venv stays on torch 2.12.0 (+torchvision 0.27.0) on purpose while everything
else is 2.14.0. Do not "harmonize" them; LTX-2 upstream is untested on 2.14.

## Commands

```bash
bash setup.sh                     # clones vendor/*, builds the LTX venv, patches it, uv sync
./setup_comfyui_env.sh            # .venv-comfy: ComfyUI + Intel XPU kernel stack (+64 GB weights)
./setup_vllm_omni_env.sh          # .venv-omni: compiles vllm + vllm-omni (30-60 min)
./setup_xpu_comfy.sh --check      # verify the XPU kernel stack, change nothing

.venv/bin/python -c "import torch; print(torch.__version__, torch.xpu.device_count())"   # smoke test
.venv/bin/vllm serve --omni --help
PYTHONPATH=vendor/ComfyUI .venv/bin/python -c "import comfy.model_management as mm; print(mm.get_torch_device())"

./gen_video.sh <project>/         # ComfyUI, both H3 modes from one server (port 8288)
./gen_video_omni.sh <project>/    # vLLM-Omni (port 18090, not 8090 — LTX UI owns that)
./h3-test.sh [GPU] ["prompt"]     # single-shot H3 via the JZL node (port 8190)
./run_xpu_comfy.sh --gpu N --port P --out DIR   # bare ComfyUI with the kernel preflight gate
./run-ltx-server.sh               # LTX web UI, port 8090
./ltx-gen.sh [--distilled] --prompt "..."
./run_vllm_tp.sh <TP> [port]      # vLLM TP inference, e.g. ./run_vllm_tp.sh 8 8000
.venv/bin/python -u stress_vllm.py <concurrency> <max_tokens> <seconds>
./gpu-monitor.sh -B 70 --interval 2   # -B 60|70 -> PCI id 0xe211/0xe223; omit to auto-detect
.venv/bin/python bench_matmul.py --n 8192 --iters 10
```

GPU index defaults are host-specific: `gen_video.sh` / `run_xpu_comfy.sh` default to
`XPU_GPU=16` and log `/dev/dri/renderD$((128+16))`, which does not exist on a small box —
pass `--gpu 0`. `h3-test.sh` already defaults to 0. On this checkout `torch.xpu` sees exactly
one device, the B70 (`0xe223`, 30.3 GiB) as `xpu:0`, and its render node is **renderD129**,
not 128 (128 is the iGPU, `0x7d67`) — so `run_xpu_comfy.sh`'s `renderD(128+N)` heuristic is
only an existence check and its log line can name the wrong node; `ZE_AFFINITY_MASK` is the
value that actually pins the card.

## Two H3 entry points, same project layout

```
<project>/input/prompt.txt   required; the whole file is the prompt
<project>/input/img/*.png    0 files -> text-to-video; 1..9 -> text+reference-to-video
<project>/output/            created if missing; the mp4 lands here
```

ComfyUI is the latency path (pruned int8_convrot DiT stays resident, 21 GB):
ref2v ~380 s with SOL, t2v ~410 s. vLLM-Omni streams the 62 GB BF16 DiT over PCIe
every step — ref2v ~696 s, t2v ~503 s. Use vLLM-Omni for the API or parity with the
reference implementation, not for speed.

Not interchangeable:

- ComfyUI serves both modes from one server. vLLM-Omni needs a different checkpoint
  partition (`FL2VA` vs `Ref2VA`), so switching modes restarts the server;
  `gen_video_omni.sh` records the running task in `vllm-omni-output/server.task` and
  refuses to reuse the wrong one.
- vLLM-Omni must name the turbo adapter **in every request**
  (`-F 'lora={"name":…,"path":…}'`). A server-side `--lora-path` only preloads it;
  `set_active_adapter(None)` on the next request silently drops back to the base model
  — which also drops the artifact's 8-step / `flow_shift=6` contract and yields a dark
  blurred clip. MiniMax-H3 does not set `diffusion_lora_fixed_adapter` (only TaoMate-H3
  does), so there is no sticky mode, and `--lora-backend distill` is unavailable
  (the H3 pipeline has no `load_lora_weights`).
- Single-card vLLM-Omni must use BF16 plus
  `--enable-distributed-layerwise-offload --dlo-resident-layers 2`; `--diffusion-offload-config`
  cannot be combined with the turbo LoRA at all (`minimax_h3/lora.py:298` rejects
  `MODEL_LEVEL`/`LAYER_WISE`, not `DISTRIBUTED_LAYER_WISE`).

## ComfyUI weights: the mapping, not a flag

ComfyUI resolves every checkpoint through `extra_model_paths.yaml` at startup, so a
wrong path surfaces only as "value not in list" minutes in. `gen_video.sh` and
`run_xpu_comfy.sh` take `--models PATH` (`H3_MODELS=`) and write a **per-run** mapping
passed as `--extra-model-paths-config`; `gen_video.sh` also preflights the exact files
the chosen mode needs before starting a server.

- That per-run mapping needs `is_default: true`, or ComfyUI loads its own yaml first
  and `add_model_folder_path` appends — a same-named checkpoint in the global tree
  wins silently.
- Two scripts still write the *global* `vendor/ComfyUI/extra_model_paths.yaml`:
  `h3-test.sh` (from `H3_MODELS`) and `setup_comfyui_env.sh` (from `MODELS_DIR`, only
  when given explicitly or no file exists).
- `comfyui_h3_gen.py` hardcodes the five filenames (`*_pruned_int8_convrot` DiT,
  `qwen3vl_32b_minimax_h3_nvfp4_awq` encoder, fp16 video VAE, fp32 audio VAE, and the
  `*_comfyui_bf16` turbo LoRA). The vLLM-Omni path instead reads Diffusers-layout
  `FL2VA/` / `Ref2VA/` dirs, and its LoRAs must **not** carry the `_comfyui_` suffix —
  the loader refuses those by name.
- Documented default root is `~/work/models/ComfyUI-MiniMax-H3`. Check what actually
  exists before running: on this checkout `~/work/models/h3-comfy` holds the
  `fl2va_pruned_fp8_scaled` DiT + the fl2v LoRA only (no ref2va DiT, no
  `int8_convrot` name), so `gen_video.sh` preflight fails as-is.

`h3_input/` is vendored and needs no external package: `custom_nodes/` (JZL node
82acd16 + LatentUpscaler 2e568df, pinned in `COMMITS`, symlinked into
`vendor/ComfyUI/custom_nodes/`), `prompts/jzl_shot1_api.json`, `assets/`,
`jzl_manager.json`. The whole H3 graph is built *inside* `JZL_MiniMaxAssetManager`
(the API workflow is 4 loaders + that one node), so ComfyUI's native `MiniMaxH3*`
nodes cannot run it.

`h3-test.sh` defaults that produce a different video when changed: 8 steps at
`res_multistep`/`beta`, CFG 1.0, seed 1001, bf16 VAE compute, 1.0 MP 16:9,
`--input-directory h3_input/assets`, `--async-offload 2 --reserve-vram 2`. Dropping
the input directory loses the reference rows (token count 41042 → 38970, frame mean
109 → 87). Env overrides: `PORT`, `TIMEOUT`, `H3_INPUT`, `H3_MODELS`, `STEPS`,
`DURATION`, `SEED`.

## `vendor/` is gitignored and rebuilt by `setup.sh`

`vendor/vllm`, `vendor/vllm-omni`, `vendor/ComfyUI` (+ `custom_nodes/ComfyUI-H3-Motion-Context`)
and `vendor/LTX-2` are clones; re-running `setup.sh` is safe (each step is guarded by
`[ -d ]` / grep checks).

`setup.sh` applies three idempotent `perl -0pi` XPU patches to LTX-2 source (XPU device
selector, fp32 audio vocoder, cuDNN SDPA backend excluded). **They exist only in the
untracked checkout** — edit them in `setup.sh`, not the vendor tree, or they vanish on
re-clone.

Never install LTX-2 packages without `--no-deps`: their `pyproject.toml` pins
`torch = { index = "torch" }` (cu132) and would replace the XPU build. Never install
their `natten` extra (pins torch 2.13.0+cu132).

## uv resolution traps baked into `pyproject.toml`

Each mirrors the official vLLM-Omni XPU Docker build; do not "simplify" them:

- `exclude-dependencies = ["triton"]` — `openai-whisper` / `xgrammar` pull stock
  triton, which shadows `triton-xpu`; the `triton` import must resolve to `triton-xpu`.
- `no-build-isolation-package = ["vllm", "vllm-omni"]` — their `setup.py` imports torch
  at module level, so they must build inside the venv with XPU torch already present.
- `[[tool.uv.dependency-metadata]]` — static `requires-dist` for both, so uv never
  executes their `setup.py`. **Bumping vllm/vllm-omni means updating these lists and
  `envs/vllm-omni/pyproject.toml`**, or the two resolutions drift from upstream.
- `extra-build-variables` sets `VLLM_TARGET_DEVICE=xpu` / `VLLM_OMNI_TARGET_DEVICE=xpu`;
  `[tool.uv.workspace] exclude = ["vendor/*"]` stops uv adopting the clones as members.

`uv sync` compiles vllm + vllm-omni from source — slow, not hung.

### Version pins that are not free choices

- **`torch==2.14.0`, not 2.14.1** — vllm's `requirements/xpu.txt` pins `2.14.0`;
  resolving 2.14.1 fails the whole graph.
- **`torchvision==0.29.0`** pairs exactly with torch 2.14.0; `0.29.1` is built against
  2.14.1 — don't float it.
- **`torchaudio==2.11.0`** is the newest XPU build ever published. It does load and run
  against torch 2.14 despite being built for 2.11.
- **`diffusers==0.40.0`** is a hard pin from vllm-omni (PyPI is at 0.41.x);
  `accelerate==1.12.0` likewise.
- **`vllm-omni==0.32.0rc1`** is a prerelease by necessity — no stable omni release
  builds against vLLM 0.32 / torch 2.14 yet.
- `transformers>=5.16.1,<5.20.0` comes from both vllm and vllm-omni.

### Failure modes already hit once — do not re-debug

- **`ModuleNotFoundError: openai_harmony`** with `oss-harmony` installed: uv cache
  corruption leaves only the dist-info. `uv cache clean oss-harmony && uv sync
  --reinstall-package oss-harmony`.
- **`output/` missing** crashes `ltx_server.py` at import (it mounts `StaticFiles` at
  module level). `mkdir -p output` first.
- **`torchcodec` needs system FFmpeg shared libraries** (`ldconfig -p | grep libavutil`),
  not the static binary `imageio-ffmpeg` ships. Ubuntu 26.04's FFmpeg 8 → `libavutil.so.60`
  matches the wheel's `core8.so`. `setup.sh` does not install system packages.
- **LTX `Input type (float) and bias type (c10::BFloat16)`** means the vocoder fp32
  patch is gone; upstream moved the file to `.../utils/blocks/audio_decoder.py` and
  `setup.sh` follows it. When LTX-2 refactors, re-check all three patch targets by hand.
- **LTX venv missing modules** (`colour`, `colour-science`) is expected fallout of
  `--no-deps`; add them to the explicit list in `setup.sh` rather than dropping the flag.

## Runtime environment quirks (they point in opposite directions per script)

- `run_vllm_tp.sh` keeps oneAPI `bin` off `PATH` so triton uses the venv's
  `intel-sycl-rt`, and wipes `~/.triton/cache` first — stale JIT artifacts are bound to
  a specific libsycl ABI and segfault. Keep the wipe.
- **TP>1 silently requires oneCCL**: `CCL_ROOT` *and* `LD_LIBRARY_PATH` must point at
  `.venv/lib`, because the wheel keeps the real `libccl.so` there and only `libccl.a`
  in `lib/ccl/cpu/lib/`. The "documented" plugin path still fails.
- **oneAPI's `libur_loader.so.0` must not beat the venv's own.** `~/.bashrc` sources
  `setvars.sh`, so `LD_LIBRARY_PATH` carries `/opt/intel/oneapi/compiler/<ver>/lib`; that
  loader (2026.2) still advertises `LIBUR_LOADER_0.12` but no longer exports
  `urKernelSetArgLocal`, which is precisely the symbol `intel-sycl-rt`'s `libsycl.so.9`
  imports — and `LD_LIBRARY_PATH` outranks the `RUNPATH=$ORIGIN` that would have found the
  venv's own copy. Symptom: `ImportError: .../libsycl.so.9: undefined symbol:
  urKernelSetArgLocal, version LIBUR_LOADER_0.12` plus `torch.xpu.is_available() is False`
  on a healthy driver. Every torch-touching script (`run_vllm_tp.sh`,
  `setup_comfyui_env.sh`, `setup_xpu_comfy.sh`, `run_xpu_comfy.sh`, `h3-test.sh`) now pins
  `LD_LIBRARY_PATH=<venv>/lib`; a manual `.venv*/bin/python` invocation needs the same.
- `ltx-gen.sh` / `run-ltx-server.sh` deliberately **source** `/opt/intel/oneapi/setvars.sh`;
  oneAPI env scripts are not `set -u` safe, hence the `set +u` / `set -u` wrappers.
- SDPA on XPU must avoid the cuDNN backend (`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`);
  MATH works.
- **`setup_xpu_comfy.sh` needs docker** for its default `KERNEL_MODE=image`: it shells out to
  bare `docker image inspect` / `create` / `cp` / `rm` (never `sudo`), so plain `docker` must
  work for the invoking user — group membership, not root. This host has `docker.io` 29.1.3
  and `acm` in the `docker` group, but **no `newgrp`/`sg`** (the `login` package is absent), so
  any shell started before the `usermod` still gets permission denied: start a fresh login
  session, or skip docker with `--kernel-from-source`. Only file extraction happens — no
  `--device /dev/dri`, no Intel container runtime. `setup_comfyui_env.sh` now checks the
  daemon and the image *up front* (it used to fail late, inside `setup_xpu_comfy.sh`, with
  `docker image inspect`'s permission error printed as "image not present"), pulls the image
  itself, and accepts `--kernel-from-source` / `--no-pull` to avoid docker entirely.

## Hardcoded / host-specific paths — check before trusting a default

- `run_vllm_tp.sh`: `MODEL=/home/lm/work/data/Qwen3.6-27B`; `stress_vllm.py` reuses that
  path plus a fixed `LINK_PATHS` BDF list and `localhost:8000` for this 8-switch topology.
- `ltx-gen.sh`: `LTX_MODELS` defaults to `$HOME/paul/models/ltx-2.5`; point it at the real
  tree (e.g. `LTX_MODELS=$HOME/work/models/ltx-2.5`). `--num-frames` must be `8k+1`
  (33/121/161); the server snaps user input to that grid.
- `vllm-omni-run.sh`: `MODEL_ROOT=$HOME/work/models/MiniMax-H3`, `LORA_DIR=<repo>/h3_input/loras`
  (gitignored; `download` fetches it). `h3-test.sh`/`gen_video.sh` default to
  `~/work/models/ComfyUI-MiniMax-H3`.

## Other verified XPU facts

- `xpu-smi` / Sysman cannot read telemetry on these cards; `gpu_monitor.py` reads `xe`
  sysfs counters instead (busy %, freq, power, temp).
- Qwen2.5-Omni needs 3 stages (thinker 16.7 + talker 3.6 + code2wav 1.5 GiB) which do
  not fit one 24 GiB card, and a global `--gpu-memory-utilization` overrides every
  per-stage value — use `--deploy-config deploy/qwen2_5_omni_2xb60.yaml`.
- No native int4 matmul (`torch.quint4x2`); emulate 4-bit GEMM as two int8 GEMMs.
- Stock ComfyUI runs **no** native quantized kernels on XPU (`supports_int8_compute()`
  hardcodes `False` for `is_intel_xpu()`), so `int8_convrot` checkpoints are
  storage-only: every linear dequantizes per forward and the GEMM is BF16. Intel's XPU
  kernel stack is what supplies real int8 (see below).
- `fuser` showing a render node "busy" proves nothing — one process can hold handles on
  all cards while running on one. An OOM signature that does not move when you cut
  resolution means someone else holds the VRAM.
- The JZL node swallows per-segment exceptions and returns an empty bus, so ComfyUI
  reports `status=success` for a run that produced nothing. Confirm the mp4 and the
  `采样完成` / `✅ 解码完成` lines instead.
- TP=8 heavy allreduce can permanently drop a PCIe root port to Gen1 and only a power
  cycle recovers it — see `vllm_tp8_pcie_downgrade.md`.

## Verified numbers (don't re-measure without a reason)

Single Arc B60, 1376×768 / 124 frames, 8-step turbo LoRA, seed 1001:

| path | per step | E2E | note |
|---|---:|---:|---|
| ComfyUI stock `.venv` | 70.1 s | 652 s | ~11 min; baseline |
| ComfyUI + `.venv-xpu` kernel | 47.2 s | 470 s | matches Intel's image within noise |
| ComfyUI + kernel + `--sol` | 39.3 s | 401 s | `gen_video.sh --sol` measured 37.6 s / 380 s |
| vLLM-Omni, 1 card BF16 + DLO 2 | — | 696 s | ref2v |
| vLLM-Omni, 2 cards FP8 + adapter | — | 338–349 s | 14.7 GiB/rank, 7–11 % faster than BF16 |

`run_xpu_comfy.sh` preflights `omni_xpu_kernel` (AOT target, torch minor, `cute`, the
Sol-Attn op) and **refuses to start** when anything is off, because a degraded kernel
raises no error — the same clip just takes 70.1 s/step instead of 47.2. Build it with
`./setup_xpu_comfy.sh --kernel-from-source` (no docker, ~10 min; the default extracts
the wheel from `intel/llm-scaler-omni:0.2.0-b2`). ComfyUI 0.42 + torch 2.14 output is
bit-identical to the 0.39 + torch 2.14.1 reference run, so the JZL nodes are forward
compatible.

**Never load the whole 135 GiB H3 partition across many cards at once.** `device_map="balanced"`
over 16 XPUs wedged three GPUs (`xe … device wedged, needs recovery`) and needed a
reboot. Stream one block at a time on one card.

Full tables (quantization × steps × memory × quality) are in `perf.md` (English) and
`性能分析.md` (Chinese, with the reasoning per number).

## Where things are

- `gen_video.sh` → `run_xpu_comfy.sh` (server) + `comfyui_h3_gen.py` (API graph builder);
  `gen_video_omni.sh` → `vllm-omni-run.sh` (`download`/`serve`/`generate`/`run`/`stop`/`status`)
- `ltx_server.py` + `static/index.html` — LTX web UI (state is server-side, browser polls)
- `envs/vllm-omni/` — standalone `.venv-omni` project (must track the root pins)
- `diffusers-output/` — committed H3 reference render + contact sheet; the 5-stage
  diffusers oracle harness itself is external (`~/work/simon/h3/minimax-h3-diffuser`,
  not in this repo)
- `ltx-gen.md` — LTX CLI guide · `vllm_tp8_pcie_downgrade.md` — PCIe downgrade record
- `vendor/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/CLAUDE.md` — upstream LTX conventions
