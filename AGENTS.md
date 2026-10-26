# AGENTS.md

Intel XPU AI-stack scaffolding repo (PyTorch + vLLM-Omni + Diffusers + ComfyUI + LTX-2) on
Intel Arc B60/B70 GPUs. Declarative env in `pyproject.toml`, everything else is thin wrappers.

## Two virtual environments — the single most important fact

| env | path | holds |
|---|---|---|
| main | `.venv/` | torch 2.14.0+xpu, vllm, vllm-omni, ComfyUI deps, the server, benchmarks |
| LTX-2 | `vendor/LTX-2/.venv/` | ltx-core / ltx-pipelines on **torch 2.12**, deliberately isolated transformers version |

The two torch versions differ on purpose (`setup.sh` pins the LTX venv to 2.12). Do not
"harmonize" them; LTX-2 upstream is untested on 2.14.

Always invoke a script with the interpreter that owns its deps. `ltx_server.py` itself runs in
`.venv` but shells out to `ltx-gen.sh`, which uses the LTX venv — the split is intentional.

## Commands

```bash
bash setup.sh                     # one-time: clones vendor/*, builds LTX venv, patches it, uv sync
.venv/bin/python -c "import torch; print(torch.__version__, torch.xpu.device_count())"   # smoke test
.venv/bin/vllm serve --omni --help
PYTHONPATH=vendor/ComfyUI .venv/bin/python -c "import comfy.model_management as mm; print(mm.get_torch_device())"

./run_vllm_tp.sh <TP> [port]      # vLLM TP inference, e.g. ./run_vllm_tp.sh 8 8000
.venv/bin/python -u stress_vllm.py <concurrency> <max_tokens> <seconds>   # PCIe downgrade stress+monitor
./run-ltx-server.sh              # LTX web UI, port 8090 (LTX_SERVER_PORT to override)
./ltx-gen.sh [--distilled] --prompt "..." [flags...]
./gpu-monitor.sh -B 70 --interval 2   # -B 60|70 maps to PCI id 0xe211/0xe223; omit for auto-detect
.venv/bin/python bench_matmul.py --n 8192 --iters 10
```

No tests, linter, formatter, typechecker, or CI exist. Verification is the smoke commands above.
Run a single thing rather than a suite.

## `vendor/` is gitignored and rebuilt by `setup.sh`

`vendor/vllm`, `vendor/vllm-omni`, `vendor/ComfyUI`, `vendor/LTX-2` are clones. Re-running
`setup.sh` on an existing tree is safe (each step is guarded by `[ -d ]` / grep checks).

`setup.sh` applies three idempotent `perl -0pi` XPU patches to LTX-2 source (XPU device selector,
fp32 audio vocoder, cuDNN SDPA backend excluded). **These live only in the untracked checkout** —
edit them in `setup.sh`, not in the vendor tree, or the change is silently lost on re-clone.

Do not add anything under `vendor/` to git, and don't install LTX-2 packages without `--no-deps`:
their `pyproject.toml` pins `torch = { index = "torch" }` (cu132) and would replace the XPU build.
Never install their `natten` extra (pins torch 2.13.0+cu132).

## uv resolution traps baked into `pyproject.toml`

Don't "simplify" these away — each mirrors the official vLLM-Omni XPU Docker build:

- `exclude-dependencies = ["triton"]` — `openai-whisper`/`xgrammar` pull stock triton, which
  shadows `triton-xpu`. The `triton` import must resolve to `triton-xpu` (3.7.1).
- `no-build-isolation-package = ["vllm", "vllm-omni"]` — their `setup.py` imports torch at module
  level, so they must build inside the venv with XPU torch already present.
- `[[tool.uv.dependency-metadata]]` — static `requires-dist` for both, so uv never executes their
  `setup.py`. **When bumping vllm/vllm-omni, update these lists too**, or resolution silently drifts
  from upstream requirements.
- `extra-build-variables` sets `VLLM_TARGET_DEVICE=xpu` / `VLLM_OMNI_TARGET_DEVICE=xpu`.
- `[tool.uv.workspace] exclude = ["vendor/*"]` stops uv auto-discovering the clones as members.

`uv sync` compiles vllm + vllm-omni from source — it is slow, not hung.

### Version pins that are not free choices

- **`torch==2.14.0`, not `2.14.1`.** The XPU index has both, but vllm's `requirements/xpu.txt`
  pins `torch==2.14.0`; resolving 2.14.1 fails the whole graph.
- **`torchvision==0.29.0`** pairs exactly with torch 2.14.0. `0.29.1` accepts `torch>=2.14.0`
  but is built against 2.14.1 — don't float it.
- **`torchaudio` is frozen at `2.11.0`.** No 2.13/2.14 XPU build was ever published; 2.11.0 is
  the newest on the index and is pinned explicitly so uv cannot silently drop it. It does load
  and run against torch 2.14 (verified), despite being built for 2.11.
- **`diffusers==0.40.0`** is a hard pin from vllm-omni, even though PyPI is at 0.41.x.
- **`vllm-omni==0.32.0rc1` is a prerelease by necessity** — no stable omni release builds
  against vLLM 0.32 / torch 2.14 yet.
- `transformers>=5.16.1,<5.20.0` comes from both vllm and vllm-omni; ComfyUI only sets a floor.

### Failure modes already hit once — do not re-debug

- **`ModuleNotFoundError: openai_harmony`** despite `oss-harmony` being installed. `oss-harmony`
  is a metadata-only shim that re-exports the `openai_harmony` module; if uv's cache holds a
  half-written install, site-packages gets only the dist-info. Fix:
  `uv cache clean oss-harmony && uv sync --reinstall-package oss-harmony`. The PyPI wheel does
  contain `openai_harmony/`, so this is cache corruption, not a bad release.
- **`output/` missing** crashes `ltx_server.py` at import (see below), and `ltx-gen.sh` only
  creates it at generation time — run `mkdir -p output` before the server.
- **`torchcodec` needs system FFmpeg shared libraries**, not just the static binary that
  `imageio-ffmpeg` provides. Without them `import torchcodec` dies on
  `libavutil.so.60: cannot open shared object file` (the wheel tries `core4..8.so` and only the
  matching FFmpeg major works). Fixed here with `sudo apt install ffmpeg` — Ubuntu 26.04 ships
  FFmpeg 8 → `libavutil.so.60`, which the wheel's `core8.so` loads. On a fresh host check
  `ldconfig -p | grep libavutil` before blaming torchcodec. `setup.sh` does not install system
  packages, so this is a manual prerequisite.
- **LTX `RuntimeError: Input type (float) and bias type (c10::BFloat16)`** means the vocoder
  fp32 patch is no longer applied. Upstream moved the file from
  `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py` to
  `.../utils/blocks/audio_decoder.py`; `setup.sh` now patches the new path and guards with
  `[ -f ]`. When LTX-2 upstream refactors, re-check all three patch targets by hand.
- **LTX venv missing modules** (`colour`, `colour-science`) is expected fallout of installing
  ltx-core with `--no-deps`: its own dependencies are never resolved. Add anything new to the
  explicit list in `setup.sh` instead of dropping `--no-deps`.

## Runtime environment requirements (opposite per script — do not "fix")

- `run_vllm_tp.sh` keeps oneAPI `bin` off `PATH` so triton uses the venv's `intel-sycl-rt`, and
  wipes `~/.triton/cache` first — stale JIT artifacts are bound to a specific libsycl ABI and
  segfault. Since `intel-sycl-rt` 2026.1 the venv ships `libsycl.so.9`, the same major as system
  oneAPI, so the old `.8`-vs-`.9` isolation is no longer needed; keep the cache wipe anyway.
- **TP>1 silently requires oneCCL.** Without `CCL_ROOT` *and* `LD_LIBRARY_PATH` pointing at
  `.venv/lib`, engine startup dies with `oneCCL could not be initialized! Could not load any
  plugin`. The wheel keeps the real `libccl.so` in `.venv/lib/` and only `libccl.a` in
  `lib/ccl/cpu/lib/`, so pointing `CCL_ROOT` at the "documented" plugin path still fails.
- `ltx-gen.sh` / `run-ltx-server.sh` deliberately **source** `/opt/intel/oneapi/setvars.sh`.
- oneAPI env scripts aren't `set -u` safe; they wrap sourcing in `set +u` / `set -u`.

## `output/` must exist before starting the server

`output/` is untracked, so a fresh clone has none — and `ltx_server.py` mounts it at import time:
`RuntimeError: Directory '.../output' does not exist`. Do `mkdir -p output` first.

## Hardcoded paths — change them for other hosts

- `run_vllm_tp.sh`: `MODEL=/home/lm/work/data/Qwen3.6-27B`
- `stress_vllm.py`: same model path, `URL=localhost:8000`, plus a fixed `LINK_PATHS` BDF list for
  this 8x-switch topology
- `ltx-gen.sh`: models default to `$HOME/paul/models/ltx-2.5`, but on this host they live in
  `/home/lm/work/models/ltx-2.5` — always pass `LTX_MODELS=/home/lm/work/models/ltx-2.5`.
- LTX `--num-frames` must be `8k+1` (33, 121, 161); the server snaps user input to that grid

## MiniMax-H3 (神笔马良) video generation

One command: `./h3-test.sh [GPU] ["prompt"] [--keep-server]`. It runs this repo's
ComfyUI (`vendor/ComfyUI`) on one card and writes to `h3_output/`.

- `h3_input/` — everything the test needs, vendored (no external package required):
  - `custom_nodes/` — `ComfyUI-JZL-MiniMax-H3` (82acd16) and
    `ComfyUI-MiniMaxH3_LatentUpscaler` (2e568df); `COMMITS` records the pins.
    The script symlinks them into `vendor/ComfyUI/custom_nodes/`.
  - `prompts/jzl_shot1_api.json` — the reference API workflow
  - `assets/` — reference images for the scene slots
  - `jzl_manager.json` — seed for the node's persisted settings
- Weights come from `~/work/models/ComfyUI-MiniMax-H3` via a generated
  `vendor/ComfyUI/extra_model_paths.yaml` (`H3_MODELS=` to override).
- The script picks `minimax_h3_video_vae_bf16` or `_fp16`, whichever exists, and
  rewrites the workflow's `VAELoader` + `manager_settings` to match.
- Env overrides: `PORT`, `TIMEOUT`, `H3_INPUT`, `H3_MODELS`, `STEPS`, `DURATION`, `SEED`.

**The whole H3 graph is built inside one custom node.** The API workflow is only
6 nodes: four loaders plus `JZL_MiniMaxAssetManager`, which internally builds
segmentation → ref2va encode → sample → video+audio VAE decode. So ComfyUI's
native `MiniMaxH3*` nodes cannot run this workflow — they would need a hand-built
graph.

Verified (2026-10-22, single Arc B60, seed 1001): 8 steps at **70.1 s/step**,
**9 min 21 s sampling / ~11 min wall**, 124 frames, 1376×768, 5.17 s, AAC 32 kHz
stereo present. Defaults that matter: `res_multistep` / `beta`, 8 steps, CFG 1.0,
seed 1001 fixed, bf16 VAE compute, 1.0 MP 16:9. Output from ComfyUI 0.42 + torch
2.14.0 is **bit-identical** to the reference package's 0.39 + torch 2.14.1 run
(mean abs pixel diff 0.00 across all 124 frames), so the JZL nodes are forward
compatible with 0.42 and the diffusers 0.40 / accelerate 1.12 pins are not a
problem for H3.

`dmesg` shows no `DEVICE_LOST` and no xe/GuC faults for a single card. The
zero-mp4 output dirs from 2026-10-20 were load-concurrency casualties (the
delivery manifest describes `UR_RESULT_ERROR_DEVICE_LOST` when ~20 GB lands on
many cards at once), not a package bug.

## MiniMax-H3 on vLLM-Omni — works, but the adapter must be in the request

Driver: `./vllm-omni-run.sh` (`download` / `serve` / `generate` / `run` / `stop` / `status`).
It handles the partition + LoRA download, TP/DLO sizing, the LoRA's mandatory
steps/flow_shift, writes to `vllm-omni-output/`, and raises
`VLLM_OMNI_DLO_DP_WAVE_TIMEOUT` so runs over 600 s don't die with
`RPC call to execute_model timed out`. Full measurement tables (runtime × precision ×
steps, memory, quality, gaps) are in `perf.md`.

**`--lora-path` only preloads an adapter; the adapter must be named in every request** via
`-F 'lora={"name":…,"path":…}'`. `set_request_adapter` runs on each request and
`set_active_adapter(None)` deactivates whatever was loaded, so a server-side `--lora-path` alone
serves the base model. MiniMax-H3 does not set `diffusion_lora_fixed_adapter` (only TaoMate-H3
does), so there is no sticky mode, and `--lora-backend distill` is unavailable because the H3
pipeline has no `load_lora_weights`. The request field is also what triggers
`_validate_turbo_sampling`, so omitting it silently drops the artifact's 8-step / `flow_shift=6`
contract as well. Without the field, 8 steps on the turbo schedule produces a dark blurred clip —
which an earlier version of this file wrongly attributed to vLLM-Omni's XPU path.

Measured on 2× Arc Pro B60, Ref2VA 135 GiB partition at `~/work/models/MiniMax-H3/Ref2VA`,
TP2 + DLO (12 resident layers) + tiled VAE, turbo LoRA
`minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors`, seed 1001, 1376×768:

| variant | steps | E2E | peak/rank | frame mean (std) |
|---|---:|---:|---:|---|
| FP8 + adapter bound in request | 8 | 338–349 s | 14.7 GiB | **114.6 (68.4)** |
| BF16 + adapter bound in request | 8 | 375 s | 18.7 GiB | **114.7 (68.9)** |
| FP8, base model, `flow_shift=12` | 50 | 943 s | 12.4 GiB | 71.8 (71.8) |
| FP8/BF16, adapter **not** bound | 9 | 360–371 s | 14.8–18.7 GiB | 40.8–43.0 (~30) |

All of these return HTTP 200 with a well-formed 124-frame mp4 carrying AAC audio and no
`DEVICE_LOST` or xe/GuC faults; only the last row looks wrong, and that is the harness bug.

**Never load the whole 135 GiB partition across many cards at once.** Doing so with
`device_map="balanced"` over 16 XPUs wedged three GPUs (`xe ... device wedged, needs recovery`,
GuC queue timeouts) and required a reboot; a second run then died mid-eval and took the box down
again. Stream one block at a time on one card instead — that shape has produced zero driver
events across repeated runs.

The diffusers 5-stage harness at `~/work/simon/h3/minimax-h3-diffuser` is the useful reference
oracle (layer-streamed, one card): `H3_WEIGHTS_ROOT=~/work/models/MiniMax-H3`,
`H3_RUNS_ROOT=<repo>/diffusers-output/runs`, spec at `diffusers-output/cfg/`.

Recipes are written for `main`, so prefer the installed code over the docs when they disagree:
the 8-step artifact requires `num_inference_steps=8` (not the 9 sigma points the recipe table
lists), and `_comfyui_` LoRA exports are refused by name.

## Other verified XPU facts

- `xpu-smi`/Sysman can't read telemetry on these cards — `gpu_monitor.py` reads `xe` sysfs instead.
- Qwen2.5-Omni needs 3 stages (thinker 16.7 GiB + talker 3.6 GiB + code2wav 1.5 GiB). On 24 GiB
  cards all three cannot share one device, and a global `--gpu-memory-utilization` overrides every
  per-stage value — use `--deploy-config deploy/qwen2_5_omni_2xb60.yaml` instead of CLI memory flags.
- No native int4 matmul (`torch.quint4x2`); emulate 4-bit GEMM as two int8 GEMMs.
- ComfyUI runs **no** native quantized kernels on XPU (`supports_int8_compute()` hardcodes `False`
  for `is_intel_xpu()`), so the `int8_convrot` checkpoints are storage-only: every linear
  dequantizes to BF16/FP16 per forward and the GEMM is BF16. `torch._int_mm` does work on XPU, so
  this is a ComfyUI policy choice, not a missing operator — see `perf.md`.
- SDPA on XPU must avoid the cuDNN backend (`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`); MATH works.
- TP=8 heavy allreduce can permanently drop a PCIe root port to Gen1 and only a power cycle
  recovers it — see `vllm_tp8_pcie_downgrade.md`.

## Where things are

- `ltx_server.py` + `static/index.html` — LTX web UI; state is server-side, browser polls
- `ltx-gen.md` — LTX CLI usage guide · `vllm_tp8_pcie_downgrade.md` — PCIe downgrade repro record
- `vendor/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/CLAUDE.md` — upstream LTX conventions