#!/usr/bin/env bash
# Build, from scratch, a uv environment that generates MiniMax-H3 video through
# vLLM-Omni on ONE Intel Arc B60.
#
# What it produces:
#   .venv-omni              torch 2.14.0+xpu + vllm + vllm-omni (source builds)
#   vendor/vllm             checkout at VLLM_VER   (reused if already present)
#   vendor/vllm-omni        checkout at VLLM_OMNI_VER
#   envs/vllm-omni/uv.lock  resolved lockfile
#
# Everything is declared in envs/vllm-omni/pyproject.toml; this script only
# clones the sources, loads the Intel oneAPI toolchain (needed to build vllm for
# XPU) and delegates to uv.
#
# Usage:
#   ./setup_vllm_omni_env.sh                 # full install
#   ./setup_vllm_omni_env.sh --check         # verify only, change nothing
#   ./setup_vllm_omni_env.sh --force         # recreate the venv from scratch
#   ./setup_vllm_omni_env.sh --skip-models   # do not require the weights
#   ./setup_vllm_omni_env.sh --download      # also fetch the 270 GB of weights
#
# Environment: OMNI_VENV, VLLM_VER, VLLM_OMNI_VER, MODEL_ROOT, PYTHON_BIN, UV_BIN.
#
# `uv sync` compiles vllm and vllm-omni from source: 30-60 minutes, and it is
# slow rather than hung. There is no XPU wheel for either on PyPI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$ROOT/envs/vllm-omni"
VENV="${OMNI_VENV:-$ROOT/.venv-omni}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/work/models/MiniMax-H3}"
LORA_DIR="${LORA_DIR:-$ROOT/h3_input/loras}"

# vLLM 0.31+ is the first line that builds against torch 2.14; vllm-omni
# 0.32.0rc1 is the matching release (there is no stable omni on vLLM 0.32 yet).
# These must match the dependency-metadata versions in envs/vllm-omni/pyproject.toml.
VLLM_VER="${VLLM_VER:-v0.32.0}"
VLLM_OMNI_VER="${VLLM_OMNI_VER:-v0.32.0rc1}"
PYTHON_BIN="${PYTHON_BIN:-3.12}"
UV_BIN="${UV_BIN:-uv}"

MODE=install
FORCE=0
NEED_MODELS=1
DOWNLOAD=0

say()  { printf '\033[1;36m[omni-setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[omni-setup] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[omni-setup] %s\033[0m\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --check)       MODE=check; shift ;;
        --force)       FORCE=1; shift ;;
        --skip-models) NEED_MODELS=0; shift ;;
        --download)    DOWNLOAD=1; shift ;;
        -h|--help)     sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)             die "unknown option: $1" ;;
    esac
done

# --- verification -----------------------------------------------------------
verify() {
    say "verifying the environment in $VENV"
    "$VENV/bin/python" - <<'PY'
import importlib.metadata as md
import sys
from pathlib import Path

fail = []

def show(label, value):
    print(f"  {label:<18} {value}")

import torch
show("torch", torch.__version__)
if not torch.__version__.endswith("+xpu"):
    fail.append(f"torch is {torch.__version__}, not an XPU build")
show("xpu available", torch.xpu.is_available())
if not torch.xpu.is_available():
    fail.append("torch.xpu.is_available() is False")
else:
    show("devices", torch.xpu.device_count())

# `import triton` must resolve to triton-xpu. openai-whisper and xgrammar pull in
# the stock triton unless [tool.uv] exclude-dependencies strips it, and the two
# packages are mutually exclusive -- a stock triton shadows the XPU kernels.
# So the check is that the *module* imports while the stock *distribution* is
# absent; its presence is the failure, not its absence.
try:
    import triton
    show("triton", f"{triton.__version__} (module)")
    try:
        md.version("triton-xpu")
    except md.PackageNotFoundError:
        fail.append("triton-xpu is not installed")
    try:
        stock = md.version("triton")
        fail.append(f"stock triton {stock} is installed and shadows triton-xpu")
    except md.PackageNotFoundError:
        pass
except Exception as exc:
    fail.append(f"triton unusable: {exc}")

import vllm
show("vllm", vllm.__version__)
import vllm_omni
show("vllm_omni", getattr(vllm_omni, "__version__", "?"))

# torchcodec needs the system FFmpeg shared libraries, not just the static
# binary imageio-ffmpeg ships; without them the import dies on libavutil.so.60.
try:
    import torchcodec  # noqa: F401
    show("torchcodec", "ok")
except Exception as exc:
    fail.append(f"import torchcodec failed: {exc}")

if fail:
    print("\nRESULT: FAIL —", "; ".join(fail))
    raise SystemExit(1)
print("\nRESULT: PASS")
PY

    # `vllm serve --omni` only exists once vllm-omni is installed alongside vllm.
    # --task-type lives in a config group, so the top-level help does not list it.
    if ! "$VENV/bin/vllm" serve --omni --help=all 2>&1 | grep -q -- "--task-type"; then
        die "vllm serve --omni does not accept --task-type; vllm-omni is not wired in"
    fi
    say "vllm serve --omni accepts --task-type"

    if [ "$NEED_MODELS" -eq 1 ]; then
        say "checking weights in $MODEL_ROOT"
        local missing=0 part
        for part in FL2VA Ref2VA; do
            if [ -f "$MODEL_ROOT/$part/model_index.json" ]; then
                printf '  ok      %s (%s)\n' "$part" "$(du -sh "$MODEL_ROOT/$part" 2>/dev/null | cut -f1)"
            else
                printf '  MISSING %s\n' "$part"; missing=$((missing + 1))
            fi
        done
        local f
        for f in minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors \
                 minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors; do
            if [ -f "$LORA_DIR/$f" ]; then
                printf '  ok      loras/%s\n' "$f"
            else
                printf '  MISSING loras/%s\n' "$f"; missing=$((missing + 1))
            fi
        done
        [ "$missing" -eq 0 ] || die "$missing weight file(s) missing; run: ./vllm-omni-run.sh download --task both"
    else
        warn "--skip-models: weights not checked"
    fi
    say "RESULT: PASS"
}

if [ "$MODE" = check ]; then
    [ -x "$VENV/bin/python" ] || die "$VENV missing — run ./setup_vllm_omni_env.sh first"
    verify
    exit 0
fi

# --- preflight --------------------------------------------------------------
command -v "$UV_BIN" >/dev/null || die "uv is required"
command -v git >/dev/null || die "git is required"
command -v curl >/dev/null || die "curl is required"
[ -f "$PROJ_DIR/pyproject.toml" ] || die "$PROJ_DIR/pyproject.toml missing"

command -v ffmpeg >/dev/null \
    || warn "ffmpeg not found; torchcodec needs the system FFmpeg shared libraries"
ldconfig -p 2>/dev/null | grep -q libavutil \
    || warn "libavutil not in ldconfig; 'import torchcodec' may fail (apt install ffmpeg)"

glibc="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
if [ -n "$glibc" ]; then
    "$UV_BIN" python find "$PYTHON_BIN" >/dev/null 2>&1 \
        || warn "python $PYTHON_BIN not installed yet; uv will fetch it"
    python3 - "$glibc" <<'EOF' || die "glibc $glibc is too old for the XPU wheels (need >= 2.34)"
import sys
raise SystemExit(0 if tuple(int(p) for p in sys.argv[1].split('.')[:2]) >= (2, 34) else 1)
EOF
fi

# --- source checkouts -------------------------------------------------------
mkdir -p "$ROOT/vendor"
[ -d "$ROOT/vendor/vllm" ] || {
    say "cloning vllm $VLLM_VER"
    git clone --depth 1 --branch "$VLLM_VER" https://github.com/vllm-project/vllm.git "$ROOT/vendor/vllm"
}
[ -d "$ROOT/vendor/vllm-omni" ] || {
    say "cloning vllm-omni $VLLM_OMNI_VER"
    git clone --depth 1 --branch "$VLLM_OMNI_VER" https://github.com/vllm-project/vllm-omni.git "$ROOT/vendor/vllm-omni"
}

# --- venv -------------------------------------------------------------------
if [ "$FORCE" -eq 1 ] && [ -d "$VENV" ]; then
    warn "--force: removing $VENV"
    rm -rf "$VENV"
fi

# Intel oneAPI toolchain + oneCCL (needed to build vllm for XPU). No-op when the
# environment is already sourced. oneAPI's scripts are not `set -u` safe (they
# read unset vars such as OCL_ICD_FILENAMES without a default).
if [ -f /opt/intel/oneapi/setvars.sh ]; then
    set +u
    source /opt/intel/oneapi/setvars.sh --force
    source /opt/intel/oneapi/ccl/2022.1/env/vars.sh --force 2>/dev/null || true
    set -u
fi

# setvars prepends oneAPI 2026.2's lib dir, and its libur_loader.so.0 no longer exports
# urKernelSetArgLocal -- the symbol the venv's intel-sycl-rt libsycl.so.9 imports. Since
# LD_LIBRARY_PATH outranks libsycl's RUNPATH=$ORIGIN, vllm's setup.py dies on
# `import torch` inside the build (no-build-isolation means that build runs in $VENV).
# Put the venv's lib first and leave oneAPI's behind it: icpx still resolves, torch works.
export LD_LIBRARY_PATH="$VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$UV_BIN" python install "$PYTHON_BIN" >/dev/null 2>&1 || true

say "syncing $PROJ_DIR into $VENV"
say "this compiles vllm and vllm-omni from source: 30-60 minutes, slow not hung"
UV_PROJECT_ENVIRONMENT="$VENV" "$UV_BIN" sync --project "$PROJ_DIR"

# --- torch-xpu boolean-mask workaround --------------------------------------
# torch-xpu's nonzero() returns the right count but a garbage, unsorted tail for many mask
# lengths on this runtime (64 fine, 65-1000 wrong, 4096/41042 fine). The H3 text encoder
# fills its image placeholders with masked_scatter and injects the DeepStack features with
# masked writes, so a ref2va request aborts inside the index kernel
# (`Indexing.h:622 ... index out of bounds`) with no Python frame to point at. Resolve
# those index lists on the CPU instead: the masks are boolean, so it costs nothing.
# Idempotent, and it lives here rather than in the venv by hand so a rebuild keeps it.
patch_h3_masks() {
    local root="$VENV/lib/python3.12/site-packages/vllm_omni/diffusion/models/minimax_h3"
    if [ ! -f "$root/encoder.py" ]; then
        warn "vllm_omni's minimax_h3 package not found under $root — skipping the mask patch"
        return
    fi
    say "patching vllm_omni's H3 boolean-mask indexing (torch-xpu nonzero workaround)"
    "$VENV/bin/python" - "$root" <<'PYMASK' || die "failed to patch vllm_omni's H3 masking"
"""Rewrite vLLM-Omni's H3 boolean-mask indexing to CPU-computed indices.

Two independent problems are handled here, each with its own sentinel so an existing
install can pick up a new hunk without being repatched from scratch:

1. `_xpu_masks_cpu_index` -- torch-xpu's nonzero() returns wrong indices for many mask
   lengths on this runtime, so boolean-mask selects/writes in the H3 encoder abort inside
   the index kernel or silently take the wrong rows. Resolve the index list on the CPU.

2. `_masked_feature_guard` -- defensive. `_aggregate_deepstack` ORs the image and video
   masks whenever both exist, then adds image rows at union positions; if one modality has
   no features the shapes disagree. Only take a mask when we have the features for it.
"""
from __future__ import annotations

import sys
from pathlib import Path

HELPER = '''"""CPU-computed index lists for boolean masks.

torch-xpu's nonzero() returns the right count but a garbage, unsorted tail for many mask
lengths on this runtime (64 fine, 65-1000 wrong, 4096/41042 fine). Boolean-mask writes
then abort inside the index kernel, and boolean-mask reads silently return the wrong
rows. A boolean mask is tiny, so resolving its index list on the CPU costs nothing.
"""

from __future__ import annotations

import torch


def idx(mask: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """Sorted index list of a boolean mask, computed on the CPU when it lives on XPU."""
    flat = mask.reshape(-1) if mask.dim() > 1 else mask
    if flat.device.type != "xpu":
        return flat.nonzero(as_tuple=True)[0]
    return flat.cpu().nonzero(as_tuple=True)[0].to(device or mask.device)


def select(t: torch.Tensor, mask: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Equivalent of t[mask] for a boolean mask along `dim`."""
    if t.device.type != "xpu" or mask.dtype is not torch.bool:
        return t[mask]
    return t.index_select(dim, idx(mask, t.device))


def assign(dst: torch.Tensor, mask: torch.Tensor, values: torch.Tensor, dim: int = 0) -> None:
    """Equivalent of dst[mask] = values for a boolean mask along `dim`, in place."""
    if dst.device.type != "xpu" or mask.dtype is not torch.bool:
        dst[mask] = values
        return
    dst.index_copy_(dim, idx(mask, dst.device), values.to(dst.dtype))


def masked_scatter(src: torch.Tensor, mask: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Equivalent of src.masked_scatter(mask, values) for an element-wise mask."""
    if src.device.type != "xpu" or mask.dtype is not torch.bool:
        return src.masked_scatter(mask, values)
    flat = idx(mask, src.device)
    out = src.reshape(-1).clone()
    out.index_copy_(0, flat, values.reshape(-1).to(out.dtype))
    return out.view(src.shape)
'''

IDX = "_xpu_masks_cpu_index"
GUARD = "_masked_feature_guard"
IMPORT_OLD = "import torch\n"
IMPORT_NEW = "import torch\n\nfrom . import _xpu_masks  # " + IDX + "\n"

# (file, old, new) -- applied only when `new` is absent, so each hunk is independently
# idempotent and a tree patched by an older version of this script still gains new ones.
PATCHES = [
    # --- 1. boolean-mask indexing through CPU indices -------------------------
    ("encoder.py",
     "            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)",
     "            # " + IDX + "\n"
     "            inputs_embeds = _xpu_masks.masked_scatter(inputs_embeds, image_mask, image_embeds)"),
    ("encoder.py",
     "            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)",
     "            # " + IDX + "\n"
     "            inputs_embeds = _xpu_masks.masked_scatter(inputs_embeds, video_mask, video_embeds)"),
    ("encoder.py",
     "            image_mask_joint = image_mask[visual_pos_masks]",
     "            # " + IDX + "\n"
     "            image_mask_joint = _xpu_masks.select(image_mask, visual_pos_masks)"),
    ("encoder.py",
     "            video_mask_joint = video_mask[visual_pos_masks]",
     "            # " + IDX + "\n"
     "            video_mask_joint = _xpu_masks.select(video_mask, visual_pos_masks)"),
    ("encoder.py",
     "                embed_joint[image_mask_joint, :] = img_embed",
     "                # " + IDX + "\n"
     "                _xpu_masks.assign(embed_joint, image_mask_joint, img_embed)"),
    ("encoder.py",
     "                embed_joint[video_mask_joint, :] = vid_embed",
     "                # " + IDX + "\n"
     "                _xpu_masks.assign(embed_joint, video_mask_joint, vid_embed)"),
    ("encoder.py",
     "        hidden_states[visual_pos_masks, :] = hidden_states[visual_pos_masks, :] + visual_embeds",
     "        # " + IDX + "\n"
     "        # The mask has one entry per sequence position; hidden_states is [seq, D] or\n"
     "        # [1, seq, D], so the selection runs along whichever dim matches the mask.\n"
     "        _ds_idx = _xpu_masks.idx(visual_pos_masks, hidden_states.device)\n"
     "        _ds_dim = 0 if hidden_states.shape[0] == visual_pos_masks.numel() else 1\n"
     "        _ds_sel = hidden_states.index_select(_ds_dim, _ds_idx)\n"
     "        _ds_add = visual_embeds if _ds_sel.shape == visual_embeds.shape else visual_embeds.unsqueeze(0)\n"
     "        hidden_states.index_copy_(_ds_dim, _ds_idx, _ds_sel + _ds_add)"),
    ("minimax_h3_transformer.py",
     "        img_global_pos = img_pos[img_mask]",
     "        # " + IDX + "\n        img_global_pos = _xpu_masks.select(img_pos, img_mask)"),
    ("minimax_h3_transformer.py",
     "        audio_global_pos = audio_pos[audio_mask]",
     "        # " + IDX + "\n        audio_global_pos = _xpu_masks.select(audio_pos, audio_mask)"),
    ("minimax_h3_transformer.py",
     "            text_pos[text_mask] - local_start,",
     "            # " + IDX + "\n"
     "            _xpu_masks.select(text_pos, text_mask) - local_start,"),
    ("minimax_h3_transformer.py",
     "            torch.nonzero(text_mask, as_tuple=False).view(-1),",
     "            _xpu_masks.idx(text_mask, text_pos.device),  # " + IDX),
    ("denoise_loop.py",
     "            target_positions = positions[update_mask]",
     "            # " + IDX + "\n"
     "            target_positions = _xpu_masks.select(positions, update_mask)"),

    # --- 1b. MRoPE position ids in the encoder's _get_rope_index ---------------
    ("encoder.py",
     "            current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]\n"
     "            input_token_type = input_token_type[attention_mask[batch_idx].bool()]",
     "            # " + IDX + "\n"
     "            _am = attention_mask[batch_idx].bool()\n"
     "            current_input_ids = _xpu_masks.select(current_input_ids, _am)\n"
     "            input_token_type = _xpu_masks.select(input_token_type, _am)"),
    ("encoder.py",
     "            position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)",
     "            # " + IDX + "\n"
     "            _am = attention_mask[batch_idx].bool()\n"
     "            _pidx = _xpu_masks.idx(_am, position_ids.device)\n"
     "            position_ids[:, batch_idx].index_copy_(1, _pidx, llm_positions.to(position_ids.device))"),

    # --- 2. only pair a mask with features that exist ------------------------
    ("encoder.py",
     "        if image_mask is not None and video_mask is not None:\n"
     "            image_mask = image_mask[..., 0]\n"
     "            video_mask = video_mask[..., 0]\n"
     "            visual_pos_masks = image_mask | video_mask",
     "        # " + GUARD + ": only pair a mask with features that exist, or the union mask\n"
     "        # and the per-modality embeddings disagree in length.\n"
     "        _has_image = bool(deepstack_image_embeds)\n"
     "        _has_video = bool(deepstack_video_embeds)\n"
     "        if image_mask is not None and video_mask is not None and _has_image and _has_video:\n"
     "            image_mask = image_mask[..., 0]\n"
     "            video_mask = video_mask[..., 0]\n"
     "            visual_pos_masks = image_mask | video_mask"),
    ("encoder.py",
     "        if image_mask is not None:\n"
     "            image_mask = image_mask[..., 0]\n"
     "            return image_mask, deepstack_image_embeds\n"
     "        if video_mask is not None:\n"
     "            video_mask = video_mask[..., 0]\n"
     "            return video_mask, deepstack_video_embeds",
     "        if image_mask is not None and _has_image:\n"
     "            image_mask = image_mask[..., 0]\n"
     "            return image_mask, deepstack_image_embeds\n"
     "        if video_mask is not None and _has_video:\n"
     "            video_mask = video_mask[..., 0]\n"
     "            return video_mask, deepstack_video_embeds"),
]


# --- 3. remaining mask sites in the denoise loop, latent mask and pipeline ----
PATCHES += [
    ("denoise_loop.py",
     "            timesteps[positions[~update_mask]] = condition_timestep",
     "            # " + IDX + "\n"
     "            timesteps[_xpu_masks.select(positions, ~update_mask)] = condition_timestep"),
    ("denoise_loop.py",
     "        video_rows[~update] = cond_anchor",
     "        # " + IDX + "\n        _xpu_masks.assign(video_rows, ~update, cond_anchor)"),
    ("denoise_loop.py",
     "        audio_rows[~audio_update] = audio_anchor",
     "        # " + IDX + "\n        _xpu_masks.assign(audio_rows, ~audio_update, audio_anchor)"),
    ("denoise_loop.py",
     "            mv_video_t = v_video.float()[update]",
     "            # " + IDX + "\n            mv_video_t = _xpu_masks.select(v_video.float(), update)"),
    ("denoise_loop.py",
     "            mv_audio_t = v_audio.float()[audio_update]",
     "            # " + IDX + "\n            mv_audio_t = _xpu_masks.select(v_audio.float(), audio_update)"),
    ("denoise_loop.py",
     "                    video_rows[update],",
     "                    # " + IDX + "\n                    _xpu_masks.select(video_rows, update),"),
    ("denoise_loop.py",
     "                    model_video_rows[update],",
     "                    # " + IDX + "\n                    _xpu_masks.select(model_video_rows, update),"),
    ("denoise_loop.py",
     "            new_target = video_solver.step(video_rows[update], x0_video, step)",
     "            # " + IDX + "\n"
     "            new_target = video_solver.step(_xpu_masks.select(video_rows, update), x0_video, step)"),
    ("denoise_loop.py",
     "            video_rows[update] = new_target",
     "            # " + IDX + "\n            _xpu_masks.assign(video_rows, update, new_target)"),
    ("denoise_loop.py",
     "                video_rows[~update] = cond_anchor  # per-step imgvid cond reset",
     "                # " + IDX + "\n"
     "                _xpu_masks.assign(video_rows, ~update, cond_anchor)  # per-step imgvid cond reset"),
    ("denoise_loop.py",
     "                    audio_rows[audio_update],",
     "                    # " + IDX + "\n                    _xpu_masks.select(audio_rows, audio_update),"),
    ("denoise_loop.py",
     "                    model_audio_rows[audio_update],",
     "                    # " + IDX + "\n                    _xpu_masks.select(model_audio_rows, audio_update),"),
    ("denoise_loop.py",
     "            new_audio = audio_solver.step(audio_rows[audio_update], x0_audio, step)",
     "            # " + IDX + "\n"
     "            new_audio = audio_solver.step(_xpu_masks.select(audio_rows, audio_update), x0_audio, step)"),
    ("denoise_loop.py",
     "            audio_rows[audio_update] = new_audio if positive.locked_audio_rows is None else positive.locked_audio_rows",
     "            # " + IDX + "\n"
     "            _xpu_masks.assign(audio_rows, audio_update,\n"
     "                              new_audio if positive.locked_audio_rows is None else positive.locked_audio_rows)"),
    ("denoise_loop.py",
     "                audio_rows[~audio_update] = audio_anchor  # per-step audio ref reset",
     "                # " + IDX + "\n"
     "                _xpu_masks.assign(audio_rows, ~audio_update, audio_anchor)  # per-step audio ref reset"),
    ("latent_mask.py",
     "    model_rows[update_mask], target_timesteps = edit.prepare(\n        rows[update_mask],",
     "    # " + IDX + "\n"
     "    model_rows[update_mask], target_timesteps = edit.prepare(\n"
     "        _xpu_masks.select(rows, update_mask),"),
    ("pipeline_minimax_h3.py",
     "            full_video[branch.update_mask] = initial_video",
     "            # " + IDX + "\n"
     "            _xpu_masks.assign(full_video, branch.update_mask, initial_video)"),
    ("pipeline_minimax_h3.py",
     "            full_audio[branch.audio_update_mask] = initial_audio",
     "            # " + IDX + "\n"
     "            _xpu_masks.assign(full_audio, branch.audio_update_mask, initial_audio)"),
]

root = Path(sys.argv[1])
if not (root / "encoder.py").is_file():
    raise SystemExit(f"  {root}/encoder.py not found")

(helper := root / "_xpu_masks.py").write_text(HELPER)

by_file: dict[str, list[tuple[str, str]]] = {}
for name, old, new in PATCHES:
    by_file.setdefault(name, []).append((old, new))

for name, pairs in by_file.items():
    path = root / name
    text = path.read_text()
    applied = 0
    for old, new in pairs:
        if new in text:
            continue
        if old not in text:
            raise SystemExit(f"  pattern not found in {name}: {old.strip().splitlines()[0][:70]!r}")
        text = text.replace(old, new, 1)
        applied += 1
    if IDX in text and IMPORT_NEW not in text:
        if IMPORT_OLD not in text:
            raise SystemExit(f"  import anchor not found in {name}")
        text = text.replace(IMPORT_OLD, IMPORT_NEW, 1)
        applied += 1
    if applied:
        path.write_text(text)
        print(f"  patched {name} ({applied} hunk(s))")
    else:
        print(f"  already patched: {name}")

PYMASK
}

patch_h3_masks

# --- weights ----------------------------------------------------------------
if [ "$DOWNLOAD" -eq 1 ]; then
    say "downloading weights (~270 GB for both partitions)"
    OMNI_VENV="$VENV" "$ROOT/vllm-omni-run.sh" download --task both
fi

verify
say "done. generate with: OMNI_VENV=$VENV ./gen_video_omni.sh <project-dir>"
