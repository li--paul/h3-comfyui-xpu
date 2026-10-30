#!/usr/bin/env bash
# Build, from scratch, a uv environment that generates MiniMax-H3 video on ONE Arc B60.
#
# What it produces:
#   .venv-comfy/            torch 2.14.0+xpu + ComfyUI deps + Intel's XPU kernel stack
#   vendor/ComfyUI/         ComfyUI checkout (reused if present)
#   models/comfyui-h3/      the H3 weights ComfyUI needs (~64 GB)
#   vendor/ComfyUI/extra_model_paths.yaml   points ComfyUI at models/comfyui-h3
#
# The XPU part (omni_xpu_kernel + Kitchen xpu provider + the OmniXPU node) is NOT
# duplicated here — it calls setup_xpu_comfy.sh with XPU_NO_COPY=1, so there is a single
# implementation of that logic.
#
# Usage:
#   ./setup_comfyui_env.sh                 # full install
#   ./setup_comfyui_env.sh --skip-models   # env only, reuse an existing model dir
#   ./setup_comfyui_env.sh --check         # verify only
#
# Environment: COMFY_VENV, MODELS_DIR, COMFYUI_REF, PYTHON_BIN, UV_BIN.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${COMFY_VENV:-$ROOT/.venv-comfy}"
COMFY="$ROOT/vendor/ComfyUI"
MODELS="${MODELS_DIR:-$ROOT/models/comfyui-h3}"
COMFYUI_REPO="${COMFYUI_REPO:-https://github.com/comfyanonymous/ComfyUI.git}"
COMFYUI_REF="${COMFYUI_REF:-v0.42.0}"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
UV_BIN="${UV_BIN:-uv}"
TORCH_VERSION="${TORCH_VERSION:-2.14.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.29.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
TORCH_XPU_INDEX="https://download.pytorch.org/whl/xpu"
PYPI_INDEX="https://pypi.org/simple"
HF_REPO="Comfy-Org/MiniMax-H3"
LORA_REPO="lightx2v/Minimax-h3-Turbo"

SKIP_MODELS=0
MODE=install

say()  { printf '\033[1;36m[comfy-setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[comfy-setup] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[comfy-setup] %s\033[0m\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-models) SKIP_MODELS=1; shift ;;
        --check) MODE=check; shift ;;
        -h|--help) sed -n '2,22p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

verify() {
    say "verifying the XPU stack in $VENV"
    XPU_VENV="$VENV" "$ROOT/setup_xpu_comfy.sh" --check
    say "verifying the H3 weights in $MODELS"
    local missing=0 f
    for f in \
        diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
        diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors \
        text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors \
        vae/minimax_h3_video_vae_fp16.safetensors \
        vae/minimax_h3_audio_vae_fp32.safetensors \
        loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors \
        loras/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors
    do
        if [ -f "$MODELS/$f" ]; then
            printf '  ok    %s (%s)\n' "$f" "$(du -h "$MODELS/$f" | cut -f1)"
        else
            printf '  MISSING %s\n' "$f"; missing=$((missing + 1))
        fi
    done
    [ "$missing" -eq 0 ] || die "$missing model file(s) missing under $MODELS"
    say "RESULT: PASS"
}

if [ "$MODE" = check ]; then
    [ -x "$VENV/bin/python" ] || die "$VENV missing — run ./setup_comfyui_env.sh first"
    verify
    exit 0
fi

# --- preflight --------------------------------------------------------------
command -v "$UV_BIN" >/dev/null || die "uv is required"
command -v git >/dev/null || die "git is required"
command -v curl >/dev/null || die "curl is required"
command -v docker >/dev/null || die "docker is required (the XPU kernel wheel comes from the image)"
command -v ffmpeg >/dev/null || warn "ffmpeg not found; mp4 muxing may fail"
command -v "$PYTHON_BIN" >/dev/null || die "$PYTHON_BIN not found"

glibc="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
if [ -n "$glibc" ]; then
    "$PYTHON_BIN" - "$glibc" <<'EOF' || die "glibc $glibc is too old for the XPU wheels (need >= 2.34)"
import sys
raise SystemExit(0 if tuple(int(p) for p in sys.argv[1].split('.')[:2]) >= (2, 34) else 1)
EOF
fi

# --- venv + torch -----------------------------------------------------------
if [ ! -x "$VENV/bin/python" ]; then
    say "creating $VENV with $PYTHON_BIN"
    "$UV_BIN" venv --python "$PYTHON_BIN" "$VENV"
else
    say "reusing $VENV"
fi
PY="$VENV/bin/python"

have_torch="$("$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null | tail -1 || true)"
if [ "$have_torch" = "${TORCH_VERSION}+xpu" ]; then
    say "torch ${have_torch} already installed"
else
    say "installing torch ${TORCH_VERSION}+xpu (this is the only place the XPU index is needed)"
    UV_INDEX_STRATEGY=unsafe-best-match "$UV_BIN" pip install \
        --python "$PY" \
        --index-url "$PYPI_INDEX" \
        --extra-index-url "$TORCH_XPU_INDEX" \
        "torch==${TORCH_VERSION}+xpu" \
        "torchvision==${TORCHVISION_VERSION}+xpu" \
        "torchaudio==${TORCHAUDIO_VERSION}+xpu"
fi
"$PY" -c 'import torch; assert torch.xpu.is_available()' \
    || die "torch.xpu.is_available() is False; check the Intel GPU driver"
say "torch $( "$PY" -c 'import torch; print(torch.__version__)' 2>/dev/null | tail -1 ) on $("$PY" -c 'import torch; print(torch.xpu.device_count())' 2>/dev/null | tail -1) XPU device(s)"

# --- ComfyUI ----------------------------------------------------------------
if [ -d "$COMFY/.git" ]; then
    say "ComfyUI checkout present: $COMFY"
else
    say "cloning ComfyUI ($COMFYUI_REF)"
    mkdir -p "$ROOT/vendor"
    if ! git clone --depth 1 --branch "$COMFYUI_REF" "$COMFYUI_REPO" "$COMFY" 2>/dev/null; then
        warn "tag $COMFYUI_REF not found; cloning the default branch instead"
        git clone --depth 1 "$COMFYUI_REPO" "$COMFY"
    fi
fi
say "installing ComfyUI requirements"
"$UV_BIN" pip install --python "$PY" -r "$COMFY/requirements.txt"

# --- Intel XPU kernel stack (single implementation lives in setup_xpu_comfy.sh) ---
say "installing the XPU kernel stack in place (omni_xpu_kernel + Kitchen xpu provider + OmniXPU)"
XPU_VENV="$VENV" XPU_SRC_VENV="$VENV" XPU_NO_COPY=1 "$ROOT/setup_xpu_comfy.sh"

# --- models -----------------------------------------------------------------
if [ "$SKIP_MODELS" -eq 1 ]; then
    warn "--skip-models: not downloading; expecting $MODELS to be populated"
else
    command -v hf >/dev/null || die "the 'hf' CLI is required to download weights (pip install huggingface_hub)"
    mkdir -p "$MODELS"/{diffusion_models,text_encoders,vae,loras}
    say "downloading H3 weights into $MODELS (~64 GB on first run)"
    hf download "$HF_REPO" --include \
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors" \
        "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" \
        "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" \
        "vae/minimax_h3_video_vae_fp16.safetensors" \
        "vae/minimax_h3_audio_vae_fp32.safetensors" \
        --local-dir "$MODELS" >/dev/null
    hf download "$LORA_REPO" --include \
        "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors" \
        "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors" \
        --local-dir "$MODELS/loras" >/dev/null
    # hf keeps the repo-relative path, which nests loras/ one level deeper.
    [ -d "$MODELS/loras/loras" ] && mv "$MODELS/loras/loras/"*.safetensors "$MODELS/loras/" && rmdir "$MODELS/loras/loras"
fi

# --- point ComfyUI at the weights ------------------------------------------
cat > "$COMFY/extra_model_paths.yaml" <<EOF
h3:
    base_path: $MODELS/
    diffusion_models: diffusion_models/
    text_encoders: text_encoders/
    clip: text_encoders/
    vae: vae/
    loras: loras/
EOF
say "wrote $COMFY/extra_model_paths.yaml"

verify
say "done. generate with: ./gen_video.sh <project-dir>"
