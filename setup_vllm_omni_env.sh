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

"$UV_BIN" python install "$PYTHON_BIN" >/dev/null 2>&1 || true

say "syncing $PROJ_DIR into $VENV"
say "this compiles vllm and vllm-omni from source: 30-60 minutes, slow not hung"
UV_PROJECT_ENVIRONMENT="$VENV" "$UV_BIN" sync --project "$PROJ_DIR"

# --- weights ----------------------------------------------------------------
if [ "$DOWNLOAD" -eq 1 ]; then
    say "downloading weights (~270 GB for both partitions)"
    OMNI_VENV="$VENV" "$ROOT/vllm-omni-run.sh" download --task both
fi

verify
say "done. generate with: OMNI_VENV=$VENV ./gen_video_omni.sh <project-dir>"
