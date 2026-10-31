#!/usr/bin/env bash
# Start ComfyUI from the XPU-enabled venv (.venv-xpu) on one specific card.
#
# Four things must be passed or the run silently produces the wrong video — all
# four were learned the hard way:
#   * ZE_AFFINITY_MASK          pin one card; without it the process grabs all
#   * --input-directory         must be h3_input/assets, otherwise the reference
#                               images are not found: token count drops 41042 ->
#                               38970, sink rows 45 -> 13, and the frame
#                               statistics fall from 109/70 to 87/59
#   * --output-directory        keep JZL's "Nth generation" counter per run
#   * --async-offload 2 --reserve-vram 2
#                               the offload policy h3-test.sh uses; changing it
#                               makes timings incomparable
#   * extra_model_paths         the weights are not a flag: ComfyUI reads its own
#                               extra_model_paths.yaml. Pass --models (or set
#                               H3_MODELS) to get a per-run mapping instead, so
#                               the choice is per run rather than "whoever wrote
#                               the global yaml last".
#
# Usage:
#   ./run_xpu_comfy.sh                          # card 16, port 8288
#   ./run_xpu_comfy.sh --gpu 17 --port 8388
#   ./run_xpu_comfy.sh --out h3_output/mytest -- --verbose
#   ./run_xpu_comfy.sh --models /data/ComfyUI-MiniMax-H3
#   ./run_xpu_comfy.sh --build-kernel           # compile the XPU kernel first
#
# Before launching it probes omni_xpu_kernel (target, torch minor, cute, and the
# Sol-Attn op) and refuses to start if anything is off — a degraded kernel is
# silent, just ~1.5x slower. Build it once with
# `./setup_xpu_comfy.sh --kernel-from-source`; --build-kernel does that inline.
#
# Extra args after `--` are forwarded to ComfyUI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${XPU_VENV:-$ROOT/.venv-xpu}"
COMFY="$ROOT/vendor/ComfyUI"
GPU="${XPU_GPU:-16}"
PORT="${XPU_PORT:-8288}"
OUT="$ROOT/h3_output/xpu_comfy"
IN_DIR="$ROOT/h3_input/assets"
EXTRA=()
BUILD_KERNEL="${XPU_KERNEL_AUTOBUILD:-0}"
# Empty means "use ComfyUI's own extra_model_paths.yaml". Set it (or pass
# --models) to get a per-run mapping instead, see the note above exec.
H3_MODELS="${H3_MODELS:-}"

say()  { printf '\033[1;36m[xpu-comfy]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[xpu-comfy] %s\033[0m\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)  GPU="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --out)  OUT="$2"; shift 2 ;;
        --models) H3_MODELS="$2"; shift 2 ;;
        --build-kernel) BUILD_KERNEL=1; shift ;;
        --)     shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1 (use -- to pass args to ComfyUI)" ;;
    esac
done

[ -x "$VENV/bin/python" ] || die "$VENV not found — run ./setup_xpu_comfy.sh first"
[ -d "$COMFY" ] || die "$COMFY not found — run ./setup.sh first"
[ -d "$IN_DIR" ] || die "input assets missing: $IN_DIR"

# The XPU index maps to /dev/dri/renderD(128+index) on this host. Override with
# RENDER_NODE=... on a different topology.
RENDER_NODE="${RENDER_NODE:-/dev/dri/renderD$((128 + GPU))}"
[ -e "$RENDER_NODE" ] || die "$RENDER_NODE does not exist (XPU $GPU not present?)"

# A "busy" render node can be a process that merely holds every handle, so look
# for an *active* holder rather than trusting fuser's presence alone.
holders="$(fuser "$RENDER_NODE" 2>/dev/null | tr -d ' ' || true)"
if [ -n "$holders" ]; then
    say "warning: $RENDER_NODE is held by pid(s): $holders"
    for p in $holders; do
        printf '    %s %s\n' "$p" "$(ps -o etime=,cmd= -p "$p" 2>/dev/null | cut -c1-110)"
    done
    say "another tenant's job will distort timings; continue anyway"
fi

if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    die "port $PORT already in use; pick another with --port"
fi

mkdir -p "$OUT"

# --- model mapping ----------------------------------------------------------
# ComfyUI reads extra_model_paths only at startup, and main.py loads its own
# extra_model_paths.yaml BEFORE any --extra-model-paths-config, appending search
# paths as it goes. A plain override file would therefore be searched *after* the
# global one and lose to any same-named checkpoint. `is_default: true` is what
# puts these paths at the front: folder_paths.add_model_folder_path inserts at
# position 0 when that flag is set, otherwise it appends.
MODEL_ARGS=()
if [ -n "$H3_MODELS" ]; then
    [ -d "$H3_MODELS" ] || die "H3_MODELS is not a directory: $H3_MODELS"
    H3_MODELS="$(cd "$H3_MODELS" && pwd)"
    MODEL_YAML="$OUT/.extra_model_paths.yaml"
    cat > "$MODEL_YAML" <<EOF
h3:
    is_default: true
    base_path: $H3_MODELS/
    diffusion_models: diffusion_models/
    text_encoders: text_encoders/
    clip: text_encoders/
    vae: vae/
    loras: loras/
EOF
    MODEL_ARGS=(--extra-model-paths-config "$MODEL_YAML")
    say "models  : $H3_MODELS (per-run mapping, not the global extra_model_paths.yaml)"
fi

# --- XPU kernel gate --------------------------------------------------------
# A missing or mismatched omni_xpu_kernel does NOT raise: every adapter quietly
# falls back to eager torch code and the same clip takes 70.1 s/step instead of
# 47.2. So this is a hard gate, not a warning.
kernel_probe() {
    "$VENV/bin/python" - <<'PY' 2>/dev/null
import sys
try:
    import torch
    import omni_xpu_kernel as omni
    from omni_xpu_kernel import cute
except Exception:
    sys.exit(1)
want = ".".join(torch.__version__.split("+")[0].split(".")[:2])
got = ".".join(omni.__torch_version__.split(".")[:2])
ok = (omni.__xpu_target__ == "bmg" and got == want
      and omni.is_available() and cute.is_available()
      # b3 moved Sol-Attn to a separate wheel; without this symbol --sol still
      # runs but is dense (39.3 -> 47.2 s/step, no error).
      and hasattr(cute, "sol_attn"))
sys.exit(0 if ok else 1)
PY
}

kernel_desc() {
    "$VENV/bin/python" - <<'PY' 2>/dev/null
import torch, omni_xpu_kernel as omni
from omni_xpu_kernel import cute
print(f"{omni.__version__} on torch {omni.__torch_version__}, "
      f"cute={cute.is_available()}, sol_attn={hasattr(cute, 'sol_attn')}")
PY
}

ensure_kernel() {
    if kernel_probe; then
        say "xpu kernel ok: $(kernel_desc)"
        return
    fi
    if [ "$BUILD_KERNEL" -eq 1 ]; then
        say "xpu kernel missing or built for another torch/target"
        say "building from source (30-90 min, once) — setup_xpu_comfy.sh --kernel-from-source"
        XPU_VENV="$VENV" XPU_SRC_VENV="$VENV" XPU_NO_COPY=1 \
            "$ROOT/setup_xpu_comfy.sh" --kernel-from-source \
            || die "kernel build failed"
        kernel_probe || die "build finished but the probe still fails"
        say "xpu kernel ok: $(kernel_desc)"
    else
        die "omni_xpu_kernel is missing or was built for another torch/target.
      ComfyUI would run ~1.5x slower with NO error (70.1 vs 47.2 s/step).
      Fix:  ./setup_xpu_comfy.sh --kernel-from-source
      or:   ./run_xpu_comfy.sh --build-kernel ..."
    fi
}

ensure_kernel

say "ComfyUI from $VENV on XPU $GPU ($RENDER_NODE), port $PORT, out $OUT"
exec env ZE_AFFINITY_MASK="$GPU" "$VENV/bin/python" "$COMFY/main.py" \
    --listen 127.0.0.1 --port "$PORT" \
    --disable-auto-launch \
    --input-directory "$IN_DIR" \
    --output-directory "$OUT" \
    --async-offload 2 --reserve-vram 2 \
    "${MODEL_ARGS[@]+"${MODEL_ARGS[@]}"}" \
    "${EXTRA[@]+"${EXTRA[@]}"}"
