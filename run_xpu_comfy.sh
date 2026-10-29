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
#
# Usage:
#   ./run_xpu_comfy.sh                          # card 16, port 8288
#   ./run_xpu_comfy.sh --gpu 17 --port 8388
#   ./run_xpu_comfy.sh --out h3_output/mytest -- --verbose
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

say()  { printf '\033[1;36m[xpu-comfy]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[xpu-comfy] %s\033[0m\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu)  GPU="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --out)  OUT="$2"; shift 2 ;;
        --)     shift; EXTRA=("$@"); break ;;
        -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
say "ComfyUI from $VENV on XPU $GPU ($RENDER_NODE), port $PORT, out $OUT"
exec env ZE_AFFINITY_MASK="$GPU" "$VENV/bin/python" "$COMFY/main.py" \
    --listen 127.0.0.1 --port "$PORT" \
    --disable-auto-launch \
    --input-directory "$IN_DIR" \
    --output-directory "$OUT" \
    --async-offload 2 --reserve-vram 2 \
    "${EXTRA[@]+"${EXTRA[@]}"}"
