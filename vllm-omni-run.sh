#!/usr/bin/env bash
# MiniMax-H3 video generation with vLLM-Omni on Intel XPU.
#
# Subcommands:
#   download   fetch the checkpoint partition and/or turbo LoRA
#   serve      start the API server (background, pidfile in vllm-omni-output/)
#   generate   POST a generation request and save the mp4
#   run        download (if needed) -> serve -> wait -> generate -> stop
#   stop       stop the server started by this script
#   status     show server state and last request summary
#
# Examples:
#   ./vllm-omni-run.sh download --task ref2va
#   ./vllm-omni-run.sh serve --gpus 2 --precision fp8 --task ref2va
#   ./vllm-omni-run.sh generate --prompt "a fox in the snow" --image h3_input/assets/03-村口-实景.png
#   ./vllm-omni-run.sh run --gpus 2 --precision fp8 --lora 8step --steps 9
#   ./vllm-omni-run.sh run --profile smoke          # 1024x576, 2 steps, ~2 min
#
# WARNING (2026-10-22): on XPU the engine serves correctly and returns a
# well-formed mp4 with audio, but the picture is degenerate -- dark/blurred. This
# is not a precision, attention-backend or LoRA problem (all three were isolated);
# it looks like the XPU diffusion path itself. ComfyUI remains the working H3
# path; see AGENTS.md. This script exists to make that reproducible and to test
# future fixes quickly.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$ROOT/vllm-omni-output"
MODEL_ROOT="${MODEL_ROOT:-$HOME/work/models/MiniMax-H3}"
LORA_DIR="${LORA_DIR:-$ROOT/h3_input/loras}"
PIDFILE="$OUT_DIR/server.pid"
LOG="$OUT_DIR/server.log"
PORT="${PORT:-8090}"
BASE="http://127.0.0.1:$PORT"

# --- defaults (each is a flag below) ---------------------------------------
GPU_COUNT=2
PRECISION="fp8"
TASK="ref2va"
ATTN=""
RESIDENT_LAYERS=12
DLO=1
LORA="8step"
PROMPT="An empty village road in the low morning sun before anyone comes. Photorealistic live-action film look, natural daylight."
IMAGE="$ROOT/h3_input/assets/03-村口-实景.png"
WIDTH=1376
HEIGHT=768
FPS=24
DURATION=5
STEPS=""
FLOW_SHIFT=""
SEED=1001
OUT_FILE=""
KEEP_SERVER=0
DOWNLOAD_SCOPE=""

say() { printf '\033[1;36m[h3-vl]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[h3-vl] %s\033[0m\n' "$*" >&2; }
die() { printf '\033[1;31m[h3-vl] %s\033[0m\n' "$*" >&2; exit 1; }

usage() {
    awk 'NR>1 { if ($0 !~ /^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
}

# --- helpers ----------------------------------------------------------------
load_oneapi() {
    [ -f /opt/intel/oneapi/setvars.sh ] || return 0
    set +u
    # shellcheck disable=SC1091
    source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
    set -u
}

server_pid() {
    [ -f "$PIDFILE" ] || return 1
    local pid; pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && echo "$pid"
}

wait_health() {
    local pid="$1" timeout="${2:-600}" i
    for ((i = 0; i < timeout; i += 5)); do
        if curl -sf "$BASE/health" >/dev/null 2>&1; then return 0; fi
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 5
    done
    return 1
}

# LoRA artifacts are validated by filename, and each mandates its own schedule.
# steps/flow_shift come from the vLLM-Omni recipe table; the server rejects
# mismatches rather than guessing.
lora_file() {
    case "$TASK:$LORA" in
        ref2va:8step) echo "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors" ;;
        ref2va:4step) echo "minimax_h3_ref2v_turbo_4step_v1.0_768p_bf16.safetensors" ;;
        fl2va:8step)  echo "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors" ;;
        fl2va:4step)  echo "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors" ;;
        *:none)       echo "" ;;
        *)            die "--lora must be 8step, 4step or none (got '$LORA')" ;;
    esac
}

lora_schedule() {
    case "$LORA" in
        8step) echo "9 6" ;;   # 8 forwards -> 9 sigma points, flow_shift 6
        4step) echo "5 6" ;;   # 4 forwards -> 5 sigma points, flow_shift 6
        none)  echo "50 12" ;;
    esac
}

cmd_download() {
    command -v hf >/dev/null || die "the 'hf' CLI is required (pip install huggingface_hub)"
    local scope="${DOWNLOAD_SCOPE:-$TASK}"

    case "$scope" in
        ref2va|both)
            if [ -f "$MODEL_ROOT/Ref2VA/model_index.json" ]; then
                say "Ref2VA already present at $MODEL_ROOT/Ref2VA"
            else
                say "downloading Ref2VA (~135 GiB) -> $MODEL_ROOT"
                hf download MiniMaxAI/MiniMax-H3 --include "Ref2VA/**" --local-dir "$MODEL_ROOT"
            fi
            ;;
    esac
    case "$scope" in
        fl2va|both)
            if [ -f "$MODEL_ROOT/FL2VA/model_index.json" ]; then
                say "FL2VA already present at $MODEL_ROOT/FL2VA"
            else
                say "downloading FL2VA (~135 GiB) -> $MODEL_ROOT"
                hf download MiniMaxAI/MiniMax-H3 --include "FL2VA/**" --local-dir "$MODEL_ROOT"
            fi
            ;;
    esac

    if [ "$LORA" != "none" ]; then
        local f; f="$(lora_file)"
        if [ -f "$LORA_DIR/$f" ]; then
            say "turbo LoRA already present: $f"
        else
            say "downloading turbo LoRA: $f"
            mkdir -p "$LORA_DIR"
            hf download lightx2v/Minimax-h3-Turbo "$f" --local-dir "$LORA_DIR"
        fi
    fi
    say "download complete"
}

cmd_serve() {
    local pid; if pid="$(server_pid)"; then
        die "server already running (pid $pid); use './vllm-omni-run.sh stop' first"
    fi

    local partition
    case "$TASK" in
        ref2va) partition="$MODEL_ROOT/Ref2VA" ;;
        fl2va)  partition="$MODEL_ROOT/FL2VA" ;;
    esac
    [ -d "$partition" ] || die "checkpoint partition missing: $partition (run: $0 download --task $TASK)"

    local lora_path=""
    if [ "$LORA" != "none" ]; then
        local f; f="$(lora_file)"
        lora_path="$LORA_DIR/$f"
        [ -f "$lora_path" ] || die "LoRA missing: $lora_path (run: $0 download --task $TASK)"
    fi

    # A LoRA dictates steps/flow_shift; refuse silent disagreement.
    if [ "$LORA" != "none" ]; then
        read -r want_steps want_flow <<<"$(lora_schedule)"
        if [ -n "$STEPS" ] && [ "$STEPS" != "$want_steps" ]; then
            warn "steps=$STEPS overridden to $want_steps (required by $LORA LoRA)"
        fi
        if [ -n "$FLOW_SHIFT" ] && [ "$FLOW_SHIFT" != "$want_flow" ]; then
            warn "flow_shift=$FLOW_SHIFT overridden to $want_flow (required by $LORA LoRA)"
        fi
        STEPS="$want_steps"; FLOW_SHIFT="$want_flow"
    else
        STEPS="${STEPS:-50}"; FLOW_SHIFT="${FLOW_SHIFT:-12}"
    fi

    mkdir -p "$OUT_DIR"
    local gpu_ids; gpu_ids="$(seq -s, 0 $((GPU_COUNT - 1)))"

    # oneAPI's setvars.sh assigns a global array named `args` (and resets
    # positional params), so it must run BEFORE we build the command array.
    load_oneapi

    local srv_args=(
        serve "$partition"
        --omni --trust-remote-code --host 127.0.0.1 --port "$PORT"
        --task-type "$TASK"
        --num-gpus "$GPU_COUNT"
        --tensor-parallel-size "$GPU_COUNT"
        --text-encoder-tp-size "$GPU_COUNT"
        --vae-patch-parallel-size "$GPU_COUNT"
        --usp 1 --ring 1
        --vae-parallel-mode tile --vae-use-tiling
        --enforce-eager
    )
    [ "$PRECISION" = "fp8" ] && srv_args+=(--quantization fp8)
    [ "$DLO" -eq 1 ] && srv_args+=(--enable-distributed-layerwise-offload --dlo-no-use-allgather \
                                    --dlo-resident-layers "$RESIDENT_LAYERS")
    [ -n "$ATTN" ] && srv_args+=(--diffusion-attention-backend "$ATTN")
    [ -n "$lora_path" ] && srv_args+=(--lora-backend peft --lora-path "$lora_path")

    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_OMNI_VIDEO_SYNC_TIMEOUT="${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-14400}"
    export ZE_AFFINITY_MASK="$gpu_ids"

    say "starting: $TASK on GPU $gpu_ids, precision=$PRECISION, DLO=$DLO/$RESIDENT_LAYERS layers, lora=$LORA"
    nohup "$ROOT/.venv/bin/vllm" "${srv_args[@]}" > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    say "pid $(cat "$PIDFILE"), log $LOG"

    if wait_health "$(cat "$PIDFILE")" 900; then
        say "server ready on $BASE (steps=$STEPS flow_shift=$FLOW_SHIFT)"
    else
        tail -20 "$LOG" >&2
        die "server failed to become healthy; see $LOG"
    fi
}

cmd_generate() {
    server_pid >/dev/null || die "no server running; start one with '$0 serve' or use '$0 run'"
    [ -f "$IMAGE" ] || warn "reference image not found: $IMAGE (ref2va needs --image)"

    local steps="${STEPS:-9}" flow="${FLOW_SHIFT:-6}"
    OUT_FILE="${OUT_FILE:-$OUT_DIR/${TASK}_${PRECISION}_${WIDTH}x${HEIGHT}_${steps}step.mp4}"
    mkdir -p "$(dirname "$OUT_FILE")"

    local form=(
        -F "prompt=$PROMPT"
        -F "width=$WIDTH" -F "height=$HEIGHT" -F "fps=$FPS"
        -F "num_inference_steps=$steps" -F "flow_shift=$flow" -F "seed=$SEED"
        -F "extra_params={\"task\":\"$TASK\",\"duration\":$DURATION,\"audio_flow_shift\":3.0}"
    )
    if [ "$TASK" = "ref2va" ] && [ -f "$IMAGE" ]; then
        form+=(-F "input_reference=@$IMAGE;type=image/png")
    elif [ "$TASK" = "fl2va" ]; then
        form+=(-F "aspect_ratio=16:9")
    fi

    say "generating ${WIDTH}x${HEIGHT}, ${DURATION}s, $steps steps, seed $SEED"
    local headers="$OUT_DIR/last_headers.txt"
    local rc=0
    curl -sS -D "$headers" -o "$OUT_FILE" --max-time "${GEN_TIMEOUT:-7200}" \
        -X POST "$BASE/v1/videos/sync" "${form[@]}" || rc=$?
    [ "$rc" -eq 0 ] || die "request failed (curl exit $rc); see $headers"

    grep -qi 'HTTP/1.1 200' "$headers" || { tail -5 "$headers" >&2; die "non-200 response"; }
    say "saved $OUT_FILE"
    grep -iE 'x-inference-time-s|x-peak-memory-mb' "$headers" | sed 's/^/  /' || true

    "$ROOT/.venv/bin/python" - "$OUT_FILE" <<'PY' || true
import sys
try:
    import av
    c = av.open(sys.argv[1]); v = c.streams.video[0]
    print(f"  video: {v.codec_context.name} {v.codec_context.width}x{v.codec_context.height} "
          f"{v.frames} frames {float(v.duration*v.time_base):.2f}s")
    print(f"  audio: {'yes' if c.streams.audio else 'MISSING'}")
except Exception as e:
    print(f"  (probe skipped: {e})")
PY
    echo "  $(cat "$headers" | grep -c .) header lines -> $headers"
}

cmd_run() {
    cmd_download
    cmd_serve
    local rc=0
    cmd_generate || rc=$?
    if [ "$KEEP_SERVER" -eq 0 ]; then cmd_stop; fi
    return "$rc"
}

cmd_stop() {
    local pid; if pid="$(server_pid)"; then
        say "stopping server pid $pid"
        kill "$pid" 2>/dev/null || true
        for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -9 "$pid" 2>/dev/null || true
    else
        say "no server running"
    fi
    rm -f "$PIDFILE"
}

cmd_status() {
    local pid
    if pid="$(server_pid)"; then
        echo "server : running (pid $pid) on $BASE"
        curl -sf "$BASE/health" >/dev/null 2>&1 && echo "health : ok" || echo "health : unreachable"
    else
        echo "server : stopped"
    fi
    echo "model  : $MODEL_ROOT"
    for part in Ref2VA FL2VA; do
        [ -f "$MODEL_ROOT/$part/model_index.json" ] && echo "  $part: present" || echo "  $part: missing"
    done
    echo "lora   : $(ls "$LORA_DIR" 2>/dev/null | tr '\n' ' ' || echo none)"
    echo "output : $OUT_DIR"
}

# --- arg parsing ------------------------------------------------------------
case "${1:-}" in
    ""|--help|-h|help) usage; [ -n "${1:-}" ] && exit 0 || exit 2 ;;
esac
CMD="$1"; shift

while [ $# -gt 0 ]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --gpus) GPU_COUNT="$2"; shift 2 ;;
        --precision) PRECISION="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --attn) ATTN="$2"; shift 2 ;;
        --resident-layers) RESIDENT_LAYERS="$2"; shift 2 ;;
        --no-dlo) DLO=0; shift ;;
        --lora) LORA="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --image) IMAGE="$2"; shift 2 ;;
        --width) WIDTH="$2"; shift 2 ;;
        --height) HEIGHT="$2"; shift 2 ;;
        --fps) FPS="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --flow-shift) FLOW_SHIFT="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --out) OUT_FILE="$2"; shift 2 ;;
        --port) PORT="$2"; BASE="http://127.0.0.1:$PORT"; shift 2 ;;
        --keep-server) KEEP_SERVER=1; shift ;;
        --profile)
            case "$2" in
                # smoke disables the LoRA so a 2-step request is actually legal:
                # the LoRA artifacts forbid anything but their own schedule.
                smoke) WIDTH=1024; HEIGHT=576; STEPS=2; FLOW_SHIFT=12; LORA=none ;;
                quality) WIDTH=1376; HEIGHT=768 ;;
                *) die "unknown profile '$2' (smoke|quality)" ;;
            esac
            shift 2 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$TASK" in ref2va|fl2va) ;; *) die "--task must be ref2va or fl2va" ;; esac
case "$PRECISION" in fp8|bf16) ;; *) die "--precision must be fp8 or bf16" ;; esac
case "$LORA" in 8step|4step|none) ;; *) die "--lora must be 8step, 4step or none" ;; esac
[ "$GPU_COUNT" -ge 1 ] || die "--gpus must be >= 1"

mkdir -p "$OUT_DIR"

case "$CMD" in
    download) cmd_download ;;
    serve)    cmd_serve ;;
    generate) cmd_generate ;;
    run)      cmd_run ;;
    stop)     cmd_stop ;;
    status)   cmd_status ;;
    *)        usage; die "unknown subcommand: $CMD" ;;
esac