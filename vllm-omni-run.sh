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
#   ./vllm-omni-run.sh serve --gpus 1 --precision bf16 --task t2va   # single card
#   ./vllm-omni-run.sh generate --prompt "a fox in the snow" --image h3_input/assets/03-村口-实景.png
#   ./vllm-omni-run.sh run --gpus 2 --precision fp8 --lora 8step
#   ./vllm-omni-run.sh run --gpu-ids 30,31 --precision bf16 --lora none --steps 20
#   ./vllm-omni-run.sh run --profile smoke          # 1024x576, 2 steps, ~2 min (deliberately under-denoised)
#
# --task t2va and --task fl2va share the FL2VA partition: --task-type names the
# partition, extra_params.task names the request. --task ref2va needs the Ref2VA
# partition, so switching between text-only and reference-image generation needs
# a different server, not just a different request.
#
# On one GPU the resident-layer default drops to 2 (12 needs ~15 GB and OOMs on
# a 24 GB card). OMNI_VENV points the script at another venv, e.g. the one
# setup_vllm_omni_env.sh builds.
#
# NOTE (retraction, 2026-10-24): an earlier version of this header claimed the XPU
# diffusion path itself produced dark/blurred video. That was wrong. The adapter
# was never activated -- it must be named in each request's `lora` field, and a
# server-side --lora-path alone is deactivated again by the next request that omits
# it -- so the base model ran the turbo schedule. With the adapter bound (cmd_generate
# always binds it) the output is correct; see perf.md rows 1-2. Dark/blurred remains
# the signature to recognise when a run silently falls back to the base model
# (perf.md rows 4-6), but it is a harness bug, not a runtime fault.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$ROOT/vllm-omni-output"
# Which venv holds vllm/vllm-omni. Override to use a dedicated env built by
# setup_vllm_omni_env.sh (e.g. OMNI_VENV=$ROOT/.venv-omni).
# Which venv holds vllm/vllm-omni: OMNI_VENV wins, then the standalone .venv-omni that
# setup_vllm_omni_env.sh builds, then the combined .venv.
if [ -n "${OMNI_VENV:-}" ]; then
    VENV="$OMNI_VENV"
elif [ -d "$ROOT/.venv-omni" ]; then
    VENV="$ROOT/.venv-omni"
else
    VENV="$ROOT/.venv"
fi
# ~/.bashrc sources oneAPI setvars, whose libur_loader.so.0 no longer exports
# urKernelSetArgLocal — the symbol the venv's intel-sycl-rt libsycl.so.9 imports — and
# LD_LIBRARY_PATH outranks libsycl's RUNPATH=$ORIGIN. Without this every `import torch`
# fails, and with `pipefail` the very first device probe aborts the script *silently*
# (empty output, exit 1). Pin the venv's lib first; load_oneapi() re-prepends it after
# sourcing setvars for the server.
export LD_LIBRARY_PATH="$VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/work/models/MiniMax-H3}"
LORA_DIR="${LORA_DIR:-$ROOT/h3_input/loras}"
PIDFILE="$OUT_DIR/server.pid"
LOG="$OUT_DIR/server.log"
PORT="${PORT:-8090}"
BASE="http://127.0.0.1:$PORT"

# --- defaults (each is a flag below) ---------------------------------------
GPU_COUNT=2
GPU_IDS=""          # explicit device list, e.g. 30,31 or 30:31; overrides the 0..N-1 default
PRECISION="fp8"
TASK="ref2va"
ATTN=""
RESIDENT_LAYERS=12
RESIDENT_LAYERS_SET=0   # so --gpus 1 can pick its own default
DLO=1
LORA="8step"
PROFILE=""
PROMPT="An empty village road in the low morning sun before anyone comes. Photorealistic live-action film look, natural daylight."
IMAGE=("$ROOT/h3_input/assets/03-村口-实景.png")   # ref2va: one or more; --image appends
WIDTH=1376
HEIGHT=768
FPS=24
DURATION=5
STEPS=""
FLOW_SHIFT=""
SAMPLER=""          # euler (server default) | res_multistep; sent only when set
QUALITY=""          # lossless (server default) | high (Cache-DiT); sent only when set
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
    # setvars puts oneAPI 2026.2's libur_loader ahead of the venv's own, and the venv's
    # libsycl.so.9 needs the symbols that 2026.2 dropped (urKernelSetArgLocal), so every
    # `import torch` then fails. Keep the venv's lib first, oneAPI's after it — the same
    # ordering run_vllm_tp.sh uses.
    export LD_LIBRARY_PATH="$VENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

server_pid() {
    [ -f "$PIDFILE" ] || return 1
    local pid; pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && echo "$pid"
}

wait_health() {
    local pid="$1" timeout="${2:-600}" i
    for ((i = 0; i < timeout; i += 5)); do
        # Check liveness BEFORE the probe: another tenant's server can already be
        # listening on this port, in which case /health answers while ours is dead
        # (bind failed) and every later request silently goes to the wrong engine.
        kill -0 "$pid" 2>/dev/null || return 1
        if curl -sf "$BASE/health" >/dev/null 2>&1; then return 0; fi
        sleep 5
    done
    return 1
}

# True when something already accepts connections on this port.
port_in_use() {
    if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
        return 0
    fi
    return 1
}

# --gpu-ids accepts "30,31", "30:31" or "30-31" (a bare range expands). The
# script otherwise assumes devices 0..N-1, which is wrong on a 32-XPU box where
# other tenants hold the low ids.
normalize_gpu_ids() {
    local spec="${1//：/:}"
    if [[ "$spec" =~ ^([0-9]+)[:\-]([0-9]+)$ ]]; then
        local a="${BASH_REMATCH[1]}" b="${BASH_REMATCH[2]}"
        [ "$a" -le "$b" ] || die "--gpu-ids range must be ascending (got '$1')"
        seq -s, "$a" "$b"
    else
        echo "$spec"
    fi
}

# LoRA artifacts are validated by filename, and each mandates its own schedule.
# steps/flow_shift come from the vLLM-Omni recipe table; the server rejects
# mismatches rather than guessing.
lora_file() {
    case "$TASK:$LORA" in
        ref2va:8step) echo "minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors" ;;
        ref2va:4step) echo "minimax_h3_ref2v_turbo_4step_v1.0_768p_bf16.safetensors" ;;
        fl2va:8step|t2va:8step) echo "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors" ;;
        fl2va:4step|t2va:4step) echo "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors" ;;
        *:none)       echo "" ;;
        *)            die "--lora must be 8step, 4step or none (got '$LORA')" ;;
    esac
}

lora_schedule() {
    case "$LORA" in
        8step) echo "8 6" ;;   # artifact spec: 8 denoise steps, video flow_shift 6
        4step) echo "4 6" ;;   # artifact spec: 4 denoise steps, video flow_shift 6
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
        ref2va)      partition="$MODEL_ROOT/Ref2VA" ;;
        fl2va|t2va)  partition="$MODEL_ROOT/FL2VA" ;;
    esac
    [ -d "$partition" ] || die "checkpoint partition missing: $partition (run: $0 download --task $TASK)"

    # A t2va request is served by the fl2va partition; --task-type names the
    # partition, while extra_params.task names the request.
    local SERVER_TASK="$TASK"
    [ "$TASK" = t2va ] && SERVER_TASK=fl2va

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

    if [ "$PROFILE" = "smoke" ]; then
        warn "profile smoke: $STEPS steps with no adapter -- the clip is deliberately under-denoised"
    fi

    mkdir -p "$OUT_DIR"
    local gpu_ids
    if [ -n "$GPU_IDS" ]; then
        gpu_ids="$GPU_IDS"
        GPU_COUNT="$(awk -F, '{print NF}' <<<"$gpu_ids")"
        # Cards can vanish from the pool between runs (other tenants). Without this
        # the engine dies deep inside worker init with a misleading
        # "device index is out of range".
        local avail
        avail="$("$VENV/bin/python" -c 'import torch; print(torch.xpu.device_count())' 2>/dev/null | tail -1 || true)"
        [ -n "$avail" ] || die "cannot query torch.xpu with $VENV/bin/python (import torch failed).
      A libsycl/libur_loader mismatch is the usual cause; see LD_LIBRARY_PATH above."
        if [[ "$avail" =~ ^[0-9]+$ ]]; then
            local id
            for id in ${gpu_ids//,/ }; do
                [ "$id" -lt "$avail" ] || die "device $id does not exist (torch.xpu.device_count()=$avail, valid ids 0..$((avail - 1)))"
            done
        fi
    else
        gpu_ids="$(seq -s, 0 $((GPU_COUNT - 1)))"
    fi

    # 12 resident DiT layers need ~15 GB; on one 24 GB card that evicts the
    # activations and OOMs. 2 is the measured single-card value (19.7 GiB peak,
    # perf.md row 4). Only applies when --resident-layers was not given.
    if [ "$GPU_COUNT" -eq 1 ] && [ "$RESIDENT_LAYERS_SET" -eq 0 ]; then
        RESIDENT_LAYERS=2
        say "single GPU: defaulting to $RESIDENT_LAYERS resident DiT layers (--resident-layers overrides)"
    fi

    # oneAPI's setvars.sh assigns a global array named `args` (and resets
    # positional params), so it must run BEFORE we build the command array.
    load_oneapi

    local srv_args=(
        serve "$partition"
        --omni --trust-remote-code --host 127.0.0.1 --port "$PORT"
        --task-type "$SERVER_TASK"
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
    # NOTE: the adapter is NOT passed here. vLLM-Omni activates LoRA per
    # request from the request's `lora` field; a server-side --lora-path only
    # preloads it and, on its own, leaves every request running the base model.
    # cmd_generate always sends that field.

    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    export VLLM_OMNI_VIDEO_SYNC_TIMEOUT="${VLLM_OMNI_VIDEO_SYNC_TIMEOUT:-14400}"
    # One execute_model RPC must cover the whole request; the 600 s default is
    # short for a 50-step base-model run (~15 min).
    export VLLM_OMNI_DLO_DP_WAVE_TIMEOUT="${VLLM_OMNI_DLO_DP_WAVE_TIMEOUT:-7200}"
    export ZE_AFFINITY_MASK="$gpu_ids"

    say "starting: $TASK on GPU $gpu_ids, precision=$PRECISION, DLO=$DLO/$RESIDENT_LAYERS layers, lora=$LORA"
    if port_in_use; then
        die "port $PORT already has a listener (another tenant's server?); refusing to start."$'\n'"      Ours would fail to bind while /health answered from theirs, and the request would go to the wrong engine."$'\n'"      Pick a free port with --port (e.g. --port 18090)."
    fi
    nohup "$VENV/bin/vllm" "${srv_args[@]}" > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    say "pid $(cat "$PIDFILE"), log $LOG"

    if wait_health "$(cat "$PIDFILE")" 900; then
        say "server ready on $BASE (steps=$STEPS flow_shift=$FLOW_SHIFT)"
    else
        tail -20 "$LOG" >&2
        # A crashed engine can leave the parent alive (orchestrator shutdown hang),
        # which would keep the pidfile occupied and block the next attempt.
        cmd_stop || true
        die "server failed to become healthy; see $LOG"
    fi
}

cmd_generate() {
    server_pid >/dev/null || die "no server running; start one with '$0 serve' or use '$0 run'"
    [ ${#IMAGE[@]} -eq 0 ] || [ -f "${IMAGE[0]}" ] || warn "reference image not found: ${IMAGE[0]} (ref2va needs --image)"

    local steps="${STEPS:-8}" flow="${FLOW_SHIFT:-6}"
    # Non-default sampler/quality must appear in the name: two runs that differ only
    # in sampler otherwise overwrite each other.
    local tag=""
    [ -n "$SAMPLER" ] && [ "$SAMPLER" != "euler" ] && tag+="_${SAMPLER}"
    [ -n "$QUALITY" ] && [ "$QUALITY" != "lossless" ] && tag+="_${QUALITY}"
    OUT_FILE="${OUT_FILE:-$OUT_DIR/${TASK}_${PRECISION}_${WIDTH}x${HEIGHT}_${steps}step${tag}.mp4}"
    mkdir -p "$(dirname "$OUT_FILE")"

    local adapter=""
    if [ "$LORA" != "none" ]; then
        adapter="$LORA_DIR/$(lora_file)"
        [ -f "$adapter" ] || die "LoRA missing: $adapter (run: $0 download --task $TASK)"
    fi

    # Request-scoped sampler/quality live in extra_params (see the vLLM-Omni
    # recipe: 20 steps is only sanctioned together with sampler=res_multistep,
    # and quality selects a Cache-DiT policy -- lossless is the reference path).
    local extra="{\"task\":\"$TASK\",\"duration\":$DURATION,\"audio_flow_shift\":3.0"
    [ -n "$SAMPLER" ] && extra+=",\"sampler\":\"$SAMPLER\""
    [ -n "$QUALITY" ] && extra+=",\"quality\":\"$QUALITY\""
    extra+="}"

    local form=(
        -F "prompt=$PROMPT"
        -F "width=$WIDTH" -F "height=$HEIGHT" -F "fps=$FPS"
        -F "num_inference_steps=$steps" -F "flow_shift=$flow" -F "seed=$SEED"
        -F "extra_params=$extra"
    )
    # Activate the turbo adapter for THIS request. Without this field the server
    # silently runs the base model, which at 8 steps and flow_shift 6 produces a
    # dark, blurred clip (that is the trap that cost several runs).
    if [ -n "$adapter" ]; then
        form+=(-F "lora={\"name\":\"h3-${LORA}\",\"path\":\"${adapter}\"}")
    fi
    if [ "$TASK" = "ref2va" ]; then
        # One reference uses input_reference; several must use repeated
        # input_references fields (the server rejects mixing the two).
        case ${#IMAGE[@]} in
            0) ;;
            1) [ -f "${IMAGE[0]}" ] && form+=(-F "input_reference=@${IMAGE[0]};type=image/png") ;;
            *) for f in "${IMAGE[@]}"; do
                   [ -f "$f" ] || die "reference image not found: $f"
                   form+=(-F "input_references=@$f;type=image/png")
               done ;;
        esac
    elif [ "$TASK" = "fl2va" ] || [ "$TASK" = "t2va" ]; then
        form+=(-F "aspect_ratio=16:9")
    fi

    say "generating ${WIDTH}x${HEIGHT}, ${DURATION}s, $steps steps, seed $SEED, sampler=${SAMPLER:-euler}${QUALITY:+, quality=$QUALITY}"
    local headers="$OUT_DIR/last_headers.txt"
    local rc=0
    curl -sS -D "$headers" -o "$OUT_FILE" --max-time "${GEN_TIMEOUT:-7200}" \
        -X POST "$BASE/v1/videos/sync" "${form[@]}" || rc=$?
    [ "$rc" -eq 0 ] || die "request failed (curl exit $rc); see $headers"

    grep -qi 'HTTP/1.1 200' "$headers" || { tail -5 "$headers" >&2; die "non-200 response"; }
    say "saved $OUT_FILE"
    grep -iE 'x-inference-time-s|x-peak-memory-mb' "$headers" | sed 's/^/  /' || true

    "$VENV/bin/python" - "$OUT_FILE" <<'PY' || true
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
        --gpu-ids) GPU_IDS="$(normalize_gpu_ids "$2")"; shift 2 ;;
        --precision) PRECISION="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --attn) ATTN="$2"; shift 2 ;;
        --resident-layers) RESIDENT_LAYERS="$2"; RESIDENT_LAYERS_SET=1; shift 2 ;;
        --no-dlo) DLO=0; shift ;;
        --lora) LORA="$2"; shift 2 ;;
        --prompt) PROMPT="$2"; shift 2 ;;
        --image) if [ ${#IMAGE[@]} -eq 1 ] && [ "${IMAGE[0]}" = "$ROOT/h3_input/assets/03-村口-实景.png" ]; then IMAGE=(); fi; IMAGE+=("$2"); shift 2 ;;
        --width) WIDTH="$2"; shift 2 ;;
        --height) HEIGHT="$2"; shift 2 ;;
        --fps) FPS="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --flow-shift) FLOW_SHIFT="$2"; shift 2 ;;
        --sampler) SAMPLER="$2"; shift 2 ;;
        --quality) QUALITY="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --out) OUT_FILE="$2"; shift 2 ;;
        --port) PORT="$2"; BASE="http://127.0.0.1:$PORT"; shift 2 ;;
        --keep-server) KEEP_SERVER=1; shift ;;
        --profile)
            case "$2" in
                # smoke disables the LoRA so a 2-step request is actually legal:
                # the LoRA artifacts forbid anything but their own schedule.
                smoke) WIDTH=1024; HEIGHT=576; STEPS=2; FLOW_SHIFT=12; LORA=none; PROFILE=smoke ;;
                quality) WIDTH=1376; HEIGHT=768; PROFILE=quality ;;
                *) die "unknown profile '$2' (smoke|quality)" ;;
            esac
            shift 2 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$TASK" in ref2va|fl2va|t2va) ;; *) die "--task must be ref2va, fl2va or t2va" ;; esac
case "$PRECISION" in fp8|bf16) ;; *) die "--precision must be fp8 or bf16" ;; esac
case "$LORA" in 8step|4step|none) ;; *) die "--lora must be 8step, 4step or none" ;; esac
case "${SAMPLER:-euler}" in euler|res_multistep) ;; *) die "--sampler must be euler or res_multistep" ;; esac
case "${QUALITY:-lossless}" in lossless|high) ;; *) die "--quality must be lossless or high" ;; esac
# A fixed distilled base_schedule (every turbo adapter) is only validated with
# Euler; the server rejects RES for it outright.
if [ "$SAMPLER" = "res_multistep" ] && [ "$LORA" != "none" ]; then
    die "--sampler res_multistep is rejected for turbo adapters (fixed distilled base_schedule requires euler)."$'\n'"      Use --lora none for a RES comparison run."
fi
[ "$GPU_COUNT" -ge 1 ] || die "--gpus must be >= 1"
if [ -n "$GPU_IDS" ]; then
    case "$GPU_IDS" in
        *[!0-9,]*|,*|*,|*,,*) die "--gpu-ids must be a comma list or a range like 30:31 (got '$GPU_IDS')" ;;
    esac
fi

# The base model needs ~50 steps; on a turbo-length schedule it produces the same
# dark/blurred clip as an unbound adapter, which is indistinguishable from the bug
# this script exists to avoid. Only --profile smoke may under-denoise on purpose.
if [ "$LORA" = "none" ] && [ "$PROFILE" != "smoke" ]; then
    requested_steps="${STEPS:-50}"
    if [[ "$requested_steps" =~ ^[0-9]+$ ]] && [ "$requested_steps" -lt 20 ]; then
        die "--lora none with --steps $requested_steps under-denoises (the base model needs ~50 steps, flow_shift=12)."$'\n'"      Use --lora 8step for a short run, or --profile smoke if you only want a plumbing check."
    fi
fi

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