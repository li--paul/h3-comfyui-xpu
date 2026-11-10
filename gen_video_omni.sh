#!/usr/bin/env bash
# Generate a 720p 5-second MiniMax-H3 clip through vLLM-Omni from a project dir.
#
# Project layout:
#   <project>/input/prompt.txt     required, the whole file is the prompt
#   <project>/input/img/*.png      0 files -> text-to-video        (task t2va)
#                                  1..9      -> text+ref-to-video  (task ref2va)
#   <project>/output/              created here, the mp4 lands in it
#
# Usage:
#   ./gen_video_omni.sh myproject/
#   ./gen_video_omni.sh myproject/ --mode ref2v --steps 8 --keep-server
#   ./gen_video_omni.sh myproject/ --restart          # force a server restart
#   ./gen_video_omni.sh myproject/ --precision fp8    # untested on one card
#
# Single card is the default: --gpus 1 --precision bf16 --resident-layers 2.
# That combination is the measured one (perf.md row 4: 1376x768, 8 steps,
# 19.7 GiB peak, 700 s E2E). Two things must NOT be "cleaned up":
#
#   * The recipe's `--diffusion-offload-config '{"mode":"layer",...}'` cannot be
#     combined with the turbo LoRA: any request that binds an adapter fails with
#     "MiniMax-H3 Turbo dynamic LoRA does not support layerwise offload"
#     (minimax_h3/lora.py:298). The guard rejects MODEL_LEVEL and LAYER_WISE but
#     not DISTRIBUTED_LAYER_WISE, so single card + LoRA must use
#     --enable-distributed-layerwise-offload --dlo-resident-layers N.
#   * --quantization fp8 is not the single-card default. On one card the DiT
#     streams over PCIe every step; fp8 would halve that traffic but was never
#     measured on one card, so it stays opt-in.
#
# text-to-video and reference-to-video need DIFFERENT checkpoints (FL2VA vs
# Ref2VA), so they are different servers, not different requests. Reusing the
# wrong one would either 400 or silently serve the wrong model, so this script
# tracks which task the running server was started for and refuses to mix them.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$ROOT/vllm-omni-run.sh"
OMNI_OUT="$ROOT/vllm-omni-output"
PIDFILE="$OMNI_OUT/server.pid"
TASKFILE="$OMNI_OUT/server.task"

GPU="${XPU_GPU:-16}"
# 8090 is vllm-omni-run.sh's default but run-ltx-server.sh serves the LTX UI
# there too; 18090 keeps the two from colliding.
PORT="${PORT:-18090}"
MODE="auto"
STEPS=8
SEED=1001
PRECISION="bf16"
DURATION=5
WIDTH=1376
HEIGHT=768
FPS=24
RESTART=0
KEEP_SERVER=0
PROFILE=""
PROJECT=""

say()  { printf '\033[1;36m[omni]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[omni] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[omni] %s\033[0m\n' "$*" >&2; exit 1; }

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --mode)         MODE="$2"; shift 2 ;;
        --steps)        STEPS="$2"; shift 2 ;;
        --seed)         SEED="$2"; shift 2 ;;
        --precision)    PRECISION="$2"; shift 2 ;;
        --gpu)          GPU="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --duration)     DURATION="$2"; shift 2 ;;
        --width)        WIDTH="$2"; shift 2 ;;
        --height)       HEIGHT="$2"; shift 2 ;;
        --profile)      PROFILE="$2"; shift 2 ;;
        --restart)      RESTART=1; shift ;;
        --keep-server)  KEEP_SERVER=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        -*)             die "unknown option: $1" ;;
        *)              PROJECT="$1"; shift ;;
    esac
done

[ -n "$PROJECT" ] || { usage; die "missing project directory"; }
PROJECT="$(cd "$PROJECT" 2>/dev/null && pwd)" || die "project directory not found"
[ -x "$RUNNER" ] || die "$RUNNER missing"

PROMPT_FILE="$PROJECT/input/prompt.txt"
IMG_DIR="$PROJECT/input/img"
OUT_DIR="$PROJECT/output"
[ -f "$PROMPT_FILE" ] || die "missing $PROMPT_FILE"
[ -s "$PROMPT_FILE" ] || die "$PROMPT_FILE is empty"
mkdir -p "$OUT_DIR"
PROMPT="$(cat "$PROMPT_FILE")"

shopt -s nullglob
IMAGES=()
for f in "$IMG_DIR"/*.png "$IMG_DIR"/*.PNG; do IMAGES+=("$f"); done
shopt -u nullglob

if [ "$MODE" = auto ]; then
    if [ ${#IMAGES[@]} -eq 0 ]; then MODE=t2v; else MODE=ref2v; fi
fi
case "$MODE" in t2v|ref2v) ;; *) die "--mode must be auto, t2v or ref2v" ;; esac
if [ "$MODE" = ref2v ] && [ ${#IMAGES[@]} -eq 0 ]; then
    die "ref2v needs at least one png in $IMG_DIR"
fi
if [ ${#IMAGES[@]} -gt 9 ]; then
    die "H3 accepts at most 9 reference images, found ${#IMAGES[@]}"
fi

# t2va and fl2va share the FL2VA partition; ref2va needs its own.
if [ "$MODE" = t2v ]; then TASK=t2va; else TASK=ref2va; fi

say "project : $PROJECT"
say "mode    : $MODE -> task $TASK (${#IMAGES[@]} reference image(s))"
say "output  : $OUT_DIR"

# --- server -----------------------------------------------------------------
server_pid() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && cat "$PIDFILE"; }
server_task() { cat "$TASKFILE" 2>/dev/null || echo ""; }

if [ "$RESTART" -eq 1 ]; then
    say "restarting: stopping the vLLM-Omni server"
    "$RUNNER" stop >/dev/null 2>&1 || true
    rm -f "$TASKFILE"
fi

if pid="$(server_pid)"; then
    running_task="$(server_task)"
    if [ "$running_task" = "$TASK" ]; then
        say "reusing the $TASK server already running (pid $pid)"
    elif [ -z "$running_task" ]; then
        warn "a server is running (pid $pid) but its task is unknown (started before this script?)"
        warn "assuming $TASK; use --restart if the request fails"
    else
        die "the running server serves '$running_task', this project needs '$TASK'.
      They use different checkpoints (FL2VA vs Ref2VA), so it has to restart.
      Re-run with --restart."
    fi
else
    rm -f "$TASKFILE"
    say "starting vLLM-Omni: task $TASK on XPU $GPU, $PRECISION, 1 card, port $PORT"
    SERVE_ARGS=(serve --gpus 1 --precision "$PRECISION" --task "$TASK"
                --gpu-ids "$GPU" --port "$PORT" --lora 8step)
    [ -n "$PROFILE" ] && SERVE_ARGS+=(--profile "$PROFILE")
    "$RUNNER" "${SERVE_ARGS[@]}" || die "server failed to start (see $OMNI_OUT/server.log)"
    echo "$TASK" > "$TASKFILE"
fi

# --- generate ---------------------------------------------------------------
# The LoRA must be named in the request; a server-side --lora-path alone is
# deactivated by the next request that omits the field, which silently runs the
# base model on the turbo schedule (dark, blurred clip, ~41 frame mean).
GEN_ARGS=(generate --task "$TASK" --prompt "$PROMPT" --steps "$STEPS" --seed "$SEED"
          --duration "$DURATION" --width "$WIDTH" --height "$HEIGHT" --fps "$FPS"
          --lora 8step --port "$PORT"
          --out "$OUT_DIR/${MODE}_${WIDTH}x${HEIGHT}_${STEPS}step.mp4")
for f in "${IMAGES[@]+"${IMAGES[@]}"}"; do GEN_ARGS+=(--image "$f"); done

say "generating ${WIDTH}x${HEIGHT}, ${DURATION}s, $STEPS steps, seed $SEED"
"$RUNNER" "${GEN_ARGS[@]}"

# --- verify the artefact ----------------------------------------------------
# vLLM-Omni returns 200 with a well-formed mp4 even when the adapter never
# bound, so check the file and the frame statistics, not just the status code.
OUT_FILE="$OUT_DIR/${MODE}_${WIDTH}x${HEIGHT}_${STEPS}step.mp4"
[ -f "$OUT_FILE" ] || die "no artefact at $OUT_FILE"
SIZE="$(stat -c%s "$OUT_FILE")"
# Size alone rejects a legitimate low-res smoke clip, so prefer a real probe.
if command -v ffprobe >/dev/null 2>&1; then
    NFRAMES="$(ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames -of csv=p=0 "$OUT_FILE" 2>/dev/null | tr -d ',')"
    [ -n "$NFRAMES" ] && [ "$NFRAMES" -gt 0 ] \
        || die "no decodable video stream in $OUT_FILE (${SIZE} bytes): $OUT_FILE"
    say "wrote $OUT_FILE ($((SIZE / 1024 / 1024)) MiB, $NFRAMES frames)"
else
    [ "$SIZE" -gt 1000000 ] || die "output is only ${SIZE} bytes — likely an error body: $OUT_FILE"
    say "wrote $OUT_FILE ($((SIZE / 1024 / 1024)) MiB)"
fi

"${OMNI_VENV:-$ROOT/.venv-omni}/bin/python" - "$OUT_FILE" <<'PY' || true
import sys
try:
    import av, numpy as np
    c = av.open(sys.argv[1]); v = c.streams.video[0]
    fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    a = np.stack(fr).astype(np.float32)
    print(f"  video: {v.codec_context.width}x{v.codec_context.height} {len(fr)} frames "
          f"{float(v.duration * v.time_base):.2f}s")
    print(f"  audio: {'yes' if c.streams.audio else 'MISSING'}")
    print(f"  frame mean/std: {a.mean():.1f}/{a.std():.1f}")
    if a.mean() < 60:
        print("  WARNING: a mean under 60 is the signature of the base model running the")
        print("           turbo schedule — the adapter was probably not bound.")
except Exception as e:
    print(f"  (probe skipped: {e})")
PY

if [ "$KEEP_SERVER" -eq 0 ]; then
    "$RUNNER" stop >/dev/null 2>&1 || true
    rm -f "$TASKFILE"
    say "server stopped"
fi
