#!/usr/bin/env bash
# Generate a 720p 5-second MiniMax-H3 clip from a project directory.
#
# Project layout:
#   <project>/input/prompt.txt     required, the whole file is the prompt
#   <project>/input/img/*.png      0 files -> text-to-video, 1..9 -> text+reference-to-video
#   <project>/output/              created here, the mp4 lands in it
#
# Usage:
#   ./gen_video.sh myproject/
#   ./gen_video.sh myproject/ --mode ref2v --sol --steps 8
#   ./gen_video.sh myproject/ --restart --keep-server
#
# Reference images are pushed through POST /upload/image, so the ComfyUI server does
# not have to be restarted (or its --input-directory changed) when you switch projects.
# Forgetting that flag on a *fresh* server is what silently drops the reference rows:
# token count 41042 -> 38970 and frame mean 109 -> 87.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${COMFY_VENV:-$ROOT/.venv-xpu}"
COMFY="$ROOT/vendor/ComfyUI"
GPU="${XPU_GPU:-16}"
PORT="${XPU_PORT:-8288}"
MODE="auto"
STEPS=8
SOL=0
SEED=1001
RESTART=0
KEEP_SERVER=0
WIDTH=1376
HEIGHT=768
LENGTH=124
FPS=24
PROJECT=""

say()  { printf '\033[1;36m[gen]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[gen] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[gen] %s\033[0m\n' "$*" >&2; exit 1; }

usage() { sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --mode)     MODE="$2"; shift 2 ;;
        --steps)    STEPS="$2"; shift 2 ;;
        --seed)     SEED="$2"; shift 2 ;;
        --sol)      SOL=1; shift ;;
        --no-sol)   SOL=0; shift ;;
        --gpu)      GPU="$2"; shift 2 ;;
        --port)     PORT="$2"; shift 2 ;;
        --restart)  RESTART=1; shift ;;
        --keep-server) KEEP_SERVER=1; shift ;;
        --width)    WIDTH="$2"; shift 2 ;;
        --height)   HEIGHT="$2"; shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        -*)         die "unknown option: $1" ;;
        *)          PROJECT="$1"; shift ;;
    esac
done

[ -n "$PROJECT" ] || { usage; die "missing project directory"; }
PROJECT="$(cd "$PROJECT" 2>/dev/null && pwd)" || die "project directory not found"
[ -x "$VENV/bin/python" ] || die "$VENV missing — run ./setup_comfyui_env.sh first"
[ -d "$COMFY" ] || die "$COMFY missing — run ./setup.sh or ./setup_comfyui_env.sh"

PROMPT_FILE="$PROJECT/input/prompt.txt"
IMG_DIR="$PROJECT/input/img"
OUT_DIR="$PROJECT/output"
[ -f "$PROMPT_FILE" ] || die "missing $PROMPT_FILE"
[ -s "$PROMPT_FILE" ] || die "$PROMPT_FILE is empty"
mkdir -p "$OUT_DIR"

shopt -s nullglob
IMAGES=()
for f in "$IMG_DIR"/*.png "$IMG_DIR"/*.PNG; do IMAGES+=("$f"); done
shopt -u nullglob

if [ "$MODE" = auto ]; then
    if [ ${#IMAGES[@]} -eq 0 ]; then MODE=t2v; else MODE=ref2v; fi
fi
case "$MODE" in t2v|ref2v) ;; *) die "--mode must be t2v or ref2v" ;; esac
if [ "$MODE" = ref2v ] && [ ${#IMAGES[@]} -eq 0 ]; then
    die "ref2v needs at least one png in $IMG_DIR"
fi
if [ ${#IMAGES[@]} -gt 9 ]; then
    die "H3 accepts at most 9 reference images, found ${#IMAGES[@]}"
fi

say "project : $PROJECT"
say "mode    : $MODE (${#IMAGES[@]} reference image(s))"
say "output  : $OUT_DIR"

# --- server -----------------------------------------------------------------
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }

if [ "$RESTART" -eq 1 ] && port_busy; then
    say "restarting: killing whatever holds port $PORT"
    fuser -k "$PORT/tcp" >/dev/null 2>&1 || true
    sleep 3
fi

if ! port_busy; then
    say "starting ComfyUI on XPU $GPU (port $PORT)"
    ( "$ROOT/run_xpu_comfy.sh" --gpu "$GPU" --port "$PORT" --out "$OUT_DIR" \
        > "$OUT_DIR/comfyui.log" 2>&1 & )
    for i in $(seq 1 60); do
        curl -sf -m 3 "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1 && break
        sleep 5
    done
    curl -sf -m 3 "http://127.0.0.1:$PORT/system_stats" >/dev/null 2>&1 \
        || die "server did not come up; see $OUT_DIR/comfyui.log"
    say "server ready"
else
    say "reusing the server already on port $PORT"
fi

# --- upload reference images ------------------------------------------------
REF_ARGS=()
for f in "${IMAGES[@]+"${IMAGES[@]}"}"; do
    name="$(basename "$f")"
    # Prefix with the project name so two projects cannot collide in the input dir.
    remote="$(basename "$PROJECT")_$name"
    resp="$(curl -sS -f -X POST "http://127.0.0.1:$PORT/upload/image" \
        -F "image=@$f" -F "overwrite=true" -F "type=input" 2>/dev/null)" \
        || die "upload failed: $f"
    # ComfyUI answers {"name": "...", "subfolder": "...", "type": "input"}.
    got="$(printf '%s' "$resp" | "$VENV/bin/python" -c \
        'import json,sys; d=json.load(sys.stdin); print((d.get("subfolder")+"/" if d.get("subfolder") else "")+d.get("name",""))' 2>/dev/null)"
    [ -n "$got" ] || got="$remote"
    say "uploaded $(basename "$f") -> $got"
    REF_ARGS+=(--ref-image "$got")
done

# --- generate ---------------------------------------------------------------
PREFIX="${MODE}_${WIDTH}x${HEIGHT}_${STEPS}step"
# ${SOL:+--sol} would expand for SOL=0 too (non-empty string); use an explicit list.
GEN_EXTRA=()
[ "$SOL" -eq 1 ] && GEN_EXTRA+=(--sol)
SECS="$(awk "BEGIN{printf \"%.2f\", $LENGTH/$FPS}")"
say "generating ${WIDTH}x${HEIGHT}, ${SECS}s ($LENGTH frames @ $FPS fps), $STEPS steps, seed $SEED, sol=$SOL"
COMFY_URL="http://127.0.0.1:$PORT" "$ROOT/.venv/bin/python" "$ROOT/comfyui_h3_gen.py" \
    --mode "$MODE" \
    --prompt-file "$PROMPT_FILE" \
    --prefix "$PREFIX" \
    --steps "$STEPS" \
    --seed "$SEED" \
    --width "$WIDTH" --height "$HEIGHT" --length "$LENGTH" --fps "$FPS" \
    --base-url "http://127.0.0.1:$PORT" \
    --out-dir "$OUT_DIR" \
    "${GEN_EXTRA[@]+"${GEN_EXTRA[@]}"}" \
    "${REF_ARGS[@]+"${REF_ARGS[@]}"}"

# --- verify the artefact ----------------------------------------------------
# ComfyUI can report success for a run that produced nothing, so check the file.
NEWEST="$(find "$OUT_DIR" -maxdepth 1 -name "*${PREFIX}*.mp4" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "$NEWEST" ] || die "no ${PREFIX}*.mp4 in $OUT_DIR (the job reported success)"
SIZE="$(stat -c%s "$NEWEST")"
[ "$SIZE" -gt 1000000 ] || die "output is only ${SIZE} bytes — likely an error body, not a video: $NEWEST"
say "wrote $NEWEST ($((SIZE/1024/1024)) MiB)"

if [ "$KEEP_SERVER" -eq 0 ]; then
    fuser -k "$PORT/tcp" >/dev/null 2>&1 || true
    say "server stopped"
fi
