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
#   ./gen_video.sh myproject/ --models /data/ComfyUI-MiniMax-H3
#
# Weights are not a flag in ComfyUI itself: it resolves every checkpoint through
# extra_model_paths.yaml at startup. --models (or H3_MODELS=) writes a per-run
# mapping for this launch instead of depending on whoever wrote the global file
# last. Without either, the base_path in vendor/ComfyUI/extra_model_paths.yaml is
# used. The five files this mode needs are checked before a server is started.
#
# Reference images are pushed through POST /upload/image, so the ComfyUI server does
# not have to be restarted (or its --input-directory changed) when you switch projects.
# Forgetting that flag on a *fresh* server is what silently drops the reference rows:
# token count 41042 -> 38970 and frame mean 109 -> 87.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# COMFY_VENV wins, then .venv-xpu, then .venv-comfy (what setup_comfyui_env.sh builds).
# Only the preflight and the JSON parsing below use this; the server venv is decided by
# run_xpu_comfy.sh's own XPU_VENV, which falls back the same way.
if [ -n "${COMFY_VENV:-}" ]; then
    VENV="$COMFY_VENV"
elif [ -d "$ROOT/.venv-xpu" ]; then
    VENV="$ROOT/.venv-xpu"
elif [ -d "$ROOT/.venv-comfy" ]; then
    VENV="$ROOT/.venv-comfy"
else
    VENV="$ROOT/.venv-xpu"
fi
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
# Empty means "resolve the weights through ComfyUI's own extra_model_paths.yaml".
# Set it (or pass --models) to use a directory without touching that file.
H3_MODELS="${H3_MODELS:-}"

say()  { printf '\033[1;36m[gen]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[gen] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[gen] %s\033[0m\n' "$*" >&2; exit 1; }

usage() { sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

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
        --models)   H3_MODELS="$2"; shift 2 ;;
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

# --- weights ----------------------------------------------------------------
# The generation path takes no model path of its own: ComfyUI resolves every
# checkpoint through extra_model_paths at startup, so a wrong mapping only shows
# up as "value not in list" minutes into a run. Resolve the root here and check
# the exact files this mode needs before starting a server.
MODELS_ROOT="$H3_MODELS"
if [ -z "$MODELS_ROOT" ] && [ -f "$COMFY/extra_model_paths.yaml" ]; then
    MODELS_ROOT="$(sed -n 's/^[[:space:]]*base_path:[[:space:]]*//p' \
        "$COMFY/extra_model_paths.yaml" | head -1 | sed 's:/*$::')"
fi
[ -n "$MODELS_ROOT" ] || die "no weights path.
      Set H3_MODELS=/path (or --models /path), or write a base_path into
      $COMFY/extra_model_paths.yaml. ComfyUI has no H3 weights in its own tree."
[ -d "$MODELS_ROOT" ] || die "weights directory does not exist: $MODELS_ROOT
      (from H3_MODELS/--models, or the base_path in $COMFY/extra_model_paths.yaml)"

REQUIRED=(
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    "vae/minimax_h3_video_vae_fp16.safetensors"
    "vae/minimax_h3_audio_vae_fp32.safetensors"
)
if [ "$MODE" = t2v ]; then
    REQUIRED+=(
        "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
        "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"
    )
else
    REQUIRED+=(
        "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
        "loras/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
    )
fi
MISSING=()
for f in "${REQUIRED[@]}"; do
    [ -f "$MODELS_ROOT/$f" ] || MISSING+=("$MODELS_ROOT/$f")
done
if [ "${#MISSING[@]}" -gt 0 ]; then
    printf '\033[1;31m[gen]\033[0m %d weight file(s) missing for mode %s under %s:\n' \
        "${#MISSING[@]}" "$MODE" "$MODELS_ROOT" >&2
    printf '      %s\n' "${MISSING[@]}" >&2
    die "point H3_MODELS/--models at a complete weights directory"
fi
say "weights : $MODELS_ROOT (${#REQUIRED[@]} files for $MODE)"

# --- server -----------------------------------------------------------------
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }

if [ "$RESTART" -eq 1 ] && port_busy; then
    say "restarting: killing whatever holds port $PORT"
    fuser -k "$PORT/tcp" >/dev/null 2>&1 || true
    sleep 3
fi

if ! port_busy; then
    say "starting ComfyUI on XPU $GPU (port $PORT)"
    START_ARGS=(--gpu "$GPU" --port "$PORT" --out "$OUT_DIR")
    if [ -n "$H3_MODELS" ]; then START_ARGS+=(--models "$H3_MODELS"); fi
    ( "$ROOT/run_xpu_comfy.sh" "${START_ARGS[@]}" \
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
    if [ -n "$H3_MODELS" ]; then
        warn "H3_MODELS is ignored: the running server read its mapping at startup."
        warn "Re-run with --restart to make it use $H3_MODELS."
    fi
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
# comfyui_h3_gen.py is stdlib-only, so any of our venvs can run it; use the same one
# $VENV already resolved instead of hardcoding .venv (which a ComfyUI-only host may
# never have built).
COMFY_URL="http://127.0.0.1:$PORT" "$VENV/bin/python" "$ROOT/comfyui_h3_gen.py" \
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
# Size alone is a bad proxy: a legitimate 512x288 smoke clip is ~800 kB and used to be
# rejected as "an error body". Ask ffprobe when it exists, and keep the size check only
# as a fallback for hosts without it.
FRAMES=""; HAS_AUDIO=""
if command -v ffprobe >/dev/null 2>&1; then
    FRAMES="$(ffprobe -v error -select_streams v:0 -count_frames \
        -show_entries stream=nb_read_frames -of csv=p=0 "$NEWEST" 2>/dev/null | tr -d ',')"
    HAS_AUDIO="$(ffprobe -v error -select_streams a -show_entries stream=index \
        -of csv=p=0 "$NEWEST" 2>/dev/null | head -1)"
    [ -n "$FRAMES" ] && [ "$FRAMES" -gt 0 ] \
        || die "no decodable video stream in $NEWEST (${SIZE} bytes) — ComfyUI reported success"
    [ "$FRAMES" -eq "$LENGTH" ] || warn "decoded $FRAMES frames, expected $LENGTH"
    [ -n "$HAS_AUDIO" ] || warn "no audio stream in $NEWEST (H3 should always emit AAC)"
    say "wrote $NEWEST ($((SIZE/1024/1024)) MiB, $FRAMES frames, audio=$([ -n "$HAS_AUDIO" ] && echo yes || echo NO))"
else
    [ "$SIZE" -gt 1000000 ] || die "output is only ${SIZE} bytes — likely an error body, not a video: $NEWEST"
    say "wrote $NEWEST ($((SIZE/1024/1024)) MiB)"
fi

if [ "$KEEP_SERVER" -eq 0 ]; then
    fuser -k "$PORT/tcp" >/dev/null 2>&1 || true
    say "server stopped"
fi
