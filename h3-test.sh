#!/usr/bin/env bash
# MiniMax-H3 (神笔马良) single-shot test on one Intel Arc XPU.
#
# Runs this repo's ComfyUI (vendor/ComfyUI) with the H3 nodes and workflow
# vendored under ./h3_input, weights from ~/work/models, output to ./h3_output.
# Verifies the mp4, then stops the server again (--keep-server leaves it up).
#
# Usage:
#   ./h3-test.sh                      # GPU 0, template prompt, manifest defaults
#   ./h3-test.sh 4                    # bind to GPU 4
#   ./h3-test.sh 0 "a lighthouse in a storm"   # custom prompt, single segment
#   ./h3-test.sh 0 "..." --keep-server
#
# Env overrides: PORT, TIMEOUT (seconds), H3_INPUT, H3_MODELS, STEPS, DURATION, SEED.
#
# Reference numbers: 8 steps at ~70 s/step on one Arc Pro B60 -> ~9.5 min
# sampling, ~11 min wall for 124 frames at 1376x768. Anything under 5 s/step
# means the card is shared.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$ROOT/h3_output"
IN_DIR="${H3_INPUT:-$ROOT/h3_input}"
TEMPLATE="$IN_DIR/prompts/jzl_shot1_api.json"
COMFY="$ROOT/vendor/ComfyUI"
PY="$ROOT/.venv/bin/python"
H3_MODELS="${H3_MODELS:-$HOME/work/models/ComfyUI-MiniMax-H3}"
PORT="${PORT:-8190}"
TIMEOUT="${TIMEOUT:-2400}"
STEPS="${STEPS:-8}"
DURATION="${DURATION:-5}"
SEED="${SEED:-1001}"
KEEP_SERVER=0

GPU=0
PROMPT=""
H3_VIDEO_VAE=""
for arg in "$@"; do
    case "$arg" in
        --keep-server) KEEP_SERVER=1 ;;
        -*) echo "unknown flag: $arg" >&2; exit 2 ;;
        *) if [ -z "$PROMPT" ]; then GPU="$arg"; else PROMPT="$arg"; fi ;;
    esac
done

say() { printf '\033[1;36m[h3]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[h3] %s\033[0m\n' "$*" >&2; exit 1; }

# --- preflight -------------------------------------------------------------
say "preflight"
[ -x "$PY" ]       || die "venv interpreter missing: $PY (run setup.sh)"
[ -f "$TEMPLATE" ] || die "workflow template missing: $TEMPLATE"
[ -d "$COMFY" ]    || die "ComfyUI checkout missing: $COMFY"
[ -d "$H3_MODELS" ] || die "H3 weights missing: $H3_MODELS (override with H3_MODELS=)"
[ -d "$IN_DIR/assets" ] || die "assets missing: $IN_DIR/assets"
[ -d "$IN_DIR/custom_nodes/ComfyUI-JZL-MiniMax-H3" ] || die "JZL node missing under $IN_DIR/custom_nodes"

# The template ships the bf16 video VAE name from the delivery package, but
# ~/work/models carries the fp16 weights. Pick whichever exists.
for cand in minimax_h3_video_vae_bf16.safetensors minimax_h3_video_vae_fp16.safetensors; do
    if [ -f "$H3_MODELS/vae/$cand" ]; then H3_VIDEO_VAE="$cand"; break; fi
done
[ -n "$H3_VIDEO_VAE" ] || die "no MiniMax-H3 video VAE under $H3_MODELS/vae/"

"$PY" - <<'PYEOF' || die "venv cannot see an XPU"
import torch, sys
sys.exit(0 if torch.xpu.is_available() else 1)
PYEOF
TORCH_V="$("$PY" -c 'import torch;print(torch.__version__)')"
DEV_N="$("$PY" -c 'import torch;print(torch.xpu.device_count())')"
say "torch $TORCH_V, $DEV_N XPU device(s)"

if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    die "port $PORT already in use (override with PORT=)"
fi

mkdir -p "$OUT_DIR"

# --- start one ComfyUI pinned to the chosen card --------------------------
mkdir -p "$COMFY/custom_nodes"
for node in ComfyUI-JZL-MiniMax-H3 ComfyUI-MiniMaxH3_LatentUpscaler; do
    if [ -d "$IN_DIR/custom_nodes/$node" ] && [ ! -e "$COMFY/custom_nodes/$node" ]; then
        say "linking custom node $node ($(awk -v n="$node" '$1==n{print $2}' "$IN_DIR/custom_nodes/COMMITS" 2>/dev/null || echo '?'))"
        ln -s "$IN_DIR/custom_nodes/$node" "$COMFY/custom_nodes/$node"
    fi
done

say "starting ComfyUI on GPU $GPU, port $PORT (video VAE: $H3_VIDEO_VAE)"
export ZE_AFFINITY_MASK="$GPU"
# point ComfyUI at the shared weight tree without touching its own folders
cat > "$COMFY/extra_model_paths.yaml" <<YEOF
h3:
    base_path: $H3_MODELS/
    diffusion_models: diffusion_models/
    text_encoders: text_encoders/
    clip: text_encoders/
    vae: vae/
    loras: loras/
YEOF
# the JZL node persists its settings here; seed from h3_input on first run
mkdir -p "$COMFY/user"
[ -f "$COMFY/user/jzl_manager.json" ] || cp -a "$IN_DIR/jzl_manager.json" "$COMFY/user/jzl_manager.json"
cd "$COMFY"
"$PY" main.py \
    --listen 127.0.0.1 --port "$PORT" --disable-auto-launch \
    --input-directory "$IN_DIR/assets" \
    --output-directory "$OUT_DIR" \
    --async-offload 2 --reserve-vram 2 \
    > "$ROOT/h3_output/server.log" 2>&1 &
SERVER_PID=$!

cleanup() {
    if [ "$KEEP_SERVER" -eq 0 ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        say "stopping server (pid $SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        sleep 5
        kill -9 "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT



# --- wait for the node to register ----------------------------------------
say "waiting for JZL_MiniMaxAssetManager"
for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:$PORT/object_info" 2>/dev/null \
        | grep -q '"JZL_MiniMaxAssetManager"'; then
        break
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || {
        tail -20 "$ROOT/h3_output/server.log" >&2
        die "server exited during startup"
    }
    sleep 2
done
curl -sf "http://127.0.0.1:$PORT/object_info" 2>/dev/null | grep -q '"JZL_MiniMaxAssetManager"' \
    || die "JZL_MiniMaxAssetManager never registered (check $ROOT/h3_output/server.log)"
say "node registered"

# --- submit ----------------------------------------------------------------
if [ -n "$PROMPT" ]; then
    PROMPT_FILE="$(mktemp)"
    STORIES="${PROMPT}"
    cat > "$PROMPT_FILE" <<PYEOF
[SHOT_START]
**时长**: ${DURATION}
**场景**: single segment
**角色**: 无人（空景）
**道具**: 无
===SCENE_INSTRUCTION===
{"slots":[]}
===H3_PROMPT===
subject_definitions:
${STORIES}

retention_analysis:
Keep the described scene, framing and lighting consistent across the shot.

detailed_description:
${STORIES}
Photorealistic live-action film look, natural daylight.

overall_soundscape:
Quiet outdoor ambience appropriate to the described scene.

non_diegetic_music:
None.
[SHOT_END]
PYEOF
    PAYLOAD_FILE="$(mktemp)"
    "$PY" - "$TEMPLATE" "$PROMPT_FILE" "$PAYLOAD_FILE" "$SEED" "$STEPS" "$DURATION" "$H3_VIDEO_VAE" <<'PYEOF'
import json, sys
src, prompt_file, out, seed, steps, duration, video_vae = sys.argv[1:8]
d = json.load(open(src))
prompt = d.get("prompt", d)
n9 = prompt["9"]["inputs"]
n9["story_name"] = "h3_test"
n9["video_count"] = "生成1段"
n9["external_prompt"] = open(prompt_file).read()
for node in prompt.values():
    if node.get("class_type") == "VAELoader" and node["inputs"].get("vae_name", "").endswith(
        "minimax_h3_video_vae_bf16.safetensors"):
        node["inputs"]["vae_name"] = video_vae
ms = json.loads(n9["manager_settings"])
ms["gen_params"]["duration"] = int(duration)
ms["sample_decode"]["steps"] = int(steps)
ms["sample_decode"]["seed"] = int(seed)
ms["models"]["common"]["video_vae"] = video_vae
n9["manager_settings"] = json.dumps(ms, ensure_ascii=False)
json.dump({"prompt": prompt, "client_id": "h3-test"}, open(out, "w"), ensure_ascii=False)
PYEOF
else
    PAYLOAD_FILE="$(mktemp)"
    "$PY" - "$TEMPLATE" "$PAYLOAD_FILE" "$SEED" "$STEPS" "$DURATION" "$H3_VIDEO_VAE" <<'PYEOF'
import json, sys
src, out, seed, steps, duration, video_vae = sys.argv[1:7]
d = json.load(open(src))
prompt = d.get("prompt", d)
n9 = prompt["9"]["inputs"]
n9["story_name"] = "h3_test"
n9["video_count"] = "生成1段"
for node in prompt.values():
    if node.get("class_type") == "VAELoader" and node["inputs"].get("vae_name", "").endswith(
        "minimax_h3_video_vae_bf16.safetensors"):
        node["inputs"]["vae_name"] = video_vae
ms = json.loads(n9["manager_settings"])
ms["gen_params"]["duration"] = int(duration)
ms["sample_decode"]["steps"] = int(steps)
ms["sample_decode"]["seed"] = int(seed)
ms["models"]["common"]["video_vae"] = video_vae
n9["manager_settings"] = json.dumps(ms, ensure_ascii=False)
json.dump({"prompt": prompt, "client_id": "h3-test"}, open(out, "w"), ensure_ascii=False)
PYEOF
fi

say "submitting (steps=$STEPS seed=$SEED duration=${DURATION}s)"
RESP="$(curl -sf -X POST "http://127.0.0.1:$PORT/prompt" \
    -H 'Content-Type: application/json' \
    --data-binary "@$PAYLOAD_FILE")" || die "POST /prompt failed"
PID_ID="$(printf '%s' "$RESP" | "$PY" -c 'import json,sys;print(json.load(sys.stdin).get("prompt_id",""))')"
[ -n "$PID_ID" ] || die "no prompt_id in response: $RESP"
say "prompt_id $PID_ID"

# --- wait ------------------------------------------------------------------
START=$(date +%s)
STATE="running"
while [ $(( $(date +%s) - START )) -lt "$TIMEOUT" ]; do
    STATE="$(curl -sf "http://127.0.0.1:$PORT/history/$PID_ID" 2>/dev/null | "$PY" -c '
import json,sys
try: d = json.load(sys.stdin)
except Exception: print("running"); raise SystemExit
e = next(iter(d.values()), None)
print("running" if not e else ("error" if e["status"].get("status_str") == "error" else "done"))
' || echo running)"
    [ "$STATE" = "running" ] || break
    sleep 10
done
ELAPSED=$(( $(date +%s) - START ))

if [ "$STATE" != "done" ]; then
    say "state=$STATE after ${ELAPSED}s (timeout $TIMEOUT s)"
    grep -oE '[0-9]+/'"$STEPS"' \[[0-9:]+<[0-9:]+, *[0-9.]+s/it\]' "$ROOT/h3_output/server.log" | tail -1 >&2 || true
    [ "$STATE" = "error" ] && curl -s "http://127.0.0.1:$PORT/history/$PID_ID" | tail -c 600 >&2
    exit 1
fi
say "generation finished in ${ELAPSED}s"

# --- verify ----------------------------------------------------------------
NEW="$(find "$OUT_DIR" -name '*.mp4' -newer "$PAYLOAD_FILE" 2>/dev/null | sort)"
[ -n "$NEW" ] || die "success reported but no new mp4 in $OUT_DIR"
say "output:"
printf '%s\n' "$NEW" | sed 's/^/  /'

BEST="$(printf '%s\n' "$NEW" | head -1)"
"$PY" - "$BEST" <<'PYEOF'
import sys, av
name = sys.argv[1]
c = av.open(name)
v = c.streams.video[0]
dur = float(v.duration * v.time_base)
print(f"  video : {v.codec_context.name} {v.codec_context.width}x{v.codec_context.height} "
      f"{v.frames} frames {dur:.2f}s")
if c.streams.audio:
    a = c.streams.audio[0]
    print(f"  audio : {a.codec_context.name} {a.codec_context.sample_rate}Hz "
          f"ch={a.codec_context.channels}")
else:
    print("  audio : MISSING")
means = [f.to_ndarray(format="rgb24").mean() for f in c.decode(video=0)]
print(f"  frames: mean {min(means):.1f}..{max(means):.1f} across {len(means)} frames")
if max(means) < 5:
    raise SystemExit("  FAIL: video looks black")
PYEOF

grep -oE '[0-9]+/'"$STEPS"' \[[0-9:]+<[0-9:]+, *[0-9.]+s/it\]' "$ROOT/h3_output/server.log" | tail -1 | sed 's/^/  sampling: /' || true
say "done. artifacts in $OUT_DIR (server.log there too)"
[ "$KEEP_SERVER" -eq 1 ] && say "server left running on port $PORT (--keep-server)"
exit 0