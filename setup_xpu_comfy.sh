#!/usr/bin/env bash
# Install Intel's XPU kernel stack into a COPY of our ComfyUI venv, so that
# ComfyUI 0.42.0 gets the same native int8 GEMM + CUTE attention + sparse
# attention that `intel/llm-scaler-omni` ships, without replacing our tree.
#
# Why this works (verified 2026-10-29):
#   * omni_xpu_kernel only requires torch==2.14.0 (+ onednn metadata), and our
#     .venv is exactly 2.14.0+xpu, so the ABI matches.
#   * The b2 image's provider only accepts comfy-kitchen 0.2.37, but our venv has
#     0.2.39. Building the provider from the 0.2.39-era fork source makes the
#     version check pass by construction instead of patching it.
#   * ComfyUI's own supports_int8_compute() hardcodes False for Intel XPU. The
#     switch that actually enables int8 is OmniXPU's quantized_matmul adapter,
#     which wraps comfy.ops.get_disabled_quant_formats and drops int8_tensorwise
#     from the disabled set. Our 0.42 signature (device=None) satisfies it.
#
# Result on one Arc Pro B60, 1376x768 / 124 frames, 8-step turbo LoRA, seed 1001
# (measured 2026-10-29, same graph and inputs as the image baseline):
#
#   .venv (stock)      dense 70.1 s/step, 652 s E2E,   frame 109.4/69.6
#   .venv-xpu          dense 47.2 s/step, 470.5 s E2E, frame 108.8/69.7
#   .venv-xpu + SOL    dense 39.3 s/step, 401.1 s E2E, frame 108.4/69.9
#   image (reference)  dense 46.4 s/step, 470.2 s E2E, frame 109.3/69.7
#   image + SOL        37.0 s/step,       370.2 s E2E, frame 108.4/70.1
#
# So the port reproduces the image within noise on the dense path (47.2 vs 46.4)
# and its frame statistics (108.8/69.7 vs 109.3/69.7); SOL lands ~6% behind the
# image (39.3 vs 37.0) but with the same token count (41,042) and sink rows
# ((0,45)/(38,45)).
#
# Usage:
#   ./setup_xpu_comfy.sh              # install (idempotent)
#   ./setup_xpu_comfy.sh --check      # verify only, change nothing
#   ./setup_xpu_comfy.sh --force      # rebuild .venv-xpu from scratch
#   ./setup_xpu_comfy.sh --bench      # also run one H3 and compare per-step
#
# The live .venv is never modified.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- pins (change deliberately, they are coupled) --------------------------
XPU_IMAGE="${XPU_IMAGE:-intel/llm-scaler-omni:0.2.0-b2}"
KITCHEN_REPO="${KITCHEN_REPO:-https://github.com/xiangyuT/comfy-kitchen-xpu.git}"
# llm-scaler main pins this revision together with comfy-kitchen==0.2.39, which
# is the version our .venv already has. Keep the two in lockstep.
KITCHEN_REV="${KITCHEN_REV:-3c9c495f3c648166ec4b0c69b676904f09b53267}"
XPU_TARGET="${XPU_TARGET:-bmg}"
TORCH_EXPECT="${TORCH_EXPECT:-2.14.0+xpu}"

VENV="${XPU_VENV:-$ROOT/.venv-xpu}"
SRC_VENV="${XPU_SRC_VENV:-$ROOT/.venv}"
VENDOR="$ROOT/vendor"
KITCHEN_DIR="$VENDOR/comfy-kitchen-xpu"
NODE_DIR="$VENDOR/ComfyUI-OmniXPU"
COMFY="$VENDOR/ComfyUI"
PY="$VENV/bin/python"
IMAGE_SITE="/opt/venv/lib/python3.12/site-packages"

MODE=install
FORCE=0
BENCH=0

say()  { printf '\033[1;36m[xpu-setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[xpu-setup] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[xpu-setup] %s\033[0m\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --check) MODE=check; shift ;;
        --force) FORCE=1; shift ;;
        --bench) BENCH=1; shift ;;
        -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

# --- preflight -------------------------------------------------------------
preflight() {
    command -v uv   >/dev/null || die "uv is required (the venv has no pip; we install with 'uv pip')"
    command -v git  >/dev/null || die "git is required"
    command -v docker >/dev/null || die "docker is required (the kernel wheel is extracted from the image)"
    [ -d "$SRC_VENV" ] || die "$SRC_VENV missing — run ./setup.sh first"
    [ -d "$COMFY" ]    || die "$COMFY missing — run ./setup.sh first"

    local torch_ver
    torch_ver="$("$SRC_VENV/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null | tail -1)"
    [ "$torch_ver" = "$TORCH_EXPECT" ] || die "live .venv torch is '$torch_ver', expected '$TORCH_EXPECT'.
      omni_xpu_kernel is AOT-built for one torch minor; do not mix."

    if ! docker image inspect "$XPU_IMAGE" >/dev/null 2>&1; then
        die "image $XPU_IMAGE not present; run: docker pull $XPU_IMAGE"
    fi
}

# The provider's provider.json lists compatible comfy-kitchen versions. Building
# it from the fork at KITCHEN_REV records that fork's own version, so the two
# must agree or the provider is installed but silently never activates.
check_kitchen_versions() {
    local live fork_ver
    live="$("$SRC_VENV/bin/python" -c 'import importlib.metadata as m; print(m.version("comfy-kitchen"))' 2>/dev/null | tail -1)"
    fork_ver="$(sed -n 's/^version *= *"\(.*\)"/\1/p' "$KITCHEN_DIR/pyproject.toml" | head -1)"
    say "comfy-kitchen: live=$live fork=$fork_ver"
    if [ -n "$fork_ver" ] && [ "$live" != "$fork_ver" ]; then
        warn "live comfy-kitchen ($live) != fork ($fork_ver)."
        warn "The provider will record $fork_ver and reject the live install."
        warn "Either pin KITCHEN_REV to the matching release or accept that adapters stay off."
    fi
}

# --- steps -----------------------------------------------------------------
copy_venv() {
    # Callers that already built their own venv set XPU_NO_COPY=1 and install in place.
    if [ "${XPU_NO_COPY:-0}" -eq 1 ]; then
        [ -x "$VENV/bin/python" ] || die "XPU_NO_COPY=1 but $VENV has no python"
        say "in-place install into $VENV (no copy)"
        return
    fi
    if [ -d "$VENV" ] && [ "$FORCE" -eq 0 ]; then
        say "venv copy exists: $VENV ($(du -sh "$VENV" | cut -f1)) — skipping"
        return
    fi
    [ "$FORCE" -eq 1 ] && { warn "--force: removing $VENV"; rm -rf "$VENV"; }
    say "copying .venv -> .venv-xpu (11 GiB, the live venv is untouched)"
    cp -a "$SRC_VENV" "$VENV"
}

extract_kernel() {
    local sp="$VENV/lib/python3.12/site-packages"
    if [ -d "$sp/omni_xpu_kernel" ] && [ "$FORCE" -eq 0 ]; then
        say "omni_xpu_kernel already installed — skipping extraction"
        return
    fi
    say "extracting omni_xpu_kernel + onednn metadata from $XPU_IMAGE"
    docker rm -f xpu-setup-probe >/dev/null 2>&1 || true
    docker run -d --name xpu-setup-probe "$XPU_IMAGE" sleep 300 >/dev/null
    # shellcheck disable=SC2064
    trap "docker rm -f xpu-setup-probe >/dev/null 2>&1 || true" RETURN

    local names
    names="$(docker exec xpu-setup-probe bash -lc \
        "ls $IMAGE_SITE | grep -iE '^omni_xpu_kernel|^onednn.*dist-info'" 2>/dev/null)"
    [ -n "$names" ] || die "could not list XPU packages inside $XPU_IMAGE"
    local n
    for n in $names; do
        docker cp "xpu-setup-probe:$IMAGE_SITE/$n" "$sp/" >/dev/null
        say "  installed $n"
    done
    docker rm -f xpu-setup-probe >/dev/null 2>&1 || true
}

build_kitchen() {
    if [ -d "$KITCHEN_DIR/.git" ] && [ "$FORCE" -eq 0 ]; then
        say "kitchen fork present: $KITCHEN_DIR — skipping clone"
    else
        say "cloning $KITCHEN_REPO"
        mkdir -p "$VENDOR"
        rm -rf "$KITCHEN_DIR"
        git clone -q "$KITCHEN_REPO" "$KITCHEN_DIR"
    fi
    git -C "$KITCHEN_DIR" checkout -q "$KITCHEN_REV" \
        || die "cannot checkout $KITCHEN_REV in $KITCHEN_DIR"

    if [ -f "$KITCHEN_DIR/dist/comfy_kitchen-"*.whl ] 2>/dev/null; then
        say "comfy_kitchen wheel already built — skipping"
        return
    fi
    say "building the pure-python comfy_kitchen wheel (py3-none-any, no compile)"
    ( cd "$KITCHEN_DIR" && uv build --wheel --out-dir dist >/dev/null )
}

build_provider() {
    local wheel
    wheel="$(ls "$KITCHEN_DIR"/dist/provider/comfy_kitchen_xpu_runtime-*.whl 2>/dev/null | head -1 || true)"
    if [ -z "$wheel" ] || [ "$FORCE" -eq 1 ]; then
        local src
        src="$(ls "$KITCHEN_DIR"/dist/comfy_kitchen-*.whl | head -1)"
        say "building the XPU runtime provider wheel (torch=$TORCH_EXPECT target=$XPU_TARGET)"
        "$PY" "$KITCHEN_DIR/packaging/xpu_runtime_provider/build_wheel.py" \
            --source-wheel "$src" \
            --output-dir "$KITCHEN_DIR/dist/provider" \
            --source-revision "$(git -C "$KITCHEN_DIR" rev-parse HEAD)" \
            --torch-version "$TORCH_EXPECT" \
            --xpu-target "$XPU_TARGET" >/dev/null
        wheel="$(ls "$KITCHEN_DIR"/dist/provider/comfy_kitchen_xpu_runtime-*.whl | head -1)"
    fi
    say "installing $(basename "$wheel")"
    uv pip install --python "$PY" --no-deps "$wheel" 2>&1 | tail -1
}

install_node() {
    if [ -d "$NODE_DIR" ] && [ "$FORCE" -eq 0 ]; then
        say "OmniXPU node source present: $NODE_DIR — skipping"
    else
        say "extracting the ComfyUI-OmniXPU node from $XPU_IMAGE"
        docker rm -f xpu-setup-probe >/dev/null 2>&1 || true
        docker create --name xpu-setup-probe "$XPU_IMAGE" >/dev/null
        rm -rf "$NODE_DIR"
        docker cp "xpu-setup-probe:/llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU" "$VENDOR/" >/dev/null
        docker rm -f xpu-setup-probe >/dev/null 2>&1 || true
    fi
    # vendor/ComfyUI is gitignored and rebuilt by setup.sh, so only link the node
    # in; never patch files inside the ComfyUI checkout.
    mkdir -p "$COMFY/custom_nodes"
    ln -sfn "$NODE_DIR" "$COMFY/custom_nodes/ComfyUI-OmniXPU"
    say "linked $COMFY/custom_nodes/ComfyUI-OmniXPU -> $NODE_DIR"
}

# --- verification ----------------------------------------------------------
verify() {
    say "verifying (this is the part that matters: packages installed but adapters
      not applied looks exactly like 'no speedup')"
    PYTHONPATH="$COMFY" "$PY" - "$NODE_DIR" <<'PY'
import importlib.util, sys
from pathlib import Path

node = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("omnixpu_prestartup", node / "prestartup_script.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

result = mod.apply_runtime_providers()
print("providers      :", result.get("status"), result.get("errors") or "")

import torch, comfy_kitchen as ck, comfy.ops as ops
print("torch          :", torch.__version__)
backends = ck.list_backends()
print("xpu backend    :", (backends.get("xpu") or {}).get("available"))
print("sol_attn       :", ck.sol_attn_is_available(torch.device("xpu:0")))

sys.path.insert(0, str(node))
ok, msg = __import__("adapters.quantized_matmul", fromlist=["apply"]).apply()
disabled = ops.get_disabled_quant_formats(torch.device("xpu:0"))
print("quant adapter  :", ok, msg)
print("int8 enabled   :", "int8_tensorwise" not in disabled)
print("still disabled :", sorted(disabled))

fail = []
if result.get("status") != "active":
    fail.append("provider not active")
if not (backends.get("xpu") or {}).get("available"):
    fail.append("xpu backend missing")
if "int8_tensorwise" in disabled:
    fail.append("int8 still disabled")
if fail:
    print("\nRESULT: FAIL —", "; ".join(fail))
    sys.exit(1)
print("\nRESULT: PASS")
PY
}

bench() {
    say "benchmarking one dense H3 run (8-step turbo, 1376x768, 124 frames)"
    local gpu="${BENCH_GPU:-16}" port="${BENCH_PORT:-8299}"
    local out="$ROOT/h3_output/xpu_bench"
    rm -rf "$out"; mkdir -p "$out"
    ( ZE_AFFINITY_MASK="$gpu" "$ROOT/run_xpu_comfy.sh" --gpu "$gpu" --port "$port" --out "$out" \
        > "$ROOT/h3_output/xpu_bench_server.log" 2>&1 & )
    local i
    for i in $(seq 1 60); do
        curl -sf -m 2 "http://127.0.0.1:$port/system_stats" >/dev/null && break
        sleep 5
    done
    curl -sf -m 2 "http://127.0.0.1:$port/system_stats" >/dev/null \
        || die "server did not come up; see h3_output/xpu_bench_server.log"

    PYTHONPATH="$COMFY" "$ROOT/.venv/bin/python" - "$ROOT" "$port" <<'PY'
import json, sys, time, urllib.request
root, port = sys.argv[1], sys.argv[2]
base = f"http://127.0.0.1:{port}"
tpl = json.load(open(f"{root}/h3_input/prompts/jzl_shot1_api.json"))["prompt"]
tpl["3"]["inputs"]["vae_name"] = "minimax_h3_video_vae_fp16.safetensors"
ms = json.loads(tpl["9"]["inputs"]["manager_settings"])
ms["models"]["common"]["video_vae"] = "minimax_h3_video_vae_fp16.safetensors"
tpl["9"]["inputs"]["manager_settings"] = json.dumps(ms, ensure_ascii=False)
req = urllib.request.Request(f"{base}/prompt", data=json.dumps({"prompt": tpl}).encode(),
                             headers={"Content-Type": "application/json"})
pid = json.loads(urllib.request.urlopen(req, timeout=30).read())["prompt_id"]
t0 = time.time()
while time.time() - t0 < 1800:
    time.sleep(10)
    h = json.loads(urllib.request.urlopen(f"{base}/history/{pid}", timeout=30).read())
    if pid in h:
        print(f"E2E            : {time.time() - t0:.1f} s  status={h[pid].get('status',{}).get('status_str')}")
        break
PY
    local per
    per="$(tr '\r' '\n' < "$ROOT/h3_output/xpu_bench_server.log" | grep -oE '8/8 \[[0-9:]+<[0-9:]+, +[0-9.]+s/it\]' | tail -1)"
    say "per-step: ${per:-<not found>}"
    say "expected: ~47 s/step on .venv-xpu, 70.1 s/step on the stock .venv"
    fuser -k "$port/tcp" >/dev/null 2>&1 || true
}

# --- main ------------------------------------------------------------------
if [ "$MODE" = check ]; then
    [ -d "$VENV" ] || die "$VENV missing — run ./setup_xpu_comfy.sh first"
    verify
    exit $?
fi

preflight
copy_venv
extract_kernel
build_kitchen
check_kitchen_versions
build_provider
install_node
verify
say "done. start ComfyUI with: ./run_xpu_comfy.sh --gpu 16 --port 8288"
[ "$BENCH" -eq 1 ] && bench
exit 0
