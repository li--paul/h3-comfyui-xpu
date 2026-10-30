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
#   ./setup_xpu_comfy.sh                    # install (idempotent), kernel from image
#   ./setup_xpu_comfy.sh --kernel-from-source   # compile the kernel instead
#   ./setup_xpu_comfy.sh --kernel-from-image    # explicit default (needs docker)
#   ./setup_xpu_comfy.sh --check            # verify only, change nothing
#   ./setup_xpu_comfy.sh --force            # rebuild .venv-xpu from scratch
#   ./setup_xpu_comfy.sh --bench            # also run one H3 and compare per-step
#
# The live .venv is never modified.
#
# --- where omni_xpu_kernel comes from --------------------------------------
# Two paths, same artifact, pick with KERNEL_MODE / the flags above.
#
#   image   docker extract from intel/llm-scaler-omni:0.2.0-b2 (current default)
#   source  compile from the pinned llm-scaler revision (no docker at all)
#
# The source path is not hand-rolled: it drives Kitchen's own release harness,
# packaging/omni_xpu_kernel/build_uv_wheel_matrix.sh, which is what Intel uses
# to publish these wheels. That harness builds in a throwaway venv and then
# smoke-tests the result (cute.sdp vs torch SDPA, capability symbols), so a
# successful run already proves the wheel is usable.
#
# Pins for the source path, all cross-checked against the image labels:
#   OMNI_REV          a2460e5  llm-scaler source, last 0.2.0-b2 commit
#   SYCL_TLA_COMMIT   2fc09973  image label com.intel.omni.sycl-tla.revision
#   ONEDNN_VERSION    2026.0.0  image label com.intel.omni.onednn.package.version
#   ONEAPI_VERSION    2026.0    the oneAPI generation oneDNN 3.11.2 belongs to
#
# The OMNI_REV window is bounded on both sides, and both bounds are load-bearing:
#   lower  b8ab322 (2026-10-08)  first commit whose _version.py accepts torch 2.14;
#                                anything earlier dies with "Unsupported Torch
#                                version '2.14.0'; supported XPU minors: 2.10..2.13"
#   upper  09f2953 (2026-10-20)  bumps the kernel to 0.2.0-b3 and splits Sol-Attn
#                                into a separate wheel; the core wheel goes dense-only
# a2460e5 is 09f2953^, i.e. the newest commit inside that window, and its
# Dockerfile pins TORCH/ONEDNN/XPU_TARGET/SYCL_TLA exactly as the image labels do.
#
# Do NOT re-derive this pin from the image's creation date: that label
# (org.opencontainers.image.created=2026-09-07) comes from the *base* image
# intel/omix:0.4.0 and says nothing about when the omni layer was built.
#
# Three traps the source path must keep honouring:
#   * Never move OMNI_REV past 09f2953. The failure is silent: --sol still runs,
#     it just stops being sparse (39.3 -> 47.2 s/step). verify() asserts
#     cute.sol_attn for this.
#   * The image's oneDNN is a source build with the bf16-int4-dequantization
#     patch (sha256 0a7afff4...), while the source path links pip's stock
#     2026.0.0. Same version string, not the same binary. The H3 path does not
#     reach that patch today (47.2 vs the image's 46.4 s/step), but it is a real
#     difference and not something to "clean up" later.
#   * The image does not record llm-scaler.source-revision (only
#     source-dirty=unknown), so "byte-identical to the image" is not provable.
#     The acceptance test is the measured per-step time, not the version string.
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

# --- source-build pins (only used when KERNEL_MODE=source) -------------------
# image = docker extract (default), source = compile from the llm-scaler tree.
KERNEL_MODE="${KERNEL_MODE:-image}"
OMNI_REPO="${OMNI_REPO:-https://github.com/intel/llm-scaler.git}"
# Last llm-scaler revision whose kernel still declares 0.2.0-b2 (the version
# the image ships), accepts torch 2.14, and still bundles Sol-Attn in the core
# wheel. See the header for the window this has to stay inside.
OMNI_REV="${OMNI_REV:-a2460e528ad585fb4c76b4221279d2cee2abe770}"
SYCL_TLA_REPO="${SYCL_TLA_REPO:-https://github.com/intel/sycl-tla.git}"
SYCL_TLA_COMMIT="${SYCL_TLA_COMMIT:-2fc09973bfdf15755090fcb0e3b6ad236408a992}"
ONEDNN_VERSION="${ONEDNN_VERSION:-2026.0.0}"
# Pins icpx; setvars.sh would otherwise pick compiler/latest (2026.1 here).
ONEAPI_VERSION="${ONEAPI_VERSION:-2026.0}"
KERNEL_BUILD_JOBS="${KERNEL_BUILD_JOBS:-8}"

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
        --kernel-from-source) KERNEL_MODE=source; shift ;;
        --kernel-from-image)  KERNEL_MODE=image; shift ;;
        -h|--help) sed -n '2,72p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

case "$KERNEL_MODE" in
    image|source) ;;
    *) die "KERNEL_MODE must be image or source, got '$KERNEL_MODE'" ;;
esac

# --- preflight -------------------------------------------------------------
preflight() {
    command -v uv   >/dev/null || die "uv is required (the venv has no pip; we install with 'uv pip')"
    command -v git  >/dev/null || die "git is required"
    [ -d "$SRC_VENV" ] || die "$SRC_VENV missing — run ./setup.sh first"
    [ -d "$COMFY" ]    || die "$COMFY missing — run ./setup.sh first"

    local torch_ver
    torch_ver="$("$SRC_VENV/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null | tail -1)"
    [ "$torch_ver" = "$TORCH_EXPECT" ] || die "live .venv torch is '$torch_ver', expected '$TORCH_EXPECT'.
      omni_xpu_kernel is AOT-built for one torch minor; do not mix."

    if [ "$KERNEL_MODE" = image ]; then
        command -v docker >/dev/null \
            || die "docker is required for --kernel-from-image (the kernel wheel is extracted from the image).
      Use --kernel-from-source to build it instead; that path needs no docker."
        if ! docker image inspect "$XPU_IMAGE" >/dev/null 2>&1; then
            die "image $XPU_IMAGE not present; run: docker pull $XPU_IMAGE"
        fi
    else
        local icpx="/opt/intel/oneapi/compiler/$ONEAPI_VERSION/bin/icpx"
        [ -x "$icpx" ] || die "icpx not found at $icpx
      The CUTE FMHA sidecar needs the Intel oneAPI DPC++ compiler.
      Install oneAPI $ONEAPI_VERSION or point ONEAPI_VERSION at an existing tree."
        [ -d "$KITCHEN_DIR/packaging/omni_xpu_kernel" ] \
            || die "$KITCHEN_DIR/packaging/omni_xpu_kernel missing — run ./setup.sh or check KITCHEN_REV"
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

# --- source build (KERNEL_MODE=source) --------------------------------------
OMNI_SRC="$VENDOR/llm-scaler/omni/omni_xpu_kernel"
SYCL_TLA_DIR="$VENDOR/sycl-tla"
KERNEL_WHEELHOUSE="${KERNEL_WHEELHOUSE:-$VENDOR/wheelhouse/omni_xpu_kernel}"
KERNEL_BUILD_LOG="$VENDOR/kernel-build.log"
KERNEL_STAMP="$VENDOR/.xpu-kernel-build.json"

# Same fetch shape the image Dockerfile uses: a blobless clone plus a depth-1
# fetch of the exact commit, so we never pull the full llm-scaler history.
fetch_pinned() {
    local repo="$1" dest="$2" commit="$3"
    if [ ! -d "$dest/.git" ]; then
        say "cloning $repo (blobless)"
        git clone --filter=blob:none --no-checkout "$repo" "$dest" >/dev/null 2>&1 \
            || die "cannot clone $repo"
    fi
    if [ "$(git -C "$dest" rev-parse HEAD 2>/dev/null)" = "$commit" ]; then
        say "  $(basename "$dest") already at ${commit:0:8}"
        return
    fi
    say "  fetching ${commit:0:8} into $dest"
    git -C "$dest" fetch --depth 1 origin "$commit" >/dev/null 2>&1 \
        || die "cannot fetch $commit from $repo"
    git -C "$dest" checkout -q --detach FETCH_HEAD \
        || die "cannot check out $commit in $dest"
}

# Drives Kitchen's own release harness rather than reimplementing the compile.
# That script builds in a throwaway uv venv, emits a cp312 wheel into
# KERNEL_WHEELHOUSE, and then smoke-tests it (cute.sdp vs torch SDPA, required
# capability symbols, version/target/oneDNN identity), so a zero exit already
# means the artifact is usable on this box.
build_kernel_from_source() {
    local wheel
    wheel="$(ls "$KERNEL_WHEELHOUSE"/omni_xpu_kernel-*cp312-*.whl 2>/dev/null | head -1 || true)"
    if [ -n "$wheel" ] && [ "$FORCE" -eq 0 ]; then
        say "kernel wheel already built: $(basename "$wheel") (use --force to rebuild)"
    else
        say "building omni_xpu_kernel from source (target=$XPU_TARGET, torch=$TORCH_EXPECT, oneAPI $ONEAPI_VERSION)"
        say "expect 30-90 minutes; the CUTE FMHA sidecar dominates"
        fetch_pinned "$OMNI_REPO"      "$VENDOR/llm-scaler" "$OMNI_REV"
        fetch_pinned "$SYCL_TLA_REPO"  "$SYCL_TLA_DIR"      "$SYCL_TLA_COMMIT"

        local harness="$KITCHEN_DIR/packaging/omni_xpu_kernel/build_uv_wheel_matrix.sh"
        [ -f "$harness" ] || die "$harness missing — check KITCHEN_REV / run build_kitchen first"

        # setvars.sh is not set -u safe, and PATH decides which icpx setup.py
        # finds, so pin the compiler directory explicitly after sourcing.
        if ! ( set +u
               source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
               export PATH="/opt/intel/oneapi/compiler/$ONEAPI_VERSION/bin:$PATH"
               export OMNI_XPU_KERNEL_SOURCE="$OMNI_SRC"
               export CUTLASS_SYCL_ROOT="$SYCL_TLA_DIR"
               export OMNI_XPU_DEVICE="$XPU_TARGET"
               export EXPECTED_XPU_TARGET="$XPU_TARGET"
               export EXPECTED_TORCH_MINOR="${TORCH_EXPECT%%.*}.$(echo "$TORCH_EXPECT" | cut -d. -f2)"
               export TORCH_SPEC="torch==$TORCH_EXPECT"
               export ONEDNN_SPEC="onednn==$ONEDNN_VERSION"
               export ONEDNN_DEVEL_SPEC="onednn-devel==$ONEDNN_VERSION"
               export EXPECTED_ONEDNN_VERSION="$ONEDNN_VERSION"
               export WHEELHOUSE="$KERNEL_WHEELHOUSE"
               export MATRIX_ROOT="$VENDOR/llm-scaler/.uv-wheel-matrix"
               export MAX_JOBS="$KERNEL_BUILD_JOBS"
               export CMAKE_BUILD_PARALLEL_LEVEL="$KERNEL_BUILD_JOBS"
               bash "$harness" 3.12 ) 2>&1 | tee "$KERNEL_BUILD_LOG"
        then
            die "kernel build failed — full log: $KERNEL_BUILD_LOG"
        fi

        wheel="$(ls "$KERNEL_WHEELHOUSE"/omni_xpu_kernel-*cp312-*.whl 2>/dev/null | head -1 || true)"
        [ -n "$wheel" ] || die "build finished but produced no cp312 wheel in $KERNEL_WHEELHOUSE"
    fi

    say "installing $(basename "$wheel") into $VENV"
    uv pip install --python "$PY" --no-deps --force-reinstall "$wheel" 2>&1 | tail -1

    cat > "$KERNEL_STAMP" <<EOF
{"wheel": "$(basename "$wheel")",
 "omni_repo": "$OMNI_REPO",
 "omni_rev": "$OMNI_REV",
 "sycl_tla_commit": "$SYCL_TLA_COMMIT",
 "xpu_target": "$XPU_TARGET",
 "torch": "$TORCH_EXPECT",
 "onednn": "$ONEDNN_VERSION",
 "oneapi": "$ONEAPI_VERSION"}
EOF
    say "recorded provenance in $KERNEL_STAMP"
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
    elif [ "$KERNEL_MODE" = source ]; then
        # The node ships inside the llm-scaler tree, so the source path needs no
        # docker for it either.
        local src="$VENDOR/llm-scaler/omni/ComfyUI-OmniXPU"
        [ -d "$src" ] || die "$src missing — fetch_pinned should have produced it"
        say "copying the OmniXPU node from $src"
        rm -rf "$NODE_DIR"
        cp -a "$src" "$NODE_DIR"
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

# The kernel wheel must match this venv's torch minor and carry the BMG AOT
# images, otherwise every adapter silently falls back to eager torch code.
try:
    import omni_xpu_kernel as omni
    from omni_xpu_kernel import cute
    want = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    got = ".".join(omni.__torch_version__.split(".")[:2])
    print("omni_xpu_kernel:", omni.__version__, "| target:", omni.__xpu_target__)
    kernel_ok = (omni.__xpu_target__ == "bmg" and got == want
                 and omni.is_available() and cute.is_available())
    # b3 moved Sol-Attn into a separate omni_xpu_sol_attn wheel. Without the
    # symbol, --sol still runs but is dense: 39.3 -> 47.2 s/step, no error.
    sol_ok = hasattr(cute, "sol_attn")
    print("cute sidecar   :", cute.is_available(), "| sol_attn op:", sol_ok)
except Exception as exc:
    print("omni_xpu_kernel: MISSING —", exc)
    kernel_ok = sol_ok = False

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
if not kernel_ok:
    fail.append("omni_xpu_kernel missing or built for another torch/target")
if not sol_ok:
    fail.append("cute.sol_attn absent (sparse attention would silently run dense)")
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
if [ "$KERNEL_MODE" = source ]; then
    build_kernel_from_source
else
    extract_kernel
fi
build_kitchen
check_kitchen_versions
build_provider
install_node
verify
say "done. start ComfyUI with: ./run_xpu_comfy.sh --gpu 16 --port 8288"
[ "$BENCH" -eq 1 ] && bench
exit 0
