#!/usr/bin/env bash
# Stress-test XPU all_reduce with dist_allreduce.py, run from .venv.
#
# Two things about this host that this wrapper exists to handle — both make the
# script fail before it reaches any collective:
#
#   1. The venv's libsycl.so.9 (intel-sycl-rt 2026.1.0) needs the libur_loader
#      that ships beside it in .venv/lib. A newer oneAPI (2026.2 here) puts its
#      own first on the default search path and the import dies with
#        undefined symbol: urKernelSetArgLocal, version LIBUR_LOADER_0.12
#      so .venv/lib has to lead LD_LIBRARY_PATH.
#   2. oneCCL's OFI transport needs libfabric *with provider plugins*. The venv
#      ships only the core library, and its build differs from oneAPI's, so
#      oneCCL fails with
#        fi_getinfo error: ret -61, providers 0
#      Prepending oneAPI's libfabric directory (core + plugins, same build)
#      fixes it.
#
# XPU 0-15 are deliberately off limits: --gpus defaults to 16 and up, and any id
# below 16 is refused.
#
# Usage:
#   ./run_dist_allreduce.sh                          # 4 ranks on XPU 16-19
#   ./run_dist_allreduce.sh --ranks 8                # 8 ranks on XPU 16-23
#   ./run_dist_allreduce.sh --gpus 24,25,26,27 --ranks 4
#   ./run_dist_allreduce.sh -- --steps 500 --calls-per-step 28
#   ./run_dist_allreduce.sh --timeout 600 --log /tmp/ar.log
#
# Defaults are conservative on purpose: 8-rank sustained all_reduce is the shape
# that once dropped a PCIe root port to Gen1 permanently (vllm_tp8_pcie_downgrade.md),
# and only a power cycle recovered it. Raise --ranks deliberately.
#
# Everything after `--` is forwarded to dist_allreduce.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${STRESS_VENV:-$ROOT/.venv}"
SCRIPT="$ROOT/dist_allreduce.py"
# oneAPI's libfabric (core + provider plugins) — must precede $VENV/lib.
FI_LIB_DIR="${FI_LIB_DIR:-/opt/intel/oneapi/libfabric/2.6/lib}"

MIN_GPU=16
RANKS=4
GPU_IDS=""
PORT="${MASTER_PORT:-29500}"
TIMEOUT="${TIMEOUT:-3600}"
LOG=""
PY_ARGS=()

say()  { printf '\033[1;36m[allreduce]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[allreduce] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m[allreduce] %s\033[0m\n' "$*" >&2; exit 1; }

# dmesg is dmesg_restrict'ed on this host, so try a cached sudo before giving up.
# Never prompt and never carry a password in the script.
dmesg_read() {
    if dmesg -T >/dev/null 2>&1; then
        dmesg -T
    elif sudo -n dmesg -T >/dev/null 2>&1; then
        sudo -n dmesg -T
    else
        return 1
    fi
}

usage() { sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --ranks)   RANKS="$2"; shift 2 ;;
        --gpus)    GPU_IDS="$2"; shift 2 ;;
        --port)    PORT="$2"; shift 2 ;;
        --timeout) TIMEOUT="$2"; shift 2 ;;
        --log)     LOG="$2"; shift 2 ;;
        --venv)    VENV="$2"; shift 2 ;;
        --)        shift; PY_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *)         die "unknown option: $1 (use -- to pass args to the python script)" ;;
    esac
done

[ -x "$VENV/bin/torchrun" ] || die "$VENV has no torchrun — is STRESS_VENV right?"
[ -f "$SCRIPT" ] || die "$SCRIPT missing"
[ -d "$FI_LIB_DIR" ] || die "oneAPI libfabric not found at $FI_LIB_DIR (set FI_LIB_DIR=...)"
[ "$RANKS" -ge 1 ] 2>/dev/null || die "--ranks must be a positive integer"

# --gpus accepts "16,17,18,19" or a range "16-19".
if [ -z "$GPU_IDS" ]; then
    GPU_IDS="$(seq -s, "$MIN_GPU" $((MIN_GPU + RANKS - 1)))"
elif [[ "$GPU_IDS" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    a="${BASH_REMATCH[1]}"; b="${BASH_REMATCH[2]}"
    [ "$a" -le "$b" ] || die "--gpus range must be ascending (got '$GPU_IDS')"
    GPU_IDS="$(seq -s, "$a" "$b")"
fi
case "$GPU_IDS" in
    *[!0-9,]*|,*|*,|*,,*) die "--gpus must be a comma list or a range like 16-19 (got '$GPU_IDS')" ;;
esac
IFS=',' read -r -a IDS <<<"$GPU_IDS"
[ "${#IDS[@]}" -eq "$RANKS" ] \
    || die "--gpus lists ${#IDS[@]} device(s) but --ranks is $RANKS; they must match"
for id in "${IDS[@]}"; do
    [ "$id" -ge "$MIN_GPU" ] \
        || die "XPU $id is not allowed: this box reserves 0-15, use $MIN_GPU and up"
done

# --- environment ------------------------------------------------------------
# Set every variable this run depends on explicitly instead of inheriting them.
# Two traps are worth naming:
#   * LD_LIBRARY_PATH is *replaced*, not extended: the system oneAPI compiler
#     directory must not win over the venv's libur_loader.
#   * FI_PROVIDER_PATH must be exported. It points at oneAPI's libfabric plugin
#     directory; without it oneCCL's OFI transport finds zero providers
#     ("fi_getinfo error: ret -61, providers 0") even though libfabric loads.
#     An earlier version of this script unset it and broke exactly that way.
unset ZE_AFFINITY_MASK CCL_ATL_TRANSPORT 2>/dev/null || true
export LD_LIBRARY_PATH="$FI_LIB_DIR:$VENV/lib"
export FI_PROVIDER_PATH="$FI_LIB_DIR/libfabric"
export CCL_ROOT="$VENV"
# Forced to 0, not overridable: that is oneCCL's own default, so the run takes
# the plain data path instead of the SYCL copy-engine path. dist_allreduce.py's
# header suggests 1; this wrapper overrides that deliberately.
export CCL_SYCL_COPY_ENGINE=0
export ZE_AFFINITY_MASK="$GPU_IDS"
export PYTHONUNBUFFERED=1

# Physical device count has to be measured before ZE_AFFINITY_MASK hides the
# rest: with the mask set, device_count() reports the masked subset (4), and
# comparing a physical id like 16 against that would always fail.
avail="$(env -u ZE_AFFINITY_MASK "$VENV/bin/python" -c \
    'import torch; print(torch.xpu.device_count())' 2>/dev/null | tail -1)"
for id in "${IDS[@]}"; do
    [ "$id" -lt "${avail:-0}" ] || die "XPU $id does not exist (physical device_count=$avail)"
done

# --- preflight --------------------------------------------------------------
say "venv    : $VENV"
say "ranks   : $RANKS on XPU $GPU_IDS"
say "timeout : ${TIMEOUT}s"
"$VENV/bin/python" - <<'PY' || die "preflight failed (see above)"
import sys
try:
    import torch, torch.distributed as dist
except Exception as exc:
    sys.exit(f"  torch import failed: {exc}")
print(f"  torch   : {torch.__version__}")
if not torch.xpu.is_available():
    sys.exit("  torch.xpu.is_available() is False")
print(f"  xpu     : {torch.xpu.device_count()} visible under the mask")
if not dist.is_xccl_available():
    sys.exit("  xccl backend unavailable — oneCCL is not loadable")
print("  xccl    : ok")
PY

for id in "${IDS[@]}"; do
    node="/dev/dri/renderD$((128 + id))"
    holders="$(fuser "$node" 2>/dev/null | tr -d ' ' || true)"
    if [ -n "$holders" ]; then
        warn "$node is held by pid(s): $holders"
        for p in $holders; do
            printf '    %s %s\n' "$p" "$(ps -o etime=,cmd= -p "$p" 2>/dev/null | cut -c1-100)" >&2
        done
        warn "another workload will distort the numbers; continuing anyway"
    fi
done

# --- run --------------------------------------------------------------------
if [ -z "$LOG" ]; then
    mkdir -p "$ROOT/h3_output"
    LOG="$ROOT/h3_output/dist_allreduce_$(date +%Y%m%d_%H%M%S).log"
fi
mkdir -p "$(dirname "$LOG")"
say "log     : $LOG"
say "starting: torchrun --nproc-per-node=$RANKS --master-port=$PORT"

# Remember where the kernel log ended, so the summary below reports only what
# this run produced rather than every fault since boot.
DMESG_MARK="$(dmesg_read 2>/dev/null | wc -l || true)"
case "$DMESG_MARK" in ''|*[!0-9]*) DMESG_MARK="" ;; esac

set +e
timeout "$TIMEOUT" "$VENV/bin/torchrun" \
    --nproc-per-node="$RANKS" --master-port="$PORT" \
    "$SCRIPT" "${PY_ARGS[@]+"${PY_ARGS[@]}"}" 2>&1 | tee "$LOG"
rc="${PIPESTATUS[0]}"
set -e

# --- aftermath --------------------------------------------------------------
# A hang is the failure mode this test exists to reproduce, so it gets its own
# message rather than looking like a generic non-zero exit.
if [ "$rc" -eq 124 ]; then
    warn "TIMEOUT after ${TIMEOUT}s — treating it as a hang (that is what this test hunts)"
    warn "the collective never completed; the log ends at: $(tail -1 "$LOG" 2>/dev/null | cut -c1-100)"
elif [ "$rc" -ne 0 ]; then
    warn "torchrun exited $rc; see $LOG"
fi

say "kernel messages produced by this run:"
if [ -n "${DMESG_MARK:-}" ] && hits="$(dmesg_read 2>/dev/null | tail -n "+$((DMESG_MARK + 1))" \
        | grep -iE "DEVICE_LOST|exec queue reset|xe .*(wedged|reset)|guc.*(timeout|fail)" || true)"; then
    if [ -n "$hits" ]; then
        printf '%s\n' "$hits" | tail -10 | sed 's/^/  /'
        warn "driver events appeared during this run — check the cards before reusing them"
    else
        say "  none"
    fi
else
    say "  dmesg is not readable without privileges; check manually with:"
    say "    sudo dmesg -T | grep -iE 'DEVICE_LOST|exec queue reset|xe .*wedged'"
fi

[ "$rc" -eq 0 ] || exit "$rc"
say "done"
