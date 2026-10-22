#!/bin/bash
# vLLM TP 推理启动脚本 (Intel Arc Pro B60, XPU, torch 2.14)
#
# 关键环境要求:
#  1. PATH 不含 oneAPI bin (无 icpx) → triton 只用 venv 里的 intel-sycl-rt
#  2. 清空 triton cache → JIT 产物与当前 libsycl ABI 重新匹配
#     (intel-sycl-rt 2026.1 起 venv 与 system oneAPI 都是 libsycl.so.9, 不再冲突,
#      但升级后旧 cache 仍绑定上一版 ABI)
#  3. TP>1 需要 oneCCL: CCL_ROOT 和 LD_LIBRARY_PATH 必须指向 .venv/lib,
#      该 wheel 把 libccl.so 与 lib/ccl 插件都放在那里
#
# 用法: ./run_vllm_tp.sh <TP> [port]
set -e

TP=${1:-8}
PORT=${2:-8000}
ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL=/home/lm/work/data/Qwen3.6-27B

# 清 triton cache, 强制重新 JIT (升级 intel-sycl-rt 后 ABI 会变)
rm -rf ~/.triton/cache ~/.cache/triton 2>/dev/null || true

# 干净环境: 无 icpx, 无 oneAPI LD_LIBRARY_PATH, 仅 venv bin
export PATH="/usr/bin:/bin:${ROOT}/.venv/bin"
unset LD_LIBRARY_PATH ONEAPI_ROOT ONEAPI_DEVICE_SELECTOR ZE_AFFINITY_MASK CCL_ROOT \
       CCL_CONFIGURATION ONEAPI_DEVICE_SELECTOR 2>/dev/null || true

# TP>1 需要 oneCCL 做 allreduce。oneccl wheel 把 libccl.so 和 lib/ccl 插件都放在
# .venv/lib，必须让 loader 找到它们，否则 core 起不来并报 "Could not load any plugin"。
export LD_LIBRARY_PATH="${ROOT}/.venv/lib"
export CCL_ROOT="${ROOT}/.venv"

echo "=== vLLM TP=${TP} start ==="
echo "PATH=${PATH}"
echo "icpx in PATH? $(which icpx 2>/dev/null || echo 'NO (good)')"

# `vllm serve` (推荐入口). 旧的 `python -m vllm.entrypoints.openai.api_server`
# 已标记 deprecated，vLLM 0.32 起只是转发到 launchers，故改用 CLI。
exec "${ROOT}/.venv/bin/vllm" serve "${MODEL}" \
    --tensor-parallel-size "${TP}" \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --port "${PORT}"