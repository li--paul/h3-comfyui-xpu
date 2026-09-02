#!/bin/bash
# vLLM TP 推理启动脚本 (Intel Arc Pro B60, XPU, torch 2.12)
#
# 关键环境要求:
#  1. PATH 不含 oneAPI bin (无 icpx) → triton 走 intel-sycl-rt 分支, 用 venv libsycl.so.8
#  2. 清空 triton cache → JIT 产物与 .8 重新匹配
#  3. 不引入 system oneAPI 的 libsycl.so.9 (LD_LIBRARY_PATH 保持干净)
#
# 用法: ./run_vllm_tp.sh <TP> [port]
set -e

TP=${1:-8}
PORT=${2:-8000}
ROOT="$(cd "$(dirname "$0")" && pwd)"
MODEL=/home/lm/work/data/Qwen3.6-27B

# 清 triton cache, 强制重新 JIT (与 venv libsycl.so.8 匹配)
rm -rf ~/.triton/cache ~/.cache/triton 2>/dev/null || true

# 干净环境: 无 icpx, 无 oneAPI LD_LIBRARY_PATH, 仅 venv bin
export PATH="/usr/bin:/bin:${ROOT}/.venv/bin"
unset LD_LIBRARY_PATH ONEAPI_ROOT ONEAPI_DEVICE_SELECTOR ZE_AFFINITY_MASK CCL_ROOT \
       CCL_CONFIGURATION ONEAPI_DEVICE_SELECTOR 2>/dev/null || true

echo "=== vLLM TP=${TP} start ==="
echo "PATH=${PATH}"
echo "icpx in PATH? $(which icpx 2>/dev/null || echo 'NO (good)')"

exec "${ROOT}/.venv/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --tensor-parallel-size "${TP}" \
    --dtype bfloat16 \
    --enforce-eager \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --port "${PORT}"