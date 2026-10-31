#!/usr/bin/env bash
# Wrapper for gpu_monitor.py (real-time Intel Arc B70 utilization monitor).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# gpu_monitor.py is stdlib-only, so any interpreter works; prefer a repo venv when one
# exists and fall back to the system python instead of requiring .venv.
for v in "${GPU_MONITOR_VENV:-}" "$ROOT/.venv" "$ROOT/.venv-comfy" "$ROOT/.venv-omni"; do
    [ -n "$v" ] && [ -x "$v/bin/python" ] && exec "$v/bin/python" "$ROOT/gpu_monitor.py" "$@"
done
exec python3 "$ROOT/gpu_monitor.py" "$@"
