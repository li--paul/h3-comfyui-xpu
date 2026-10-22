#!/usr/bin/env sh
set -e
cd "$(dirname "$0")"

# Detect Python: prefer the ComfyUI portable build, otherwise fall back to PATH.
PY="python3"
if [ -x "../../../python_embeded/bin/python3" ]; then
    PY="../../../python_embeded/bin/python3"
fi

echo "Installing the local llama-server runtime (llama.cpp b10436)..."
echo "Python used: $PY"
echo

"$PY" install_runtime.py "$@"

echo
echo "[OK] Runtime installed. Restart ComfyUI to use local LLM/VLM nodes."
