#!/bin/sh
set -eu
cd /app
runtime=/app/scratch/a3-linux-runtime
llama=/app/scratch/a3-models/llama-linux-cuda128
test -f "$runtime/.ready" || { echo 'A3 Linux runtime is not installed'; exit 1; }
binary=$(find "$llama" -type f -name llama-server | head -n 1)
test -n "$binary" || { echo 'llama-server is missing'; exit 1; }
export LLAMA_CPP_BINARY="$binary"
export LD_LIBRARY_PATH="$(find "$llama" -type f -name '*.so*' -printf '%h\n' | sort -u | paste -sd ':' -):${LD_LIBRARY_PATH:-}"
export TMPDIR=/app/scratch/runtime-temp
export PYTHONDONTWRITEBYTECODE=1
exec "$runtime/bin/python" -B -m uvicorn ai_extract.views.inference_app:app --host 0.0.0.0 --port 8010 --workers 1
