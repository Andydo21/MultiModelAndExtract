#!/bin/sh
set -eu
cd /app
runtime=/app/scratch/a3-linux-runtime
llama=/app/scratch/a3-models/llama-linux-cuda128
mkdir -p "$runtime" "$llama" /app/scratch/runtime-temp
export TMPDIR=/app/scratch/runtime-temp
export PIP_NO_CACHE_DIR=1
python -m venv --copies "$runtime"
"$runtime/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu121 'torch==2.5.1' 'torchvision==0.20.1'
"$runtime/bin/python" -m pip install 'surya-ocr==0.22.1' 'gliner==0.2.29' 'gliner2==2.1.0' 'transformers==5.16.1' 'sentence-transformers==3.1.1' 'accelerate==1.12.0' 'fastapi==0.129.0' 'uvicorn==0.40.0' 'pymupdf==1.28.0' 'scikit-learn==1.7.2' 'rapidfuzz==3.14.5' 'peft==0.18.1' 'timm==1.0.22' 'numpy==1.26.4'
for archive in llama-b11557-bin-ubuntu-cuda-12.8-x64.tar.gz cudart-llama-b11557-bin-ubuntu-cuda-12.8-x64.tar.gz; do
  ARCHIVE="$archive" python - <<'PY'
import os, requests
from pathlib import Path
name = os.environ['ARCHIVE']
path = Path('/app/scratch/runtime-temp') / name
url = 'https://github.com/ggml-org/llama.cpp/releases/download/b11557/' + name
with requests.get(url, stream=True, timeout=(30, 300)) as response:
    response.raise_for_status()
    with path.open('wb') as dest:
        for chunk in response.iter_content(1024 * 1024):
            if chunk:
                dest.write(chunk)
print('downloaded', name, path.stat().st_size, flush=True)
PY
  tar -xzf "$TMPDIR/$archive" -C "$llama"
  rm -f "$TMPDIR/$archive"
done
"$runtime/bin/python" -c 'import torch, transformers, surya, gliner, gliner2, fastapi, uvicorn; print("Runtime dependencies loaded:", torch.__version__, "CUDA:", torch.cuda.is_available())'
touch "$runtime/.ready"
