#!/usr/bin/env bash
# Reproducible installation for a fresh CUDA 12.8 Python 3.11+ environment.
#
# Override PYTORCH_INDEX_URL for another compatible wheel source. Override
# MAX_JOBS to control flash-attn's local compilation parallelism.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
MAX_JOBS="${MAX_JOBS:-4}"

"$PYTHON_BIN" -c '
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11+ is required, got {sys.version.split()[0]}")
'

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel packaging ninja

# Install ABI-coupled PyTorch packages together from one CUDA wheel index.
"$PYTHON_BIN" -m pip install \
    --index-url "$PYTORCH_INDEX_URL" \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchcodec==0.5.0

"$PYTHON_BIN" -m pip install -r requirements.txt

# flash-attn's official installation path expects torch to be visible in the
# build environment. MAX_JOBS=4 avoids exhausting RAM on many-core hosts.
MAX_JOBS="$MAX_JOBS" "$PYTHON_BIN" -m pip install \
    flash-attn==2.7.3 \
    --no-build-isolation

"$PYTHON_BIN" -m pip install -e .

"$PYTHON_BIN" -c '
import importlib.metadata as metadata
import torch

for package in (
    "torch",
    "torchvision",
    "torchcodec",
    "transformers",
    "flash-attn",
    "flash-linear-attention",
    "deepspeed",
    "tau0_vla",
):
    print(f"{package}=={metadata.version(package)}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"torch_cuda={torch.version.cuda}")
print(f"gpu_count={torch.cuda.device_count()}")
'
