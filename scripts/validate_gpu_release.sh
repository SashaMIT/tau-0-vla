#!/usr/bin/env bash
# End-to-end release smoke test on one GPU.
#
# Usage:
#   bash scripts/validate_gpu_release.sh \
#       /path/to/public-checkpoint \
#       /path/to/smoke-output
#
# The output directory is retained intentionally: a successful run should leave
# a loadable post-training checkpoint for inspection.
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <public-checkpoint> [output-dir]" >&2
    exit 2
fi

CHECKPOINT_DIR=$(realpath "$1")
OUTPUT_DIR=${2:-"/tmp/tau0-vla-gpu-smoke-$(date +%Y%m%d-%H%M%S)"}

export CHECKPOINT_DIR
export OUTPUT_DIR
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NPROC_PER_NODE=1
export AUTO_RESUME=0
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for path in \
    "$CHECKPOINT_DIR/model.safetensors" \
    "$CHECKPOINT_DIR/config.json" \
    "$CHECKPOINT_DIR/chat_template.jinja" \
    "$CHECKPOINT_DIR/processor_config.json" \
    "$CHECKPOINT_DIR/tokenizer.json" \
    "$CHECKPOINT_DIR/tokenizer_config.json" \
    "example_data/meta/info.json"
do
    if [ ! -f "$path" ]; then
        echo "Missing required release input: $path" >&2
        exit 1
    fi
done

"${PYTHON_BIN:-python3}" - <<'PY'
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path

import torch

checkpoint = Path(os.environ["CHECKPOINT_DIR"])
forbidden_names = {
    "latest",
    "policy_manifest.json",
    "resolved_config_full.yaml",
    "run_spec.json",
    "scheduler.pt",
    "trainer_state.json",
    "training_args.bin",
    "zero_to_fp32.py",
}
forbidden_prefixes = ("global_step", "multi_dataset_state_rank", "rng_state_", "optimizer")
for path in checkpoint.rglob("*"):
    if path.name in forbidden_names or path.name.startswith(forbidden_prefixes):
        raise SystemExit(f"Base checkpoint contains excluded training state: {path}")

config = json.loads((checkpoint / "config.json").read_text())


def walk(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk(child, (*path, str(index)))
    else:
        yield path, value


for path, value in walk(config):
    if path and path[-1] == "_name_or_path" and isinstance(value, str) and value.startswith("/"):
        raise SystemExit(f"config.json contains an absolute source path at {'.'.join(path)}")

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("The selected GPU does not support bf16")

print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
for package in (
    "torchvision",
    "torchcodec",
    "transformers",
    "flash-attn",
    "flash-linear-attention",
    "deepspeed",
    "tau0_vla",
):
    print(f"{package}={metadata.version(package)}")

data_py = Path("configs/example_agibot_world_gong/data.py").resolve()
spec = importlib.util.spec_from_file_location("tau0_public_example", data_py)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from tau0_vla.data import FinchDataLoader

loader = FinchDataLoader.from_config_name(
    "agibot_world_gong_ft",
    batch_size=1,
    num_workers=0,
    shuffle=False,
)
batch = next(iter(loader))
assert tuple(batch["state"].shape) == (1, 40)
assert tuple(batch["action"].shape) == (1, 30, 40)
assert int(batch["state_mask"].sum()) == 16
assert int(batch["action_mask"].sum()) == 16
assert set(batch["images"]) == {"head", "wrist_left", "wrist_right"}
for image in batch["images"].values():
    assert tuple(image.shape) == (1, 224, 224, 3)
print(f"dataset_anchors={len(loader.dataset)}")
print("data_batch=ok")
PY

mkdir -p "$OUTPUT_DIR"

bash scripts/train.sh configs/example_agibot_world_gong/train.yaml \
    --model_name_or_path "$CHECKPOINT_DIR" \
    --run_name gpu_release_smoke \
    --output_dir "$OUTPUT_DIR" \
    --max_steps 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 0 \
    --dataloader_persistent_workers false \
    --logging_steps 1 \
    --save_strategy no \
    --report_to none

if [ ! -f "$OUTPUT_DIR/gpu_release_smoke/model.safetensors" ]; then
    echo "Training finished but did not write the final model.safetensors" >&2
    exit 1
fi
if [ ! -f "$OUTPUT_DIR/gpu_release_smoke/policy_manifest.json" ]; then
    echo "Training finished but did not write policy_manifest.json" >&2
    exit 1
fi

echo "GPU release smoke test passed."
echo "Saved post-training output: $OUTPUT_DIR/gpu_release_smoke"
