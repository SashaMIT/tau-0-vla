# Config template

```bash
cp -r configs/_template configs/my_task
grep -rn 'YOUR_' configs/my_task     # your checklist
```

Nothing here resolves until the placeholders are gone. A template that appeared
to work would invite you to leave a field unchanged and train against the wrong
column.

## What you are filling in

| file | what it owns |
|---|---|
| `data.py` | your dataset: repo paths, camera names, annotation tracks, prompt, and the `@register_config` name |
| `train.yaml` | the run: checkpoint to warm-start from, step count, and the camera keys again (they must agree with `data.py`) |

The two files must stay in the same directory. `train.py` imports
`<yaml_dir>/data.py` by path so the `@register_config` fires, and derives
`state_dim` / `action_dim` / `action_horizon` from the config it finds. Separated,
training stops at startup and names both places it looked — it does not fall back
to a 16D head against a 40D pipeline.

Three values appear in both files and have to match:

| `data.py` | `train.yaml` |
|---|---|
| the `@register_config` function name | `data_args.config_name` |
| each `Image("name", ...)` | an entry in `data_args.camera_keys` |
| `ResizeWithPad(H, W)` | `data_args.max_pixels` (≥ H×W) |

## Then

1. Your robot needs an adapter — a class that maps your dataset's columns to the
   pipeline's semantic names. See
   [`../../src/tau0_vla/adapters/README.md`](../../src/tau0_vla/adapters/README.md);
   `_template/` there is the skeleton. The import at the top of `data.py` points
   at the G1's, which works if your layout matches it.
2. Your dataset has to conform to
   [`../../src/tau0_vla/data/DATASET_FORMAT.md`](../../src/tau0_vla/data/DATASET_FORMAT.md).
3. Compute normalization statistics, then launch training with
   `bash scripts/train.sh configs/my_task/train.yaml`.
