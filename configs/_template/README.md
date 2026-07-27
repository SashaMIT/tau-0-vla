# Config template

Copy this directory for a new post-training task:

```bash
cp -r configs/_template configs/my_task
grep -RIn 'YOUR_' configs/my_task
```

| file | owns |
|---|---|
| `data.py` | dataset roots, robot adapter, cameras, prompts, filters, horizon, modalities, and statistics |
| `train.yaml` | base checkpoint, matching data-config name, model settings, and optimization |

Keep `data.py` beside the YAML so training and deployment can discover the
same registered config.

## Values that must agree

| `data.py` | `train.yaml` / other file |
|---|---|
| normalized `@register_config` name (`_` becomes `-`) | normalized `data_args.config_name` |
| each `Image(name)` | `data_args.camera_keys` and `cam_view_names` |
| `Image(name)` | adapter `repack["images"]` and deployment camera mapping |
| `ResizeWithPad(height, width)` | `max_pixels >= height * width` |
| `action_horizon` | the training action horizon and stats command |
| merged statistics `--out` | `norm_stats_path` |
| unified adapter registry key | `per_embodiment[registry_key]` in `norm_stats.json` |

If the config lives outside `configs/<task>/data.py`, list its importable dotted
module in `data_args.config_modules`. Missing config registration is a startup
error; the loader does not silently select another shape or robot.

## Select the route

For a component route, `state=[...]` and `action=[...]` order defines the model
concat order. Component transforms also define whether actions are absolute or
relative.

For a unified route:

- use the robot's `_UnifiedMixin` class;
- set `state_padding_dim=40` and `action_padding_dim=40`;
- keep component declarations at `normalize="none"`;
- let the unified assembler apply the registry mapping, masks, relative
  arm/EEF actions, and per-embodiment 40D statistics.

Do not add EEF modalities to a joint-only dataset. Native EEF training data
must be wired by the adapter as described in the
[adapter guide](../../src/tau0_vla/adapters/README.md). Public v1 serving routes
must use joint control; EEF serving is not supported.

## Configure the task

In `data.py`:

1. replace the imported template adapter with the target robot class;
2. point `repo_id` to one dataset or a list of schema-compatible datasets;
3. declare every camera and image transform;
4. select the instruction source and prompt template;
5. select valid frames and the action horizon;
6. declare state/action modalities and padding;
7. point `norm_stats_path` to matching statistics.

All fixed camera resize targets must match. `camera_keys` order controls VLM
view order, and `max_images_per_sample` must be large enough not to truncate a
required camera.

In `train.yaml`:

1. set `model_name_or_path` to the released base checkpoint;
2. set `data_args.config_name` and the exact camera list;
3. keep image/token limits compatible with `data.py`;
4. choose the output directory and training schedule.

## Compute statistics

For a component route:

```bash
PYTHONPATH=src:. python3 -m tau0_vla.data compute_norm YOUR_CONFIG_NAME \
    --config-file configs/my_task/data.py \
    --output-dir /path/to/stats
```

Point `norm_stats_path` to the generated file.

For a unified route:

```bash
PYTHONPATH=src:. python3 scripts/norm_stats/compute_unified_ft_stats.py \
    --body YOUR_UNIFIED_ROBOT_NAME \
    --action-horizon YOUR_HORIZON \
    --repos /path/to/dataset \
    --positive-labels YOUR_POSITIVE_LABELS \
    --negative-labels YOUR_NEGATIVE_LABELS \
    --partials-dir /tmp/my_task_stats

PYTHONPATH=src:. python3 scripts/norm_stats/merge_stats.py \
    --partials /tmp/my_task_stats \
    --out /path/to/data-root/norm_stats/YOUR_TASK-unified-40d.json
```

Use the same path in `data.py`'s `_NORM_STATS`. Check that the merged file
contains the adapter's registry key and that every state/action statistic is
40D. Recompute after changing the dataset mix, filter, horizon, mapping,
transforms, or EEF/joint mode.

## Train

```bash
bash scripts/train.sh configs/my_task/train.yaml
```

Before a full run, inspect one loader batch for instruction, camera views,
state/action values, active mask indices, and shapes. After saving, reload its
Data Spec and verify `restore_action`. For a joint-control serving route, also
verify the deployment adapter's SDK order.
