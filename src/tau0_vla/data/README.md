# Data pipeline

`tau0_vla.data` maps LeRobot datasets into the model’s state, action, image,
and language inputs, then restores model outputs to native robot actions.

See:

- [Dataset format](DATASET_FORMAT.md)
- [Config template](../../../configs/_template/README.md)
- [Robot adapters](../adapters/README.md)

## Robot Config

A Robot Config defines the complete data contract:

| field | purpose |
|---|---|
| `repo_id` | one dataset path, a list of paths, or a manifest |
| `state`, `action` | state/action modalities |
| `images` | camera names and image transforms |
| `prompt_source`, `prompt` | language instruction source and template |
| `frame_filter` | valid training anchors |
| `action_horizon` | action chunk length |
| `norm_stats_path` | normalization statistics |

Training imports the `data.py` next to the YAML and resolves
`data_args.config_name`. See
[`configs/example_agibot_world_gong/data.py`](../../../configs/example_agibot_world_gong/data.py)
for a complete example.

## Batch contract

With the bundled example and a 30-step horizon:

| key | shape |
|---|---|
| `state` | `(B, 40)` |
| `action` | `(B, 30, 40)` |
| `state_mask` | `(B, 40)` |
| `action_mask` | `(B, 40)` |
| `images[cam]` | `(B, 224, 224, 3)` |
| `prompt` | `B` strings |

Images are uint8 HWC. Masks identify active unified slots; the bundled
joint-control example activates 16 dimensions.

## Modalities

State/action components include `ArmJoint`, `EefPose`, `Gripper`, `Waist`, and
`ChassisVelocity`. Common transforms include:

- `RelativeToState()` for relative actions
- `Quat2Rot6D()` for EEF orientation
- `ResizeWithPad(...)` and `ColorJitter(...)` for images

## Instructions and frame filtering

```python
prompt_source=PromptSource.from_label(source="instruction_segments")
frame_filter=FrameFilter(positive=["l3"], negative=[])
```

Prompt labels may come from segment-level (`l3`), subtask-level (`l2`), or
episode-level (`l1`) annotations. The action horizon is kept inside the selected
segment.

## Normalization statistics

For unified 40D configs:

```bash
PYTHONPATH=src:. python3 scripts/norm_stats/compute_unified_ft_stats.py \
    --body <robot-config> \
    --action-horizon <horizon> \
    --repos <dataset> \
    --partials-dir /tmp/tau0_stats

PYTHONPATH=src:. python3 scripts/norm_stats/merge_stats.py \
    --partials /tmp/tau0_stats \
    --out <norm-stats.json>
```

Use the same datasets, frame filter, and action horizon as training.

## Inference

The post-training checkpoint stores its Data Spec. Deployment uses it to apply
the same prompt, image, state, mask, and normalization transforms:

```python
from tau0_vla.data import encode_payload, load_data_spec, restore_action

data_spec = load_data_spec(checkpoint_dir, route=route)
encoded = encode_payload(payload, data_spec)
restore_state = encoded.get("state_abs", encoded["state"])
actions = restore_action(
    model_output,
    data_spec,
    state=restore_state,
)
```

[`deploy/policy.py`](../../../deploy/policy.py) is the reference inference
wrapper.
