# Data pipeline

`tau0_vla.data` converts LeRobot samples into model-ready state, action, image,
and language inputs, and stores enough of that contract to restore model
outputs to robot actions.

Related documentation:

- [Dataset format](DATASET_FORMAT.md)
- [Robot adapters](../adapters/README.md)
- [Config template](../../../configs/_template/README.md)

## Config resolution

A function decorated with `@register_config` returns one `RobotConfig`.
Training resolves `data_args.config_name` in one of two ways:

1. it imports `data.py` from the same directory as the training YAML; or
2. it imports the dotted modules listed in `data_args.config_modules`.

The second form is recorded in the checkpoint manifest so deployment can import
the same module. If the requested name is not registered, startup fails; there
is no implicit fallback config.

The bundled example keeps these together:

```text
configs/example_agibot_world_gong/
├── data.py
├── train.yaml
└── norm_stats.json
```

## RobotConfig

A `RobotConfig` defines the complete sample contract:

| field | purpose |
|---|---|
| `repo_id` | one dataset root, a list of compatible roots, or a manifest |
| `state`, `action` | ordered modality declarations |
| `images` | model-facing camera names and transforms |
| `prompt_source`, `prompt` | instruction lookup and final prompt template |
| `frame_filter`, `filter_by_segments` | valid training anchors |
| `action_horizon` | number of future action frames |
| `state_padding_dim`, `action_padding_dim` | model-facing widths; both are 40 for unified routes |
| `norm_stats_path` | explicit normalization statistics file |
| `source_kwargs` | LeRobot source options such as root, episodes, and video backend |
| `temporal_sequence` | optional state/image/action time offsets |
| `return_all_norm_forms` | include diagnostic raw and normalized forms |

`repo_id` may be a list only when every source follows the same adapter schema.
`temporal_sequence["action"]`, when set, overrides offsets generated from
`action_horizon`. Unified arm/EEF relative actions currently require a
single-frame state; do not set `temporal_sequence["state"]` on a unified route.

## Batch contract

For a unified route with horizon `H`:

| key | type and shape |
|---|---|
| `state` | float32 `(B, 40)` |
| `action` | float32 `(B, H, 40)` |
| `state_mask` | float32 `(B, 40)` |
| `action_mask` | float32 `(B, 40)` |
| `images[cam]` | uint8 `(B, height, width, 3)`, HWC |
| `prompt` | `list[str]` of length `B` |

Mask value `1` means that the unified slot is active for that sample. Inactive
slots are zero. `action_mask` is shared across and broadcast over the `H`
action steps. The bundled G1 joint example activates 16 of 40 dimensions.

When `return_all_norm_forms=True`, the batch also contains `extras`. τ₀-VLA
does not consume it directly, although a downstream transform may consume a
form such as `extras["action"]["q_norm"]`.

## State and action assembly

Component routes apply modality transforms in their declared order. Their
model-space concatenation and `restore_action` order follow that component
order; padding produces a contiguous active prefix rather than the semantic
40D slots described below. Available state/action components include:

- `ArmJoint`
- `EefPose`
- `Gripper`
- `Waist`
- `ChassisVelocity`

Common transforms include `RelativeToState()` for relative actions and
`Quat2Rot6D()` for EEF orientation.

Unified routes use the fixed
[40D adapter contract](../adapters/README.md#unified-40d-contract):

```text
native flat state/action
    -> adapter registry scatter
    -> choose native EEF or joint fallback
    -> build state_mask and action_mask
    -> make arm/EEF action relative to current state
    -> normalize with per-embodiment 40D statistics
    -> zero inactive slots
```

Registry mappings for state and action are independent. Native EEF fields are
adapter inputs, not FK results. If EEF is active for a sample, arm-joint slots
are disabled so the same motion is not represented twice.

Arm and EEF action chunks are relative to the current single-frame state:

- arm slots use joint deltas;
- EEF uses body-frame position and rotation deltas;
- gripper, waist, and chassis velocity remain absolute.

This relative conversion still happens while computing normalization
statistics, so action statistics describe the values seen by the model.
Every active arm/EEF action slot must have the corresponding current-state slot
in the same unit and order; state and action masks are built independently.

## Images

Each `Image(name, transforms=[...])` names a model-facing view. That name must
match:

- the key in the adapter's `repack["images"]`;
- `data_args.camera_keys` in the YAML;
- the deployment adapter's camera mapping.

`camera_keys` order is the VLM image/view order, and
`max_images_per_sample` may truncate cameras from the tail. Keep names,
left/right semantics, order, and count identical across the adapter, Image
specs, YAML, and deployment. All fixed `ResizeWithPad` outputs in one Data Spec
must use the same target size.

Typical transforms are:

```python
Image(
    "head",
    transforms=[
        ColorJitter(...),
        ResizeWithPad(224, 224),
    ],
)
```

Training applies stochastic transforms such as `ColorJitter`; inference uses
the same spec with training-only randomness disabled.

## Instructions and frame filtering

`PromptSource` supports:

- `PromptSource.fix(...)` for a constant instruction;
- `PromptSource.from_label(...)` for a parquet field or `l1`/`l2`/`l3`
  annotation;
- `PromptSource.random(...)` for a weighted mixture of sources.

Annotation aliases are:

| label | source |
|---|---|
| `l1` | episode-level `high_level_instruction` |
| `l2` | subtask interval in `key_frame` / `subtask_frame` |
| `l3` | frame interval in `instruction_segments` |

For example:

```python
prompt_source=PromptSource.from_label(source="instruction_segments")
frame_filter=FrameFilter(positive=["l3"], negative=["error_frame"])
```

Positive filters are intersected and negative filters are removed. An explicit
`frame_filter` takes precedence over `filter_by_segments`; otherwise
`filter_by_segments` selects the default policy. An `l3` policy keeps the full
action horizon inside the instruction segment. Positive frame filters accept
`l2`/`l3`; `l1` is a prompt source, not a frame filter.

For unified mixed-control training, an exact `Control mode:` line in the prompt
is rewritten to `eef` or `joint` from the sample's action route. Prompts without
that line are unchanged.

## Normalization statistics

Unified statistics are 40D and stored per registry key under
`per_embodiment`. Inactive slots do not contribute to their estimates.

```bash
PYTHONPATH=src:. python3 scripts/norm_stats/compute_unified_ft_stats.py \
    --body YOUR_UNIFIED_ROBOT_NAME \
    --action-horizon YOUR_HORIZON \
    --repos /path/to/dataset \
    --positive-labels YOUR_POSITIVE_LABELS \
    --negative-labels YOUR_NEGATIVE_LABELS \
    --partials-dir /tmp/tau0_stats

PYTHONPATH=src:. python3 scripts/norm_stats/merge_stats.py \
    --partials /tmp/tau0_stats \
    --out /path/to/norm_stats.json
```

Dataset roots, positive/negative filters, and action horizon must match the
training config. The merge step verifies horizon and filter compatibility
across partials; it cannot detect a changed adapter mapping. Recompute after
changing any mapping, transform, or mask.

## Inference and action restoration

The checkpoint stores a Data Spec containing its `robot_name`, registry key,
config modules, cameras, transforms, mask/assembly identifiers, and
normalization contract. Installed adapter/registry code resolves those
identifiers back to current code.
Deployment reuses that spec:

```python
from tau0_vla.data import encode_payload, load_data_spec, restore_action

data_spec = load_data_spec(checkpoint_dir, route=route)
encoded = encode_payload(payload, data_spec)

# Obtain one normalized action chunk: (H, data_spec.action_dim), 40D if unified.
model_output = ...
actions = restore_action(
    model_output,
    data_spec,
    state=encoded.get("state_abs", encoded["state"]),
)
```

For unified routes, `state_abs` is the absolute, unnormalized, scattered 40D
state needed to undo relative arm/EEF actions. Passing normalized `state`
instead produces incorrect absolute actions.

Restoration runs in reverse order:

```text
unnormalize -> unrelative -> convert EEF rot6d to quat_xyzw
            -> gather active semantic slices
```

`restore_unified_action` returns named semantic arrays.
`restore_action` concatenates them in canonical semantic order; a deployment
adapter must still reorder that flat result when the SDK uses a different
native column order. [`deploy/policy.py`](../../../deploy/policy.py) is the
reference inference wrapper.
