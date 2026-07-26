# The data pipeline

`tau0_vla.data`. You describe a dataset once as a **Robot Config**, register it by
name, and both training and inference read that same description — training to
build batches, inference to invert the normalization and reconstruct actions in
the robot's own units.

This is the reference for *calling* the pipeline. See also:

- [`DATASET_FORMAT.md`](DATASET_FORMAT.md) — what a dataset must contain
- [`../adapters/README.md`](../adapters/README.md) — supporting your own robot
- [`../../../configs/_template/`](../../../configs/_template/) — a config to copy

## Quick start

```python
from tau0_vla.data import FinchDataLoader, FrameFilter, PromptSource, register_config
from tau0_vla.data.modalities import ArmJoint, Gripper, Image, Prompt, RelativeToState
from tau0_vla.data.modalities.image import ColorJitter, ResizeWithPad
from tau0_vla.adapters.g1 import G1Agibot


@register_config
def my_task() -> G1Agibot:
    return G1Agibot(
        repo_id="/path/to/dataset_v30",
        state=[ArmJoint(normalize="mean_std"), Gripper(normalize="mean_std")],
        action=[ArmJoint(normalize="mean_std", transforms=[RelativeToState()]),
                Gripper(normalize="mean_std")],
        action_horizon=30,
        images=[Image(cam, transforms=[ColorJitter(prob=0.33), ResizeWithPad(224, 224)])
                for cam in ("head", "wrist_left", "wrist_right")],
        state_padding_dim=40,
        action_padding_dim=40,
        prompt_source=PromptSource.from_label(source="instruction_segments"),
        frame_filter=FrameFilter(positive=["l3"], negative=["error_frame"]),
        prompt=Prompt(template="You are controlling a robot.\nTask: {instruction}"),
        norm_stats_path="/path/to/norm_stats.json",
    )


loader = FinchDataLoader.from_config_name("my_task", batch_size=8, num_workers=4)
batch = next(iter(loader))
```

Registration is by import side effect: the decorator runs when the module is
imported. A config therefore reaches the registry exactly one of two ways.

**1. A `data.py` in the same directory as your YAML.** Training imports it by
file path before it resolves `config_name`. The same-directory part is a hard
requirement, not a convention — move a YAML away from its `data.py` and its
configs never register.

**2. `data_args.config_modules`** — a list of importable dotted module paths,
each calling `@register_config` at import time:

```yaml
data_args:
  config_name: my_robot_ft
  config_modules:
    - my_package.robot_configs
```

Use this when the config lives outside the YAML's directory. The list is also
written into the checkpoint's `policy_manifest.json`, and every deploy entry point
re-imports it from there — so serving and open-loop resolve the same config
without being told again. (`deploy/openloop_with_server.py` is the exception: it
talks to a remote server and has no local checkpoint to read, so it takes
`--config-module` instead.)

If neither route registers `config_name`, training stops at startup and names
both places it looked. It does not fall back: `state_dim` and `action_dim` are
derived from the config, and a fallback would silently build an action head that
disagrees with the data.

## Batch contract

Measured on the shipped example with `batch_size=2`, `action_horizon=30`,
`state_padding_dim=action_padding_dim=40`, three cameras at 224×224:

| key | type | shape | dtype |
|---|---|---|---|
| `state` | tensor | `(B, 40)` | float32 |
| `action` | tensor | `(B, 30, 40)` | float32 |
| `state_mask` | tensor | `(B, 40)` | float32 |
| `action_mask` | tensor | `(B, 40)` | float32 |
| `images` | dict of tensor | `(B, 224, 224, 3)` per camera | **uint8, HWC** |
| `prompt` | list of str | length B | |
| `extras` | dict | present when `return_all_norm_forms=True` | |

Notes that matter in practice:

- **Images are uint8 HWC**, not float CHW. Convert at the model boundary.
- **Masks mark which slots this route actually supervises.** In the example, 20
  of 40 (two 9D end-effector blocks + two gripper dimensions) are active; the
  remaining 20 are padding for the cross-embodiment layout and carry no signal.
  Use the masks to exclude padded dimensions from the loss.
- `state` has no time axis unless the config asks for one; `action` always does,
  of length `action_horizon`.
- With `temporal_sequence`, `state` and each camera gain a leading time axis:
  `state (B, T, D)`, `images[cam] (B, T, H, W, 3)`.

### `extras`

With `return_all_norm_forms=True`, both `state` and `action` are also returned
in every normalization form:

```
extras["state"]  = {"raw": (B, 40), "mean_std_norm": (B, 40), "q_norm": (B, 40)}
extras["action"] = {"raw": (B, 30, 40), "mean_std_norm": ..., "q_norm": ...}
```

`raw` is pre-normalization but post-transform (so a `RelativeToState` action is
already a delta). Useful when a downstream consumer needs a different
normalization than the one the model trains on.

## Robot Config fields

`repo_id` accepts three forms, all equivalent downstream:

```python
repo_id="/path/to/one_dataset_v30"                    # a single dataset
repo_id=["/path/a_v30", "/path/b_v30"]                # several, concatenated
repo_id="/path/manifest.txt"                          # one dataset path per line
```

| field | purpose |
|---|---|
| `state`, `action` | component lists; see Modalities below |
| `action_horizon` | frames per Action Chunk |
| `images` | one `Image(camera_name, transforms=[...])` per camera |
| `prompt_source` | where the per-sample instruction comes from |
| `prompt` | `Prompt(template=...)`, wrapping `{instruction}` |
| `frame_filter` | which frames are valid anchors |
| `state_padding_dim`, `action_padding_dim` | zero-pad to a fixed width |
| `norm_stats_path` | load statistics from this exact file |
| `norm_stats_dir` | load by fingerprint from this directory, computing them if absent |
| `return_all_norm_forms` | add `extras` to every batch |
| `temporal_sequence` | attach a time axis; see below |
| `source_kwargs` | passed through to LeRobot (`episodes`, `tolerance_s`, `video_backend`, …) |

`norm_stats_path` and `norm_stats_dir` are mutually exclusive. The path form
loads verbatim and skips the fingerprint check; the directory form derives a
filename from the config's fingerprint and computes the statistics on first use.

## Modalities

Components declare *what* the state and action vectors contain. They are
concatenated in the order listed.

| component | width | notes |
|---|---|---|
| `ArmJoint` | robot-defined | joint positions |
| `EefPose` | 7 per arm | xyz + quaternion; usually with `Quat2Rot6D` |
| `Gripper` | 1 per gripper | |
| `Waist` | robot-defined | |
| `ChassisVelocity` | robot-defined | |

Each takes `normalize=` (`"mean_std"`, `"quantile"`, or `"none"`) and an
optional `transforms=` list:

| transform | effect |
|---|---|
| `Quat2Rot6D()` | `xyz+quat` (7) → `xyz+rot6d` (9), avoiding quaternion sign ambiguity |
| `RelativeToState()` | action becomes a delta from the current state |

Image transforms live in `tau0_vla.data.modalities.image`:

| transform | effect |
|---|---|
| `ResizeWithPad(h, w)` | resize preserving aspect ratio, pad the remainder |
| `ColorJitter(prob=..., brightness=..., ...)` | probabilistic augmentation, training only |

## Prompt sources

```python
PromptSource.fix("stack the blocks")                    # one instruction for everything
PromptSource.from_label(source="instruction_segments")  # per-subtask, from annotations
PromptSource.random([a, b], probabilities=[0.2, 0.8])   # mix, resolved per sample
```

`from_label` accepts `instruction_segments` (alias `l3`), `subtask_frame`
(alias `l2`), `high_level_instruction` (alias `l1`), and `parquet` (a column in
the data files). [`DATASET_FORMAT.md`](DATASET_FORMAT.md) documents what each
one reads, including the `l2` → `l3` fallback.

`Prompt(template=...)` then wraps whatever the source produced:

```python
prompt=Prompt(template="You are controlling a robot.\nRobot type: G1\nTask: {instruction}")
```

## Frame filtering

`FrameFilter` decides which frames may serve as anchors. `positive` labels
intersect; `negative` labels are subtracted; a tail buffer of
`action_horizon - 1` frames keeps every Action Chunk inside its segment.

```python
FrameFilter(positive=["l3"], negative=["error_frame"])
```

| label | positive | negative | source |
|---|---|---|---|
| `l3` / `instruction_segments` | ✓ | | per-subtask instruction spans |
| `l2` / `subtask_frame` | ✓ | | sub-task intervals in `key_frame` |
| `l1` / `high_level_instruction` | ✓ | | episode-level instruction |
| `error_frame` | | ✓ | error spans in `key_frame` |

The shipped example reads detailed segment instructions and uses
`FrameFilter(positive=["l3"], negative=[])`. Its 30-step horizon leaves 5,484
valid annotated anchors from 9,469 frames across 25 episodes. The loader prints
the retained count at startup — worth reading before letting a run proceed.

## Normalization statistics

Statistics are fitted over the *filtered anchor stream*, so they depend on the
frame filter as well as the data. They are keyed by a fingerprint of the Robot
Config — robot class, components, transforms, action horizon, dataset basenames —
so two configs that differ only cosmetically share a cached file.

Compute them ahead of time:

```bash
python -m tau0_vla.data compute_norm my_task_stats \
    --config-file configs/my_task/data.py \
    --output-dir /path/to/norm_stats --workers 32
```

`--config-file` imports the file so its `@register_config` fires; use
`--config-module` instead if your config lives in an importable package.

The convention is a stats-only twin of your training config: same components,
filters, horizon and repos, but with `norm_stats_dir` instead of
`norm_stats_path` so the statistics can be written rather than read.

Roughly 90k samples/s on a single node; the example's 1.8M frames take about 20
seconds.

This fits statistics over the **component** vectors. A unified robot config
normalizes in the 40D Unified Layout and addresses its statistics by
`_unified_registry_key` under a `per_embodiment` block, which this command cannot
produce — pass it one and normalization raises rather than mis-scaling. Use
`scripts/norm_stats/compute_unified_ft_stats.py` and `merge_stats.py` for those;
they compute over every anchor rather than a sample, so they take minutes instead
of seconds.

## Time windows

`temporal_sequence` attaches a time axis. Offsets are **in seconds** and may be
negative (past), zero (present) or positive (future):

```python
temporal_sequence={
    "image": [-2/30, -1/30, 0.0],   # three frames ending at the anchor
    "state": [-1/30, 0.0],          # one past state plus the present
}
```

Keys are `state`, `action`, `image` (all cameras) or `image.<camera>` (one).

## Inference

Training and inference share one Data Spec, written into every checkpoint at
`finch_data_spec/`. That is what makes the two halves consistent: the same
prompt template, image resize and state pipeline are applied by construction,
and cannot be silently re-implemented differently at serving time.

```python
from tau0_vla.data import encode_payload, load_data_spec, restore_action

data_spec = load_data_spec(checkpoint_dir, route=route)
encoded = encode_payload(payload, data_spec)     # native robot units in
actions = restore_action(model_output, data_spec, state=encoded["state"])
```

`encode_payload` takes a dict of raw observations — the robot's native state
vector plus uint8 images keyed by camera — and returns normalized model inputs.
`restore_action` inverts everything the config declared, in reverse order:
un-normalize, undo `RelativeToState` against the state reference, convert rot6d
back to quaternions.

For unified routes, `encode_payload` also returns `state_abs` (the absolute
scattered vector) and `action_mask`. Pass `state_abs` to `restore_action`, not
the normalized `state` — the relative-to-absolute inverse needs absolute values.

`deploy/policy.py` is the reference implementation and is deliberately thin;
copy it as the starting point for another architecture.
