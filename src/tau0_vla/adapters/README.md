# Robot adapters

Adapters are the boundary between a robot's native data/SDK layout and the
model-facing data contract. Embodiment-specific column indices, camera names,
and hardware action order belong here, not in the general data pipeline.

Start from:

- [`_template/`](_template/README.md) when adding an embodiment
- [`g1/`](g1/) for the bundled AgiBot World implementation

Each adapter contains:

| file | owns |
|---|---|
| `layout.py` | dataset fields, native state/action semantics, camera aliases, and `RobotConfig` classes |
| `deploy_io.py` | live SDK payload parsing, state/image wiring, action-dict keys, and flat SDK action order |

## Unified 40D contract

`tau0_vla.data.robots.unified.UNIFIED_LAYOUT` is the source of truth. Ranges
below use Python slice notation: `start:end`.

| model slots | semantic value | representation |
|---|---|---|
| `0:3` | left EEF position | `(x, y, z)` in metres |
| `3:9` | left EEF orientation | rotation 6D: `R[:, 0]`, then `R[:, 1]` |
| `9:12` | right EEF position | `(x, y, z)` in metres |
| `12:18` | right EEF orientation | rotation 6D: `R[:, 0]`, then `R[:, 1]` |
| `18` | left gripper | one scalar |
| `19` | right gripper | one scalar |
| `20:22` | waist | up to two values, in adapter-declared order |
| `22:24` | chassis velocity | up to two values, in adapter-declared order |
| `24:32` | left arm joints | joints 0–7 in radians |
| `32:40` | right arm joints | joints 0–7 in radians |

Arms with fewer than eight joints occupy the lower slots in their block. The
remaining slots are exactly zero and have mask value `0`.

### Native columns to unified slots

[`dim_registry.json`](../data/assets/dim_registry.json) describes the native
flat-vector layout. Its `state_groups` and `action_groups` values are native
column indices, in joint order; they are not unified slot numbers. State and
action mappings are separate because their native layouts may differ.

The generic scatter accepts these group names:

| registry group | unified destination |
|---|---|
| `left_arm` | `24 + i`, up to 8 values |
| `right_arm` | `32 + i`, up to 8 values |
| `left_gripper` | slot `18`, first listed value |
| `right_gripper` | slot `19`, first listed value |
| `waist` | slots `20:22`, up to 2 values |
| `chassis_velocity` | slots `22:24`, up to 2 values |

Other registry metadata is descriptive unless an adapter consumes it. In
particular, `eef_dim` and `eef_inline_indices` do not enable EEF input
automatically. Validate index bounds, duplicates, left/right order, and group
width before training; unsupported groups do not enter the generic 40D tensor.

For the bundled `g1_agibot_36` route:

| value | native state columns | native action columns | unified slots |
|---|---:|---:|---:|
| left gripper | `0:1` | `0:1` | `18` |
| right gripper | `1:2` | `1:2` | `19` |
| left arm, 7 joints | `28:35` | `16:23` | `24:31` |
| right arm, 7 joints | `35:42` | `23:30` | `32:39` |

This route therefore has 16 active dimensions. Slots `31` and `39` are the
unused eighth-joint positions.

### EEF, joint fallback, and masks

Native EEF data is wired separately from the flat-vector groups. A unified
adapter declares `_eef_state_col` and `_eef_action_col`, plus
`_eef_format = "euler"` or `"quat"`. Euler input is
`[x, y, z, rx, ry, rz]`; quaternion input is
`[x, y, z, qx, qy, qz, qw]`. Both are converted to position + rotation 6D.

The routing rule is evaluated per sample:

- when native EEF values are present, EEF slots are active and arm-joint slots
  are masked off;
- when EEF values are absent, EEF slots are masked off and arm joints are the
  fallback;
- gripper, waist, and chassis masks are independent of that choice.

This prevents the same arm motion from being represented in both EEF and joint
space. The public G1 routes are joint-controlled: they do not ship a URDF and
never derive EEF poses from joints.

Training may read native EEF values from separate `_eef_state_col` /
`_eef_action_col` parquet features or from an adapter `_eef_provider()`. These
are data-assembly features, not a public v1 serving contract. The public server
supports joint-control routes only and rejects routes with EEF action slices.
Registry `eef_inline_indices` metadata and an extra top-level EEF payload key do
not enable EEF serving.

`state_mask` and `action_mask` are independent float32 40-vectors. Inactive
values are zeroed again after normalization. Training loss and inference
sampling use `action_mask`, so a wrong mapping or mask changes both learning and
deployment behavior.

### Relative actions and inverse mapping

Unified arm and EEF actions are trained relative to the current state:

- arm joint: `action - state`;
- EEF position: body-frame position delta;
- EEF orientation: relative rotation, represented as rotation 6D;
- gripper, waist, and chassis velocity remain absolute.

The forward order is:

```text
native values -> scatter to 40D -> make arm/EEF actions relative
              -> normalize -> zero inactive slots
```

Data-level restoration performs the exact inverse:

```text
model output -> unnormalize -> restore absolute arm/EEF action
             -> convert EEF rot6d to xyz + quat_xyzw -> gather active values
```

`restore_action` therefore needs the absolute scattered state returned by
`encode_payload` as `state_abs`, not the normalized model input `state`.
Public v1 serving exercises only the joint-control branch of this inverse.

## Component and unified routes

A component route keeps the declared modality order and dimension. Use it when
the checkpoint was trained with that native component contract.

A unified route subclasses `_UnifiedMixin`, always exposes 40D state/action
tensors, and uses a stable registry key:

```python
@dataclasses.dataclass(frozen=True)
class MyRobotUnified(_UnifiedMixin, MyRobot):
    robot_name: ClassVar[str] = "my_robot_unified"
    _unified_registry_key: ClassVar[str] = "my_robot_v1"
```

The registry key is serialized into checkpoints. Do not rename it or change its
column meaning after training.

## Add a new embodiment

Use the [adapter template guide](_template/README.md) for the complete
file-by-file procedure. The required chain is:

```text
dataset + SDK contract
  -> layout.py and optional unified registry entry
  -> adapter exports and robot-class registration
  -> matching data.py + train.yaml + normalization statistics
  -> deploy_io.py
  -> dataset/SDK parity and action-order validation
```

Joint-only routes stop at joint mapping; do not add FK. Native EEF routes must
declare or extract the EEF values explicitly. Every serialized `robot_name` and
registry key is a checkpoint identifier and must remain stable.

The training/data pipeline's generic EEF path assumes a dual-arm 18D EEF block.
Single-arm joint routes may use the left arm/gripper groups, but generic
single-arm EEF masking is unsupported. Mixed EEF/joint presence is a training
feature. Public v1 deployment is joint-control only; adding adapter-specific EEF
masking or restoration does not make an EEF route a supported server route.

## Hardware action order

Unified data-level restoration gathers active semantic slices in this order:

```text
left_eef, right_eef, left_gripper, right_gripper,
waist, chassis_velocity, left_arm, right_arm
```

For public v1 joint serving, the active joint-control subset is not necessarily
in SDK order. `build_sdk_action_perm` uses the
registry's `action_groups` to construct and validate the permutation. A missing
semantic group or a hole in the native vector must remain a hard error. If the
SDK expects a wider vector containing uncontrolled columns, implement an
explicit fill/preserve policy.

See the [deployment guide](../../../deploy/README.md#action-restoration-and-sdk-order)
for the concrete G1/A2D order and hardware checks.
