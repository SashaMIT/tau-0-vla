# Adapter template

Copy the template, then replace every `YOUR_...` placeholder:

```bash
cp -r src/tau0_vla/adapters/_template \
      src/tau0_vla/adapters/my_robot
grep -RIn 'YOUR_' src/tau0_vla/adapters/my_robot
grep -RIn 'adapters._template' src/tau0_vla/adapters/my_robot
```

| file | responsibility |
|---|---|
| `layout.py` | dataset layout, live observation parser, camera aliases, and `RobotConfig` |
| `deploy_io.py` | SDK state/image input and exact action output contract |

The template starts as a component-style joint-control adapter. Add unified or
native-EEF behavior only when the dataset and checkpoint require it.

## 1. Define the layout

In `layout.py`:

1. Rename `TemplateObservation` and implement `from_payload()` for the live SDK
   request. Validate vector lengths instead of silently truncating them.
2. Rename `TemplateRobot` and give it a stable `robot_name`.
3. Set `quaternion_order` when a component EEF route does not use the default
   `xyzw`.
4. Replace `repack` with the dataset's real feature names.

`repack` has two layers:

- `raw` points to the flat feature used to construct a sample;
- `semantic` maps modality names to fields declared under the raw feature's
  `field_descriptions` in `meta/info.json`.

Minimal joint example:

```python
repack = {
    "prompt": "task",
    "images": {"head": "observation.images.head"},
    "state": {
        "raw": "observation.state",
        "semantic": {
            "arm_joint": "state/joint/position",
            "gripper": "state/gripper/position",
        },
    },
    "action": {
        "raw": "action",
        "semantic": {
            "arm_joint": "action/joint/position",
            "gripper": "action/gripper/position",
        },
    },
}
```

Use only the semantic names understood by the pipeline:

| name | meaning |
|---|---|
| `arm_joint` | arm joint positions/actions |
| `gripper` | one or two gripper values |
| `eef_pose` | native end-effector pose |
| `waist` | up to two waist values |
| `chassis_velocity` | up to two base-velocity values |

Camera names in `repack["images"]` become the model-facing names used by
`Image(...)`, `data_args.camera_keys`, and deployment.

## 2. Add a unified route

For unified 40D training, add a subclass:

```python
import dataclasses
from typing import ClassVar

from tau0_vla.data.robots.unified import _UnifiedMixin


@dataclasses.dataclass(frozen=True)
class MyRobotUnified(_UnifiedMixin, MyRobot):
    robot_name: ClassVar[str] = "my_robot_unified"
    _unified_registry_key: ClassVar[str] = "my_robot_v1"
```

Then add `my_robot_v1` to
[`dim_registry.json`](../../data/assets/dim_registry.json):

```json
{
  "my_robot_v1": {
    "state_dim": 16,
    "action_dim": 16,
    "state_groups": {
      "left_arm": [0, 1, 2, 3, 4, 5, 6],
      "right_arm": [7, 8, 9, 10, 11, 12, 13],
      "left_gripper": [14],
      "right_gripper": [15]
    },
    "action_groups": {
      "left_arm": [0, 1, 2, 3, 4, 5, 6],
      "right_arm": [7, 8, 9, 10, 11, 12, 13],
      "left_gripper": [14],
      "right_gripper": [15]
    }
  }
}
```

These arrays are native flat-vector indices. They scatter to fixed unified
slots; see the [40D contract](../README.md#unified-40d-contract). State and
action arrays must be written independently even when they happen to match.
Check every index for bounds, duplicates, left/right ordering, and expected
group width. The generic path only consumes the group names listed in the 40D
contract.

For a unified route, `repack.raw` selects the flat vector and the registry—not
`repack.semantic` or component order—drives the 40D scatter.
`field_descriptions` and `repack.semantic` still define the component/checkpoint
contract used to reconstruct that flat state from an SDK payload.

The key becomes part of the checkpoint contract. Keep both its name and column
meaning stable.

### Native EEF data

Do not derive EEF values from joints. If the dataset has native EEF columns,
declare them on the unified subclass:

```python
_eef_state_col = "observation.eef_pose"
_eef_action_col = "action.eef_pose"
_eef_format = "quat"  # or "euler"
_eef_present_col = "observation.eef_present"  # optional
```

`quat` expects 7 values per arm (`xyz + quat_xyzw`); `euler` expects 6
(`xyz + XYZ extrinsic Euler`, radians). A supported generic dual-arm value
concatenates left then right: 14D quaternion or 12D Euler. The EEF action column
automatically receives the same horizon offsets as the raw action column; EEF
state remains the current frame.

Separate EEF parquet columns work for training. If EEF is embedded in the flat
vectors, an `_eef_provider()` can expose it to the unified data assembler and
local evaluation:

```python
def _eef_provider(self):
    def provide(sample):
        out = dict(sample)
        out["_eef_state_raw"] = np.asarray(out["_state_raw"])[..., STATE_EEF_INDICES]
        out["_eef_action_raw"] = np.asarray(out["_action_raw"])[..., ACTION_EEF_INDICES]
        return out

    return provide
```

This does not create a public serving contract. Public v1 serving routes must
remain joint-controlled, and the server rejects EEF action slices. Adding a
top-level EEF payload key alone has no effect. `eef_inline_indices` in the
registry is descriptive metadata and is not read automatically.

`_eef_present_col` is an optional per-sample validity flag. Without it, finite
EEF values determine presence. `_is_single_arm` is currently metadata only:
the generic assembler still reserves the dual-arm EEF block, so a single-arm
EEF route needs explicit mask/restoration logic in the data pipeline.

## 3. Define deployment I/O

The server imports these names from `deploy_io.py`:

- `load_state_field_descriptions`
- `state_dim_from_field_descriptions`
- `build_payload_adapter`
- `build_sdk_action_perm`
- `apply_sdk_action_perm`
- `canonicalize_action_dict`

Replace:

- `_STATE_CHANNELS` with SDK state values keyed by checkpoint field name;
- `_SDK_IMAGE_ACCESSORS` and `_RAW_IMAGE_ALIAS` with every configured camera;
- `canonicalize_action_dict` with the keys expected by the `/act` client;
- `build_sdk_action_perm` with the flat output order expected by the SDK.

If `canonicalize_action_dict` merges left/right unified slices, also export its
`UNIFIED_SIDE_TO_CANONICAL` table so `openloop_with_server.py` can split those
keys consistently.

The template returns `None` because it assumes restored component order already
equals SDK order. Verify that assumption even for component routes. For a
unified route it is usually false: copy and adapt the checked implementation in
[`g1/deploy_io.py`](../g1/deploy_io.py). Missing slots and holes must fail
before an action reaches hardware.

The public v1 template does not define an EEF SDK output. Keep new serving
routes joint-controlled.

## 4. Export and register

Outside this directory:

1. Update the copied `_template` imports in `__init__.py` and `deploy_io.py`.
   Export every public class and a mapping keyed by `robot_name`:

   ```python
   MY_ROBOT_UNIFIED_CLASSES = {
       "my_robot_unified": MyRobotUnified,
   }
   ```

2. Add the adapter package to `_ADAPTER_MODULES` in
   [`data/robots/__init__.py`](../../data/robots/__init__.py).
3. Add every `robot_name` and class path to `_ROBOT_CLASS_PATHS` in the same
   file. This path is used when reloading a checkpoint.
4. Create a matching `configs/<task>/data.py` and `train.yaml`.

Keep the three identifiers distinct:

| identifier | used by |
|---|---|
| `robot_name` | `_ROBOT_CLASS_PATHS`, `*_UNIFIED_CLASSES`, Data Spec, and stats `--body` |
| `_unified_registry_key` | `dim_registry.json` and `norm_stats.json -> per_embodiment` |
| `@register_config` name | training YAML `data_args.config_name` and checkpoint route |

## 5. Validate the contract

Before training, inspect a real sample and check:

- instruction text and all camera views;
- native state/action widths and each registry index;
- every active action arm/EEF dimension has a same-unit, same-order current
  state reference;
- unified values, `state_mask`, and `action_mask`;
- EEF-present versus joint-fallback behavior for EEF training data;
- arm/EEF relative actions and gripper/waist/chassis absolute actions;
- normalization followed by `restore_action`.

For a public v1 joint-control serving route, also verify `/act` semantic keys,
the flat SDK action permutation when that endpoint is applicable, and exact
dataset/SDK state parity.

The registry entry, adapter class, config name, normalization statistics, and
deployment order together form one checkpoint contract.
