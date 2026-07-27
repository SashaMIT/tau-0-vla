# Robot adapters

Adapters keep embodiment-specific data layouts and deployment I/O outside the
general data pipeline.

Start from:

- [`_template/`](_template/README.md) for a new robot
- [`g1/`](g1/) for the bundled AgiBot World example

Each adapter contains:

| file | purpose |
|---|---|
| `layout.py` | native state/action fields, camera mapping, and RobotConfig |
| `deploy_io.py` | SDK payload keys and output action order |

## Unified 40D layout

```text
[0:9]    left EEF pose          [18]     left gripper
[9:18]   right EEF pose         [19]     right gripper
[20:22]  waist                  [22:24]  chassis velocity
[24:32]  left arm joints        [32:40]  right arm joints
```

The registry entry in
[`data/assets/dim_registry.json`](../data/assets/dim_registry.json) maps native
columns into these slots. Unused slots are zero-padded and masked.

- Datasets with native EEF state/actions use the EEF slots.
- Datasets with joint state/actions stay in joint space.
- When EEF is active, arm-joint slots are masked to avoid representing the same
  motion twice.

The public G1 adapter is joint-controlled and does not require a URDF.

## Add a robot

1. Copy `_template/` to `adapters/<robot>/`.
2. Define the native layout and camera mapping in `layout.py`.
3. Define SDK input/output mapping in `deploy_io.py`.
4. Register the class in `data/robots/__init__.py`.
5. For unified training, add a `_UnifiedMixin` adapter variant and a matching
   `dim_registry.json` entry.
6. Create a matching `configs/<task>/data.py` and compute normalization stats.

Registry keys are stored in checkpoints, so keep them stable after training.

## Hardware action order

The unified output order may differ from the robot SDK order.
`build_sdk_action_perm` in the adapter maps semantic action slices back to the
SDK layout. Confirm the logged permutation before hardware deployment.
