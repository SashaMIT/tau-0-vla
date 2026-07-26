# Adapters — supporting your own robot

Everything this codebase knows about a specific robot lives in one directory
under here. The general pipeline in `tau0_vla/data/` stays untouched: it may not
name an embodiment, carry a raw column index, or know an SDK payload key.

Two ways to start:

- **`_template/`** — a skeleton with every value replaced by `<YOUR_...>`. Copy
  it, then `grep -rn 'YOUR_'` for your checklist.
- **`g1/`** — the worked example used against the bundled real data.

## The layers

| layer | file | owns | needed? |
|---|---|---|---|
| 1. data layout | `layout.py` | which native columns mean what, the Robot Config class | always |
| 3. deploy I/O | `deploy_io.py` | SDK payload keys, camera aliases, action column order | to serve or evaluate |

A joint-controlled robot needs `layout.py` and `deploy_io.py`. A robot dataset
with native end-effector fields can declare those fields directly; the public
release does not derive EEF poses from joints and ships no G01 URDF.

`deploy_io` is deliberately not re-exported from the package `__init__`: it is the
only layer needing a live Data Spec and a checkpoint directory, and importing it
from the training path would drag deployment concerns into the dataloader. Import
it explicitly.

## The 40D Unified Layout

Every robot's state and action are assembled into one 40-dimensional vector where
each slot has a fixed meaning. Your registry entry says which of *your* native
columns fill which slots; unfilled slots are zeroed and masked off. This is what
lets one checkpoint serve several embodiments.

```
[0:3]    left_eef_position (xyz, m)      [18:19]  left_gripper
[3:9]    left_eef_orientation (rot6d)    [19:20]  right_gripper
[9:12]   right_eef_position              [20:22]  waist
[12:18]  right_eef_orientation           [22:24]  chassis_velocity
                                         [24:32]  left_arm_joints  (max 8)
                                         [32:40]  right_arm_joints (max 8)
```

An arm with fewer than 8 joints fills the low slots and masks the rest. The
authoritative version is `UNIFIED_LAYOUT` in `tau0_vla/data/robots/unified.py`.

**EEF priority.** When a robot supplies native end-effector poses,
the arm-joint slots are masked off in both state and action. EEF slots are
Cartesian and comparable across robots; joint slots are not, since a 6-DoF and a
7-DoF arm occupy the same slots meaning different things. Keeping both would
double-represent the same motion.

You only need a registry entry if you want that cross-embodiment sharing. A robot
training on its own can use a component route — `G1Agibot` in `g1/layout.py` is
one — and never touch the registry.

### The registry entry

`tau0_vla/data/assets/dim_registry.json`, one row per embodiment:

```json
"my_robot_20": {
  "source": "mylab",
  "robot_type": "my_robot",
  "action_dim": 20,
  "state_dim": 20,
  "eef_dim": 0,
  "action_groups": {
    "left_arm":      [0, 1, 2, 3, 4, 5, 6],
    "right_arm":     [7, 8, 9, 10, 11, 12, 13],
    "left_gripper":  [14],
    "right_gripper": [15]
  },
  "state_groups": { ... },
  "gripper_type": "open_m",
  "is_single_arm": false
}
```

Each group lists the **native column indices** filling one Unified Layout slot.
`state_groups` and `action_groups` are separate because the two vectors often
disagree — the G1's action puts grippers at columns 0–1 while its state puts them
at 14–15.

The file stays in the general pipeline even though its rows are
embodiment-specific: the loader and the `action_groups` schema are how the 40D
contract describes *any* robot, so adapters contribute rows to a shared table
rather than each shipping their own.

**Registry keys are frozen once you train with them.** They are recorded in every
checkpoint and in each dataset's `meta/info.json`; renaming one makes
`get_registry_entry` miss, which fails on a real robot rather than at import.

## Before you drive hardware

`restore_action` returns actions in the pipeline's own column order, which is
usually **not** the order your SDK applies them in — the pipeline emits grippers
first, most SDKs want arms first. Without a reordering step gripper commands land
in the first two arm joints. On real hardware that is catastrophic, and nothing
raises.

`build_sdk_action_perm` in `g1/deploy_io.py` builds that permutation from your
registry entry's `action_groups`. **Read its docstring before writing your own**
— it is the most safety-critical prose in this repository, it explains both of
the cases where it raises rather than degrading, and it is not repeated here so
that there is only one copy to keep correct.

The server logs the permutation at startup. Confirm that line before a run on
hardware.

## Checklist

1. Copy `_template/` to `adapters/<your_robot>/`, or copy `g1/` if you want a
   filled-in example to edit down.
2. Fill in `layout.py`: the Robot Config class, `robot_name`, and the column
   `repack`. Every `<YOUR_...>` is a placeholder you must replace.
3. Fill in `deploy_io.py`: SDK state channels, camera maps, action column order.
4. For native EEF data, declare the EEF columns in the adapter. Keep joint data
   in joint space; this release provides no URDF-backed conversion.
5. Register the robot name in `tau0_vla/data/robots/__init__.py`
   (`_ADAPTER_MODULES` and `_ROBOT_CLASS_PATHS`). Without this, training works and
   inference fails with `Unknown robot`. It is also what lets deploy find your
   `deploy_io` — nothing in `deploy/` needs editing.
6. Add a `dim_registry.json` entry, if you want cross-embodiment slot sharing.
7. Write a Robot Config in a `data.py` next to your training YAML — start from
   `configs/_template/` and follow `src/tau0_vla/data/DATASET_FORMAT.md`.
8. Confirm that embodiment names, raw column indices, and SDK payload keys stay
   inside the adapter rather than leaking into `tau0_vla/data/`.
9. Compute norm stats, then check the loader's startup summary: dataset size,
    retained-frame percentage, and the state/action mask widths should all match
    what you expect.
