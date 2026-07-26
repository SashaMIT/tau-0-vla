# Skeleton adapter

Copy this directory, rename it, replace every `<YOUR_...>`.

```bash
cp -r src/tau0_vla/adapters/_template src/tau0_vla/adapters/my_robot
grep -rn 'YOUR_' src/tau0_vla/adapters/my_robot     # your checklist
```

Nothing here runs until those placeholders are gone — that is on purpose. A
skeleton that appeared to work would invite you to leave a field unchanged and
train on the wrong column.

**`../g1/` is the worked example.** This skeleton is a starting shape.

## What you are filling in

| file | what it owns | when you are done |
|---|---|---|
| `layout.py` | your dataset's column layout (`repack`) and your SDK's payload keys (`TemplateObservation`) | training works |
| `deploy_io.py` | your SDK's state channels, camera names, and action column order | serving works |

Only a joint-controlled robot is covered — arm joints plus grippers. If your
dataset provides native end-effector fields, declare them in `repack` like any
other component. This release intentionally does not synthesize EEF from joints.

## Three things outside this directory

1. **Register the robot name**, in `src/tau0_vla/data/robots/__init__.py` — two
   entries, and nowhere else:

   ```python
   _ADAPTER_MODULES = ("tau0_vla.adapters.g1", "tau0_vla.adapters.my_robot")

   _ROBOT_CLASS_PATHS = {
       "g1_agibot": ("tau0_vla.adapters.g1", "G1Agibot"),
       "<YOUR_ROBOT_NAME>": ("tau0_vla.adapters.my_robot", "MyRobot"),
   }
   ```

   This is how a checkpoint rebuilds your class: a Data Spec records
   `robot_name`, and inference looks it up here. Skip it and training works while
   inference fails with `Unknown robot`. The imports there are lazy on purpose —
   your layout subclasses `RobotConfig` from the pipeline, so a top-level import
   would break whichever of the two loads first.

2. **A Robot Config that uses your class**, in a `data.py` next to your training
   YAML. Copy `configs/_template/`, then replace `G1A2dJointUnified` with your
   class. The same-directory placement matters — see the template's README.

3. **An entry in `tau0_vla/data/robots/__init__.py`** — `_ADAPTER_MODULES` plus
   your robot name in `_ROBOT_CLASS_PATHS`. That is also how `deploy/server.py`
   finds your `deploy_io`: it reads the robot name off the checkpoint's Data
   Spec and imports the matching adapter, so nothing under `deploy/` needs
   editing.

If your robot shares a checkpoint with others through the 40D Unified Layout, you
also need a `dim_registry.json` entry — see `../README.md`.

## Before you drive hardware

Read `build_sdk_action_perm` in `../g1/deploy_io.py`. It maps the pipeline's
action column order onto what your SDK expects, and the two are usually not the
same — the pipeline emits grippers first, most SDKs want arms first. Getting it
wrong writes gripper commands into arm joints and nothing raises.

The skeleton's version returns `None` (no reordering), which is correct only if
your SDK applies columns exactly as `restore_action` emits them. Confirm that
before trusting it.
