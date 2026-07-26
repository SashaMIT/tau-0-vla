# Deploy

Thin layer over `tau0_vla.data` for serving / evaluating tau-0-vla checkpoints.

| File | Role |
|---|---|
| `policy.py` | `Tau0VLAPolicy.from_checkpoint` + `.infer(payload)` — the VLA deployment wrapper. |
| `server.py` | HTTP server wrapping `Tau0VLAPolicy`; A2D SDK compatible. |
| `openloop.py` | Open-loop eval: iterate a dataset, call `policy.infer`, report MSE / L1, optional plots. |
| `openloop_with_server.py` | The same eval against a running server instead of a local checkpoint — takes `--server-url` and `--config-module`, no `--ckpt`. Use it to confirm the HTTP wire format agrees with local inference. |
| `_bootstrap.py` | Fires `@register_config` for `configs/*/data.py` before the registry is queried, resolves which adapter's `deploy_io` serves a checkpoint, and retrofits `policy_manifest.json` for checkpoints predating it. |
| `warmup.py` | Shared warmup helpers behind `server.py`'s `--infer-mode` / `--max-prefix-len` / `--warmup-steps`. Not an entry point. |
| `check_parity.py` | Manual tool: verify the CUDA-graph inference path against eager. |
| `__init__.py` | Makes `deploy` a package so `python -m deploy.server` works from an installed copy. |

## Start the HTTP server

```bash
# single-route checkpoint
python -m deploy.server --model <ckpt>

# multi-route checkpoint — pick the leaf route
python -m deploy.server --model <ckpt> --route pick-red-block-joint-g1

# override host / port
python -m deploy.server --model <ckpt> --host 0.0.0.0 --port 10088
```

Endpoints: `POST /act` (pickle canonical payload → name-keyed action dict),
`POST /act_lerobot_bytes` (pickle A2D SDK dict → flat JSON action chunk), `GET /health`.

The embodiment-specific wire layer (`adapters/<robot>/deploy_io.py`) is resolved
from the checkpoint: the Data Spec records a robot name, and that name names the
adapter. Serving a different robot needs no edit here. `--adapter
<dotted.package>` overrides it, for a Robot Config registered outside
`tau0_vla.adapters` — and raises rather than falling back if that package has no
`deploy_io`, because silently serving the G1's column order would write gripper
commands into arm joints.

## Action & state contract

Both endpoints run the same `policy.infer` (encode → tokenize → forward →
`restore_action`); they differ only in how the action chunk is laid out on the
wire.

**Input** (built by `server.adapt` from the A2D SDK dict, or sent directly to
`/act`): `{prompt, images, state, meta}`.
- `state` — raw native vector in `field_descriptions` order: arm joints at
  `state/joint/position` (0–13 = `[Larm×7, Rarm×7]`), grippers at
  `state/{left,right}_effector/position` (14, 15). The unified scatter reads
  exactly these indices via the registry `state_groups`; extra channels
  (`end/*`, `head`, …) are ignored. Same units/joint-order as training.
- `images` — keyed by `cam_keys` (`head`, `wrist_right`, `wrist_left`); a
  missing camera raises (no silent degrade).

**Output** — the model controls arms + grippers only (16 active dims); it
predicts *relative* actions, so `restore_action` needs the current absolute
state (supplied automatically from `encode_payload`'s `state_abs`).

| Endpoint | Shape | Layout |
|---|---|---|
| `POST /act` | `{name: [[chunk×dim]]}` dict | canonical slot names (`left_arm` / `right_arm` / `left_gripper` / `right_gripper`, or `left_eef`/… for EEF routes). Read **by name** — order-independent. |
| `POST /act_lerobot_bytes` | flat `[chunk, 16]` list | **A2D SDK native order**: `[Larm×7, Rarm×7, gripL, gripR]` (registry `action_groups` indices 0–15), applied positionally by the SDK. |

> **Unified-40D remap (important).** `restore_action` emits the robot-agnostic
> `_UNIFIED_NATIVE_ORDER` (`[gripL, gripR, Larm×7, Rarm×7]` — grippers first).
> The A2D SDK applies `/act_lerobot_bytes` positionally in `action_groups` order
> (arms first) — the same layout the pre-unified *component* checkpoints emitted.
> `adapters.g1.deploy_io.build_sdk_action_perm` scatters each restored slice back to its
> `action_groups` index so the wire vector matches the SDK; component routes are
> already native and pass through unchanged. The remap is logged at startup
> (`/act_lerobot_bytes SDK-native remap …`); confirm it before driving hardware.
> `/act`'s dict is safe regardless (name-keyed) — the remap applies only to the
> flat SDK endpoint.

## Run open-loop evaluation

```bash
# with plots
python deploy/openloop.py --ckpt <ckpt> --out-dir <out> --max-inferences 100

# metrics only, no plots
python deploy/openloop.py --ckpt <ckpt> --no-plot

# multi-route: pick the route and/or swap eval dataset
python deploy/openloop.py --ckpt <ckpt> --route <leaf> --config <other-config> --no-plot
```
