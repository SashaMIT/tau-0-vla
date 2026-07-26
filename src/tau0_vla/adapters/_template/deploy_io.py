"""Layer 3 — your robot's deploy I/O.

Everything the serving path needs to know about *this robot's wire format*:
which keys your SDK sends, what your cameras are called, how your state channels
line up with the checkpoint's ``field_descriptions.json``, and — the dangerous
one — what column order your SDK expects actions back in.

``deploy/server.py`` imports six names from here. Keep all six, with these
signatures; the HTTP layer calls them and nothing else:

    load_state_field_descriptions(artifacts_dir)  -> dict
    state_dim_from_field_descriptions(state_fd)   -> int
    build_payload_adapter(cam_keys, state_fd, state_dim) -> callable
    build_sdk_action_perm(data_spec, slices)      -> list[int] | None
    apply_sdk_action_perm(actions, perm)          -> np.ndarray
    canonicalize_action_dict(split)               -> dict

**Nothing outside this directory has to change.** ``deploy/server.py`` reads the
robot name off the checkpoint's Data Spec and imports the matching adapter's
``deploy_io``, so registering your robot in
``tau0_vla/data/robots/__init__.py`` is all it takes. (``--adapter`` exists to
name one explicitly, for a config registered outside the adapters package.)

Only the first three functions are genuinely yours. For the last three, copy
``../g1/deploy_io.py`` verbatim unless your SDK's action layout differs from your
registry's ``action_groups`` order — and read
``build_sdk_action_perm``'s docstring there before deciding that it does not.
Getting that one wrong writes gripper commands into arm joints, silently, on real
hardware.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tau0_vla.adapters._template.layout import TemplateObservation

# ──────────────────────────────────────────────────────────────────────────────
# SDK payload -> canonical policy payload
# ──────────────────────────────────────────────────────────────────────────────

# Change this. Where each of your SDK's state channels lands in the checkpoint's
# declared state vector. Keys are ``field_descriptions.json`` entries — the same
# field names your ``repack["state"]["semantic"]`` values use. Values pull the
# matching slice off the parsed observation.
#
# Note the G1 has to split its gripper: the SDK sends one 2-vector while the two
# effectors are declared as separate fields. Whether you need that depends on how
# you declared them.
_STATE_CHANNELS: dict[str, Callable[[TemplateObservation], Any]] = {
    "<YOUR_STATE_ARM_JOINT_FIELD>": lambda obs: obs.arm_joint_states,
    "<YOUR_STATE_GRIPPER_FIELD>": lambda obs: obs.gripper_states,
}

# Change this. Keys are the camera names your training config lists under
# ``camera_keys``; values read the image off the parsed observation. This is the
# join between what the config calls a camera and what your SDK calls it.
_SDK_IMAGE_ACCESSORS: dict[str, Callable[[TemplateObservation], Any]] = {
    "<YOUR_CAMERA_NAME>": lambda obs: obs.head_image,
}

# Change this. Fallback for cam_keys the observation does not parse: map the
# config's camera name onto the raw payload key, read straight out of the request
# body. A camera missing from both this and ``_SDK_IMAGE_ACCESSORS`` raises a
# KeyError per request rather than silently serving a black frame — keep that.
_RAW_IMAGE_ALIAS: dict[str, str] = {
    "<YOUR_CAMERA_NAME>": "<YOUR_SDK_HEAD_IMAGE_KEY>",
}


def load_state_field_descriptions(artifacts_dir: str | Path) -> dict[str, Any]:
    """Read the ``state`` block of a checkpoint's ``field_descriptions.json``.

    Generic — copy as-is.
    """
    path = Path(artifacts_dir) / "field_descriptions.json"
    return json.loads(path.read_text())["state"]


def state_dim_from_field_descriptions(state_fd: Mapping[str, Any]) -> int:
    """Width of the raw state vector = highest declared index + 1.

    Generic — copy as-is.
    """
    return max((max(e.get("indices") or [-1]) for e in state_fd.values()), default=-1) + 1


def build_payload_adapter(
    *,
    cam_keys: Sequence[str],
    state_fd: Mapping[str, Any],
    state_dim: int,
) -> Callable[[Mapping[str, Any]], dict]:
    """Build the SDK-payload → ``{prompt, images, state, meta}`` adapter.

    Bound once per process to the route's camera list and the checkpoint's state
    field descriptions; the returned callable runs per request. The body is
    generic once ``_STATE_CHANNELS`` and the two image maps above are yours.
    """
    cam_keys = tuple(cam_keys)

    def adapt(raw: Mapping[str, Any]) -> dict:
        obs = TemplateObservation.from_payload(raw)
        # Scatter each SDK channel into the slice the checkpoint declared for it.
        # Channels the checkpoint does not declare are skipped, so a robot with
        # extra sensors can send them without breaking an older checkpoint.
        state = np.zeros(state_dim, dtype=np.float32)
        for key, accessor in _STATE_CHANNELS.items():
            ix = state_fd.get(key, {}).get("indices") or []
            if ix:
                arr = np.asarray(accessor(obs), dtype=np.float32).reshape(-1)
                state[ix[0] : ix[-1] + 1] = arr[: len(ix)]
        images = {}
        for cam in cam_keys:
            accessor = _SDK_IMAGE_ACCESSORS.get(cam)
            if accessor is not None:
                images[cam] = accessor(obs)
                continue
            images[cam] = raw[_RAW_IMAGE_ALIAS.get(cam, cam)]
        return {
            "prompt": obs.instruction,
            "images": images,
            "state": state,
            "meta": {},
        }

    return adapt


# ──────────────────────────────────────────────────────────────────────────────
# SDK-native action column order — the dangerous part
# ──────────────────────────────────────────────────────────────────────────────


def canonicalize_action_dict(split: dict) -> dict:
    """Merge per-side unified slices into the keys your client reads.

    The G1's client wants ``arm_joint`` / ``gripper`` / ``eef_pose`` combined,
    while unified routes emit ``left_arm`` / ``right_arm`` / ... . If your client
    reads per-side keys, this is the identity function and you can return
    ``split`` unchanged — which is what this skeleton does.

    **If you copy the G1's merging version, copy its
    ``UNIFIED_SIDE_TO_CANONICAL`` table with it.** The merge is a two-way wire
    contract: ``deploy/openloop_with_server.py`` reads that table off your module
    to split each canonical key back into its two sides. Merging without
    publishing the table makes every server-backed eval fail with
    ``response missing 'left_gripper' slice``.
    """
    return dict(split)


def build_sdk_action_perm(data_spec, slices) -> list[int] | None:
    """Column permutation mapping ``restore_action``'s output to your SDK's order.

    **Read ``../g1/deploy_io.py``'s version of this function before writing
    yours.** Returning ``None`` means "no reordering needed", which is correct
    only when your SDK applies action columns in exactly the order
    ``restore_action`` emits them. For unified-40D routes it usually is not: the
    pipeline emits grippers first, while an SDK typically wants arms first.

    Getting this wrong writes gripper commands into the first arm joints. Nothing
    raises. On hardware that is the worst failure mode in this codebase, which is
    why the G1's version raises in two cases rather than degrading — an EEF slot
    with no ``action_groups`` entry, and a permuted vector with holes. Keep both
    checks when you copy it.

    Returning ``None`` here is safe only for component routes, which already emit
    declared-component order.
    """
    return None


def apply_sdk_action_perm(actions, sdk_action_perm: Sequence[int] | None) -> np.ndarray:
    """Reorder a ``[chunk, native_dim]`` action array into SDK column order.

    Generic — copy as-is. Identity when the permutation is ``None``.
    """
    arr = np.asarray(actions, dtype=np.float32)
    if sdk_action_perm is not None:
        arr = arr[..., sdk_action_perm]
    return arr


__all__ = [
    "apply_sdk_action_perm",
    "build_payload_adapter",
    "build_sdk_action_perm",
    "canonicalize_action_dict",
    "load_state_field_descriptions",
    "state_dim_from_field_descriptions",
]
