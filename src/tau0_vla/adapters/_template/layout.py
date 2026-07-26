"""Layer 1 — your robot's native data layout.

Copy this file, rename the classes, and replace every ``<YOUR_...>``. The
worked example next door (``../g1/layout.py``) is heavily commented; read it
alongside this skeleton when a field is unclear.

Two things a new embodiment declares, and nothing else:

1. ``TemplateObservation`` — how to read your robot's live SDK payload. Only the
   deploy path uses this; training reads the dataset directly.
2. ``TemplateRobot.repack`` — how your dataset's columns map onto the pipeline's
   semantic names.

Everything downstream — normalization, chunking, assembling the 40D Unified
Layout, inverting all of it at inference — is the general pipeline's job and
needs nothing further from you.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Mapping

import numpy as np

from tau0_vla.data.robots.base import RobotConfig


@dataclasses.dataclass(frozen=True)
class TemplateObservation:
    """One live observation, parsed out of your robot's SDK payload.

    Used only by ``deploy_io.py``. Training never constructs this — it reads the
    dataset through ``repack`` below, so the two paths can disagree and this is
    the file where you make them line up.

    Add or drop fields to match your robot. The G1's version
    (``../g1/layout.py``) carries waist and head channels as well, plus derived
    properties that reorder joints into the layout its SDK expects.
    """

    instruction: str
    # One entry per camera. The names here are yours, but they must be the names
    # ``deploy_io._SDK_IMAGE_ACCESSORS`` maps to your training config's
    # ``camera_keys`` — see the comments there.
    head_image: Any
    wrist_image: Any
    arm_joint_states: np.ndarray
    gripper_states: np.ndarray

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "TemplateObservation":
        """Read the keys your SDK actually sends.

        Replace every key string below. Missing keys raise ``KeyError`` per
        request, which is what you want — a silently zero-filled observation
        produces confident nonsense.

        The instruction lookup is a precedence chain: the G1 tries
        ``sub_task_instruction`` then ``instruction`` then ``task``, so one
        server can accept payloads from clients that disagree. Keep whichever
        keys your clients send.
        """
        return cls(
            instruction=str(payload.get("<YOUR_INSTRUCTION_KEY>") or payload.get("task") or ""),
            head_image=payload["<YOUR_SDK_HEAD_IMAGE_KEY>"],
            wrist_image=payload["<YOUR_SDK_WRIST_IMAGE_KEY>"],
            arm_joint_states=np.asarray(payload["<YOUR_SDK_ARM_JOINT_KEY>"], dtype=np.float32),
            gripper_states=np.asarray(payload["<YOUR_SDK_GRIPPER_KEY>"], dtype=np.float32),
        )


@dataclasses.dataclass(frozen=True)
class TemplateRobot(RobotConfig):
    """Your robot's Robot Config. This class is the whole of layer 1.

    Registered by name and shared between training and deployment: a checkpoint
    records ``robot_name`` in its data spec, and inference rebuilds this class
    from it. That is why the name has to be stable.
    """

    # Change this. Pick it once — it is written into every checkpoint trained
    # with this config, and renaming it later orphans them.
    robot_name: ClassVar[str] = "<YOUR_ROBOT_NAME>"

    # Change this if your dataset stores end-effector orientation as quaternions
    # in an order other than xyzw. Ignored unless something in ``repack``
    # declares an ``eef_pose``, so a joint-controlled robot can leave it.
    quaternion_order: ClassVar[str] = "xyzw"

    # ── The one thing you must rewrite ────────────────────────────────────────
    #
    # Read this as: {what the pipeline calls it: what your dataset calls it}.
    #
    # KEYS on the left are a fixed vocabulary — the ``component_key`` values
    # declared in ``tau0_vla/data/modalities/``: ``arm_joint``, ``gripper``,
    # ``eef_pose``, ``waist``, ``chassis_velocity``. Do not invent new ones; a
    # key with no matching modality can never be requested.
    #
    # VALUES on the right are your dataset's own feature and field names. Every
    # one of them is a placeholder here.
    #
    # This is a capability list, not a checklist: it declares what your robot
    # COULD expose. Which entries are actually read is decided by the
    # ``state=[...]`` / ``action=[...]`` components a concrete config asks for
    # (see ``configs/_template/data.py``), so listing a field you do not
    # use is harmless. Omit whatever your robot does not have — the 40D Unified
    # Layout masks off everything you leave out.
    repack: ClassVar[dict[str, object]] = {
        # Where the language instruction lives in the dataset. On LeRobot
        # datasets this is usually "task".
        "prompt": "task",
        # Keys must match your training YAML's ``camera_keys``; values are the
        # video feature names in your dataset's ``meta/info.json``. The two
        # routinely differ — translating between them is what this is for.
        "images": {
            "<YOUR_CAMERA_NAME>": "<YOUR_DATASET_VIDEO_FEATURE>",
        },
        "state": {
            # The dataset feature holding the flat state vector.
            "raw": "<YOUR_DATASET_STATE_FEATURE>",
            # Which slices of that vector mean what. A list means "concatenate
            # these, in this order" — the order fixes the column order the model
            # sees and that ``restore_action`` returns.
            "semantic": {
                "arm_joint": "<YOUR_STATE_ARM_JOINT_FIELD>",
                "gripper": "<YOUR_STATE_GRIPPER_FIELD>",
            },
        },
        # Same shape, over your action feature. It does not have to mirror the
        # state — the G1's actions carry a chassis velocity its state does not.
        "action": {
            "raw": "<YOUR_DATASET_ACTION_FEATURE>",
            "semantic": {
                "arm_joint": "<YOUR_ACTION_ARM_JOINT_FIELD>",
                "gripper": "<YOUR_ACTION_GRIPPER_FIELD>",
            },
        },
    }

    @classmethod
    def supports_sdk_payload(cls) -> bool:
        """True if ``deploy_io.py`` can build a payload adapter for this robot.

        Leave this as True once you have filled in ``TemplateObservation``.
        """
        return True

    @classmethod
    def observation_from_payload(cls, payload: Mapping[str, Any]) -> TemplateObservation:
        return TemplateObservation.from_payload(payload)
