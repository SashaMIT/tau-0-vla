"""G1 native data layout and public joint-control adapters.

The public release intentionally contains no G01 URDF. Datasets that provide
joint state/action values therefore stay in joint space. Native end-effector
datasets can still use the generic EEF-column path in ``_UnifiedMixin``; this
adapter never derives EEF poses from joints.

The AgiBot World layout is fixed by ``assets/dim_registry.json``:

* state arm joints: ``observation.state[28:42]``
* action arm joints: ``action[16:30]``
* state/action grippers: flat indices ``[0, 1]``
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Mapping

import numpy as np

from tau0_vla.data.robots.base import RobotConfig
from tau0_vla.data.robots.unified import _UnifiedMixin
from tau0_vla.data.stats import NormStats, normalize_array


@dataclasses.dataclass(frozen=True)
class G1Observation:
    instruction: str
    head_image: Any
    right_hand_image: Any
    left_hand_image: Any
    arm_joint_states: np.ndarray
    gripper_states: np.ndarray
    waist_joint_states: np.ndarray
    head_joint_states: np.ndarray

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "G1Observation":
        return cls(
            instruction=str(
                payload.get("sub_task_instruction") or payload.get("instruction") or payload.get("task") or ""
            ),
            head_image=payload["observation.images.head"],
            right_hand_image=payload["observation.images.hand_right"],
            left_hand_image=payload["observation.images.hand_left"],
            arm_joint_states=np.asarray(payload["observation.states.arm_joint_states"], dtype=np.float32),
            gripper_states=np.asarray(payload["observation.states.gripper_states"], dtype=np.float32),
            waist_joint_states=np.asarray(payload["observation.states.waist_joint_states"], dtype=np.float32),
            head_joint_states=np.asarray(payload["observation.states.head_joint_states"], dtype=np.float32),
        )

    @property
    def joint_state(self) -> np.ndarray:
        """Return ``[L_arm7, L_gripper, R_arm7, R_gripper]``."""
        joints = np.asarray(self.arm_joint_states, dtype=np.float32).reshape(-1)
        grippers = np.asarray(self.gripper_states, dtype=np.float32).reshape(-1)
        if joints.shape[0] != 14 or grippers.shape[0] != 2:
            raise ValueError(
                f"G1 joint/gripper state must have shapes 14 and 2, got {joints.shape[0]} and {grippers.shape[0]}"
            )
        return np.concatenate([joints[:7], grippers[:1], joints[7:], grippers[1:]], axis=0).astype(np.float32)

    @property
    def waist_pitch(self) -> float:
        return float(np.asarray(self.waist_joint_states, dtype=np.float32).reshape(-1)[0])

    @property
    def waist_lift(self) -> float:
        waist = np.asarray(self.waist_joint_states, dtype=np.float32).reshape(-1)
        return float(waist[1]) if waist.shape[0] >= 2 else 0.0

    @property
    def head_joint_radians(self) -> np.ndarray:
        return np.asarray(self.head_joint_states, dtype=np.float32).reshape(-1)


@dataclasses.dataclass(frozen=True)
class G1RobotConfig(RobotConfig):
    """Common G1 SDK adapter.

    Public G1 routes are joint-controlled. ``is_eef=True`` is rejected instead
    of silently depending on an unavailable URDF.
    """

    robot_name = "g1"

    @classmethod
    def supports_sdk_payload(cls) -> bool:
        return True

    @classmethod
    def observation_from_payload(cls, payload: Mapping[str, Any]) -> G1Observation:
        return G1Observation.from_payload(payload)

    @classmethod
    def build_sdk_state_with_contract(
        cls,
        payload: Mapping[str, Any],
        *,
        norm_stats: Mapping[str, Any],
        state_key: str,
        use_quantile: bool,
        is_eef: bool,
        mode: str,
    ) -> np.ndarray:
        del mode
        if is_eef:
            raise NotImplementedError(
                "The public G1 adapter has no URDF-backed joint-to-EEF conversion. "
                "Public v1 serving supports joint-control checkpoints only; native "
                "EEF fields remain available for training and data workflows."
            )
        state = cls.observation_from_payload(payload).joint_state
        if state_key not in norm_stats:
            return state.astype(np.float32)
        raw_stats = norm_stats[state_key]
        if isinstance(raw_stats, dict):
            stats_obj = NormStats(
                mean=np.asarray(raw_stats["mean"], dtype=np.float32),
                std=np.asarray(raw_stats["std"], dtype=np.float32),
                q01=None if raw_stats.get("q01") is None else np.asarray(raw_stats["q01"], dtype=np.float32),
                q99=None if raw_stats.get("q99") is None else np.asarray(raw_stats["q99"], dtype=np.float32),
            )
        else:
            stats_obj = raw_stats
        return normalize_array(
            state.reshape(1, -1),
            stats_obj,
            use_quantiles=use_quantile,
        ).reshape(-1)

    @classmethod
    def native_dim_labels(cls, *, is_eef: bool, native_dim: int) -> list[str]:
        if is_eef:
            return [f"eef_dim_{idx}" for idx in range(native_dim)]
        labels = [f"L_j{i}" for i in range(7)] + ["L_grip"] + [f"R_j{i}" for i in range(7)] + ["R_grip"]
        return labels[:native_dim]


@dataclasses.dataclass(frozen=True)
class G1Agibot(G1RobotConfig):
    """G1 data layout used by the bundled AgiBot World example."""

    robot_name: ClassVar[str] = "g1_agibot"
    repack: ClassVar[dict[str, object]] = {
        "prompt": "task",
        "images": {
            "head": "observation.images.top_head",
            "wrist_left": "observation.images.hand_left",
            "wrist_right": "observation.images.hand_right",
        },
        "state": {
            "raw": "observation.state",
            "semantic": {
                "arm_joint": "state/joint/position",
                "gripper": ["state/left_effector/position", "state/right_effector/position"],
                "waist": "state/waist/position",
            },
        },
        "action": {
            "raw": "action",
            "semantic": {
                "arm_joint": "action/joint/position",
                "gripper": ["action/left_effector/position", "action/right_effector/position"],
                "waist": "action/waist/position",
                "chassis_velocity": "action/robot/velocity",
            },
        },
    }


@dataclasses.dataclass(frozen=True)
class G1AgibotUnified(_UnifiedMixin, G1Agibot):
    """AgiBot World G1: joint + gripper control in the unified 40D layout."""

    robot_name: ClassVar[str] = "g1_agibot_unified"
    _unified_registry_key: ClassVar[str] = "g1_agibot_36"


@dataclasses.dataclass(frozen=True)
class G1DaasUnified(_UnifiedMixin, G1Agibot):
    """DAAS G1: joints, grippers, waist and chassis in unified 40D."""

    robot_name: ClassVar[str] = "g1_daas_unified"
    _unified_registry_key: ClassVar[str] = "g1_daas_36"


@dataclasses.dataclass(frozen=True)
class G1A2dJointUnified(_UnifiedMixin, G1Agibot):
    """Classic A2D state-93/action-41 joint layout."""

    robot_name: ClassVar[str] = "g1_a2d_joint_unified"
    _unified_registry_key: ClassVar[str] = "a2d_41_joint"


G1_UNIFIED_CLASSES: dict[str, type] = {
    "g1_agibot_unified": G1AgibotUnified,
    "g1_daas_unified": G1DaasUnified,
    "g1_a2d_joint_unified": G1A2dJointUnified,
}


__all__ = [
    "G1A2dJointUnified",
    "G1Agibot",
    "G1AgibotUnified",
    "G1DaasUnified",
    "G1Observation",
    "G1RobotConfig",
    "G1_UNIFIED_CLASSES",
]
