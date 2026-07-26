"""Unified cross-embodiment action/state layout.

Defines a fixed 40D vector layout where each position has permanent semantic
meaning across all robot types. Robots that don't populate a given slot
(e.g. no gripper, no EEF) zero-pad it and set the corresponding mask bit to 0.

Layout:
    [0:3]    left_eef_position (xyz)       meters
    [3:9]    left_eef_orientation           rot6d (from euler/quat)
    [9:12]   right_eef_position (xyz)       meters
    [12:18]  right_eef_orientation          rot6d
    [18:19]  left_gripper                   per-type normalized
    [19:20]  right_gripper                  per-type normalized
    [20:22]  waist                          per-type normalized
    [22:24]  chassis_velocity               per-type normalized
    [24:32]  left_arm_joints (max 8D)       rad
    [32:40]  right_arm_joints (max 8D)      rad
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Callable, ClassVar

import numpy as np

from tau0_vla.data.modalities.transforms import Euler2Rot6D, Quat2Rot6D, RelativeToState

# Matches the ``Control mode: <x>`` line of the structured tau-0-vla prompt
# (``You are controlling a robot.\nRobot type: ...\nControl mode: eef\n...``). Used
# to keep that label honest per-sample: a body whose EEF is present on only a
# subset of frames (e.g. ABC-130k Yam, ~46%) is statically templated ``eef`` but
# actually falls back to joint supervision on the EEF-absent frames. We rewrite the
# label from the *actual* per-sample action mask so ``Control mode`` always matches
# what is supervised. No-op when the prompt has no such line (other prompt styles).
_CONTROL_MODE_RE = re.compile(r"(?m)^(Control mode:[ \t]*).*$")


def _rewrite_control_mode(prompt: Any, mode: str) -> Any:
    """Return ``prompt`` with its ``Control mode:`` line set to ``mode`` (``eef`` /
    ``joint``). Leaves non-str prompts and prompts without the line untouched."""
    if not isinstance(prompt, str) or "Control mode:" not in prompt:
        return prompt
    return _CONTROL_MODE_RE.sub(rf"\g<1>{mode}", prompt, count=1)

# ─────────────────────────────────────────────────────────────────────────────
# Layout constants
# ─────────────────────────────────────────────────────────────────────────────

UNIFIED_DIM = 40

LEFT_EEF_POS = slice(0, 3)
LEFT_EEF_ORI = slice(3, 9)
RIGHT_EEF_POS = slice(9, 12)
RIGHT_EEF_ORI = slice(12, 18)
LEFT_GRIPPER = slice(18, 19)
RIGHT_GRIPPER = slice(19, 20)
WAIST = slice(20, 22)
CHASSIS_VELOCITY = slice(22, 24)
LEFT_ARM = slice(24, 32)
RIGHT_ARM = slice(32, 40)
RESERVED = slice(40, 40)

LEFT_EEF = slice(0, 9)
RIGHT_EEF = slice(9, 18)

UNIFIED_LAYOUT = {
    "left_eef_pos": LEFT_EEF_POS,
    "left_eef_ori": LEFT_EEF_ORI,
    "right_eef_pos": RIGHT_EEF_POS,
    "right_eef_ori": RIGHT_EEF_ORI,
    "left_gripper": LEFT_GRIPPER,
    "right_gripper": RIGHT_GRIPPER,
    "waist": WAIST,
    "chassis_velocity": CHASSIS_VELOCITY,
    "left_arm": LEFT_ARM,
    "right_arm": RIGHT_ARM,
    "reserved": RESERVED,
}

# ``assets/`` ships inside the package (…/tau0_vla/data/assets), so the registry
# resolves from an installed wheel and not just from a source checkout.
_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_DIM_REGISTRY_PATH = _ASSETS_DIR / "dim_registry.json"

_DIM_REGISTRY_CACHE: dict[str, Any] | None = None

_euler2rot6d = Euler2Rot6D()
_quat2rot6d_xyzw = Quat2Rot6D(quat_order="xyzw")


# ─────────────────────────────────────────────────────────────────────────────
# Dim registry loader
# ─────────────────────────────────────────────────────────────────────────────


def load_dim_registry(path: str | Path | None = None) -> dict[str, Any]:
    global _DIM_REGISTRY_CACHE
    if _DIM_REGISTRY_CACHE is not None and path is None:
        return _DIM_REGISTRY_CACHE
    p = Path(path) if path else _DIM_REGISTRY_PATH
    with open(p) as f:
        reg = json.load(f)
    if path is None:
        _DIM_REGISTRY_CACHE = reg
    return reg


def get_registry_entry(registry_key: str) -> dict[str, Any]:
    reg = load_dim_registry()
    if registry_key not in reg:
        raise KeyError(
            f"Unknown registry_key={registry_key!r}. "
            f"Available: {sorted(reg.keys())}"
        )
    return reg[registry_key]


# ─────────────────────────────────────────────────────────────────────────────
# Mask builder
# ─────────────────────────────────────────────────────────────────────────────


def build_unified_mask(
    registry_entry: dict[str, Any],
    *,
    has_eef: bool,
    role: str = "action",
) -> np.ndarray:
    """Build a [UNIFIED_DIM] boolean mask for one robot type.

    ``role`` is "action" or "state" — determines which groups dict to check.

    EEF-priority: when ``has_eef=True`` the arm-joint slots are excluded from BOTH
    the state and action masks (the body is presented purely in EEF space); see the
    arm block below.
    """
    mask = np.zeros(UNIFIED_DIM, dtype=np.float32)
    groups = registry_entry.get(f"{role}_groups", registry_entry.get("action_groups", {}))

    if has_eef:
        mask[LEFT_EEF] = 1.0
        mask[RIGHT_EEF] = 1.0

    if "left_gripper" in groups:
        mask[LEFT_GRIPPER] = 1.0
    if "right_gripper" in groups:
        mask[RIGHT_GRIPPER] = 1.0

    n_waist = len(groups.get("waist", []))
    if n_waist > 0:
        mask[WAIST.start : WAIST.start + min(n_waist, 2)] = 1.0

    n_chassis = len(groups.get("chassis_velocity", []))
    if n_chassis > 0:
        mask[CHASSIS_VELOCITY.start : CHASSIS_VELOCITY.start + min(n_chassis, 2)] = 1.0

    # EEF-priority contract: when a body supplies native EEF values, the
    # arm-joint slots are dropped from BOTH the state and action masks — the body
    # is presented purely in EEF space (EEF pose proprio in, EEF delta out),
    # matching the legacy G1 EefPose contract. Rationale: EEF slots [0:18] are
    # cross-embodiment aligned (Cartesian pose), whereas the arm slots [24:40] are
    # not (a 6-DoF and a 7-DoF arm occupy the same slots with different semantics),
    # so keeping joints would double-represent the same motion and pollute
    # cross-embodiment learning. Joint-only bodies (no EEF) still use arm joints,
    # and the no-EEF fallback mask (has_eef=False) also keeps them.
    supervise_arms = not has_eef

    n_left = len(groups.get("left_arm", []))
    if n_left > 0 and supervise_arms:
        mask[LEFT_ARM.start : LEFT_ARM.start + min(n_left, 8)] = 1.0

    n_right = len(groups.get("right_arm", []))
    if n_right > 0 and supervise_arms:
        mask[RIGHT_ARM.start : RIGHT_ARM.start + min(n_right, 8)] = 1.0

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# EEF conversion helpers
# ─────────────────────────────────────────────────────────────────────────────


def _euler_block_to_rot6d(euler_6d: np.ndarray) -> np.ndarray:
    """Convert a single [6D] euler block (xyz+euler) → [9D] (xyz+rot6d).

    Handles both [6] and [..., 6] shapes. Uses the same Euler2Rot6D
    transform used elsewhere in the pipeline.
    """
    return _euler2rot6d.forward(
        euler_6d, context=None, component=None, state_reference=None
    )


def _quat_block_to_rot6d(quat_7d: np.ndarray) -> np.ndarray:
    """Convert a single [7D] quat block (xyz+quat_xyzw) → [9D] (xyz+rot6d)."""
    return _quat2rot6d_xyzw.forward(
        quat_7d, context=None, component=None, state_reference=None
    )


# ─────────────────────────────────────────────────────────────────────────────
# Relative (delta) action contract
# ─────────────────────────────────────────────────────────────────────────────
# All unified embodiments train with RELATIVE actions, mirroring the legacy
# per-component transforms (every legacy config applies ``RelativeToState`` to
# arm_joint / eef_pose action components, while gripper / waist / chassis stay
# absolute). The unified scatter is a fixed layout, so the policy is expressed as
# fixed slot groups rather than declared components:
#   - EEF blocks ([0:9], [9:18]): pose-relative (body-frame xyz delta + rotation
#     delta), reusing ``RelativeToState._forward_eef_pose`` / ``_inverse_eef_pose``.
#   - Arm blocks ([24:32], [32:40]): plain ``action - state`` per active dim.
#   - gripper / waist / chassis: untouched (absolute).
# The delta is taken vs the single-frame (current) state, broadcast over the
# action horizon — same as ``RelativeToState`` on an action chunk.
_RELATIVE_EEF_BLOCKS = (LEFT_EEF, RIGHT_EEF)
_RELATIVE_ARM_BLOCKS = (LEFT_ARM, RIGHT_ARM)


def relativize_unified_action(action: np.ndarray, state: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    """Forward: make active EEF + arm action slots relative to ``state``.

    ``action`` is ``[*horizon, 40]``, ``state`` is ``[40]`` (absolute, scattered,
    pre-normalize). EEF blocks are gated on the whole 9D block being active
    (rot6d-of-zeros would NaN); arm blocks use per-dim masking so a 7-DoF arm in
    an 8-slot group is handled correctly.
    """
    action = np.array(action, dtype=np.float32, copy=True)
    state = np.asarray(state, dtype=np.float32)
    mask = np.asarray(action_mask, dtype=np.float32)
    for blk in _RELATIVE_EEF_BLOCKS:
        if bool(np.all(mask[blk] > 0)):
            action[..., blk] = RelativeToState._forward_eef_pose(action[..., blk], state[blk])
    for blk in _RELATIVE_ARM_BLOCKS:
        action[..., blk] = action[..., blk] - state[blk] * mask[blk]
    return action


def unrelativize_unified_action(action: np.ndarray, state: np.ndarray, action_mask: np.ndarray) -> np.ndarray:
    """Inverse of :func:`relativize_unified_action` (deploy side): add ``state`` back."""
    action = np.array(action, dtype=np.float32, copy=True)
    state = np.asarray(state, dtype=np.float32)
    mask = np.asarray(action_mask, dtype=np.float32)
    for blk in _RELATIVE_EEF_BLOCKS:
        if bool(np.all(mask[blk] > 0)):
            action[..., blk] = RelativeToState._inverse_eef_pose(action[..., blk], state[blk])
    for blk in _RELATIVE_ARM_BLOCKS:
        action[..., blk] = action[..., blk] + state[blk] * mask[blk]
    return action


# ─────────────────────────────────────────────────────────────────────────────
# Unified assembler
# ─────────────────────────────────────────────────────────────────────────────


class UnifiedAssembler:
    """Assemble raw joint + EEF data into the fixed 40D unified layout.

    Replaces ``ComponentAssembler`` for unified configs. Each robot type's
    raw data is scattered into semantically-fixed positions, and a per-sample
    mask indicates which positions are valid.
    """

    def __init__(
        self,
        *,
        registry_key: str,
        has_eef_action: bool = True,
        has_eef_state: bool = True,
        eef_format: str = "euler",
        is_single_arm: bool = False,
        norm_stats: dict[str, Any] | None = None,
        per_embodiment_stats: dict[str, dict[str, Any]] | None = None,
        use_quantiles: bool = False,
        disable_normalization: bool = False,
        eef_provider: "Callable[[dict[str, Any]], dict[str, Any]] | None" = None,
        relative_action: bool = True,
        prefer_eef: bool = True,
        return_all_norm_forms: bool = False,
    ):
        self.entry = get_registry_entry(registry_key)
        # When True, __call__ also emits ``extras`` (raw / mean_std_norm / q_norm
        # per role) so downstream consumers like the FAST action tokenizer
        # (TransformAction2Fast, which needs extras['action']['q_norm']) work for
        # unified leaves the same way they already do for legacy modality leaves.
        self.return_all_norm_forms = bool(return_all_norm_forms)
        self.has_eef_action = has_eef_action
        self.has_eef_state = has_eef_state
        self.eef_format = eef_format
        self.is_single_arm = is_single_arm
        self.norm_stats = norm_stats or {}
        self.use_quantiles = use_quantiles
        self.disable_normalization = disable_normalization
        # Train RELATIVE (delta) actions for arm_joint / eef_pose slots — the
        # unified analog of the legacy per-component RelativeToState. Applied to
        # the action AFTER scatter and BEFORE normalize (transform → normalize
        # order, matching legacy). Independent of normalization: even with
        # ``disable_normalization`` (stats computation) the action is relativized,
        # so the stats are of the delta. The deploy inverse is
        # ``data_spec.restore_unified_action``.
        self.relative_action = relative_action
        # Optional body-local hook that populates `_eef_state_raw`/`_eef_action_raw`
        # on the sample BEFORE scatter (for example, inline-index extraction).
        # The generic assembler stays body-agnostic: derivation lives
        # with the robot family. None → EEF comes from the
        # declared `_eef_*_col` parquet columns (the common case).
        self.eef_provider = eef_provider

        self.action_groups = self.entry.get("action_groups", {})
        self.state_groups = self.entry.get("state_groups", {})

        self.action_mask = build_unified_mask(
            self.entry, has_eef=has_eef_action, role="action",
        )
        self.state_mask = build_unified_mask(
            self.entry, has_eef=has_eef_state, role="state",
        )
        # Per-sample mask variants (EEF on vs off). Used when an embodiment has
        # EEF only on a subset of episodes and signals it via an `eef_present`
        # flag column (see `_eef_present_col`). For embodiments without the
        # flag, presence is inferred from the raw EEF being finite.
        #   - EEF ABSENT  -> ``*_mask_noeef``: EEF off, ARM on (joint fallback).
        #   - EEF PRESENT -> ``*_mask_eef``  : EEF on, ARM off when ``prefer_eef``
        #     (prefer-EEF policy — supervise/predict EEF only, never the redundant
        #     joints; matches legacy G01 ``eef``/``eef_wbc`` configs, which carry no
        #     ArmJoint component). gripper/waist/chassis stay on in both.
        self._action_mask_noeef = build_unified_mask(self.entry, has_eef=False, role="action")
        self._state_mask_noeef = build_unified_mask(self.entry, has_eef=False, role="state")
        self.prefer_eef = prefer_eef
        self._action_mask_eef = self._arm_off(self.action_mask) if prefer_eef else self.action_mask
        self._state_mask_eef = self._arm_off(self.state_mask) if prefer_eef else self.state_mask

        if per_embodiment_stats and registry_key in per_embodiment_stats:
            self._resolved_norm_stats = per_embodiment_stats[registry_key]
        else:
            self._resolved_norm_stats = self.norm_stats

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        if self.eef_provider is not None:
            sample = self.eef_provider(sample)
        updated = copy.deepcopy(sample)
        joint_state_raw = np.asarray(updated.pop("_state_raw"), dtype=np.float32)
        joint_action_raw = np.asarray(updated.pop("_action_raw"), dtype=np.float32)
        eef_state_raw = updated.pop("_eef_state_raw", None)
        eef_action_raw = updated.pop("_eef_action_raw", None)
        eef_present_raw = updated.pop("_eef_present_raw", None)
        updated.pop("_field_descriptions", None)

        if eef_state_raw is not None:
            eef_state_raw = np.asarray(eef_state_raw, dtype=np.float32)
        if eef_action_raw is not None:
            eef_action_raw = np.asarray(eef_action_raw, dtype=np.float32)

        # Per-sample EEF presence: when an `eef_present` flag is provided, EEF
        # slots are only active (and supervised) for samples that actually carry
        # a pose; joints/grippers are always active. Without a flag, presence is
        # inferred from the raw EEF (unchanged behaviour for existing robots).
        present_state = self._sample_has_eef(eef_state_raw, eef_present_raw, self.has_eef_state)
        present_action = self._sample_has_eef(eef_action_raw, eef_present_raw, self.has_eef_action)
        # Prefer-EEF: when EEF is present the arm slots are masked off (EEF-only
        # supervision); when absent, joints are the fallback (``*_mask_noeef``).
        state_mask = self._state_mask_eef if present_state else self._state_mask_noeef
        action_mask = self._action_mask_eef if present_action else self._action_mask_noeef

        state = self._scatter(
            joint_raw=joint_state_raw,
            eef_raw=eef_state_raw,
            groups=self.state_groups,
            has_eef=present_state,
        )
        action = self._scatter(
            joint_raw=joint_action_raw,
            eef_raw=eef_action_raw,
            groups=self.action_groups,
            has_eef=present_action,
        )

        # Relative (delta) action BEFORE normalize, using the absolute scattered
        # state. Runs regardless of normalization so stats are computed on the
        # delta. Deploy inverse: ``data_spec.restore_unified_action``.
        if self.relative_action:
            action = relativize_unified_action(action, state, action_mask)

        # Pre-normalize (post-relativize) scattered values — the unified analog of
        # the legacy ``raw`` extras form, and the input both normalized forms use.
        state_pre, action_pre = state, action

        if not self.disable_normalization:
            state = self._normalize(state, role="state", mask=state_mask)
            action = self._normalize(action, role="action", mask=action_mask)
        else:
            # No normalization (e.g. stats computation): still re-zero inactive
            # slots so masked-off arms (prefer-EEF) don't leak their scattered raw
            # values into the output / the (mask-aware) stats accumulator.
            state = state * state_mask
            action = action * action_mask

        # Keep the prompt's ``Control mode`` label honest per-sample: it reflects
        # what is actually SUPERVISED in the action vector — ``eef`` when this
        # sample's EEF slots are active, ``joint`` on the EEF-absent fallback. For
        # always-eef / always-joint bodies this equals the static template (no-op);
        # only mixed-presence bodies (e.g. Yam) flip. Based on the ACTION presence
        # (the space the policy predicts), which the control mode conditions.
        prompt = _rewrite_control_mode(
            updated.get("prompt"), "eef" if present_action else "joint"
        )
        result = {
            "images": copy.deepcopy(updated.get("images", {})),
            "prompt": prompt,
            "state": state,
            "action": action,
            "state_mask": state_mask.copy(),
            "action_mask": action_mask.copy(),
        }
        if self.return_all_norm_forms:
            result["extras"] = {
                "state": self._norm_forms(state_pre, role="state", mask=state_mask),
                "action": self._norm_forms(action_pre, role="action", mask=action_mask),
            }
        return result

    @staticmethod
    def _arm_off(mask: np.ndarray) -> np.ndarray:
        """Copy ``mask`` with the LEFT_ARM/RIGHT_ARM slots zeroed (prefer-EEF)."""
        m = mask.copy()
        m[LEFT_ARM] = 0.0
        m[RIGHT_ARM] = 0.0
        return m

    @staticmethod
    def _sample_has_eef(eef_raw, eef_present_raw, has_eef_cfg: bool) -> bool:
        """Whether this sample's EEF slots should be active.

        ``has_eef_cfg`` gates on the config declaring EEF columns at all. When an
        ``eef_present`` flag is provided it takes precedence (>0.5 == present);
        otherwise presence is inferred from the raw EEF being finite.
        """
        if not has_eef_cfg:
            return False
        if eef_present_raw is not None:
            return bool(np.all(np.asarray(eef_present_raw, dtype=np.float32) > 0.5))
        if eef_raw is None:
            return False
        return bool(np.isfinite(np.asarray(eef_raw)).all())

    def _scatter(
        self,
        *,
        joint_raw: np.ndarray,
        eef_raw: np.ndarray | None,
        groups: dict[str, list[int]],
        has_eef: bool,
    ) -> np.ndarray:
        """Scatter raw data into the unified 40D layout.

        ``joint_raw`` shape: [D] for state, [H, D] for action (with horizon).
        ``eef_raw`` shape: [E] or [H, E], may be None.
        """
        prefix_shape = joint_raw.shape[:-1]
        out = np.zeros((*prefix_shape, UNIFIED_DIM), dtype=np.float32)

        # --- EEF slots ---
        if has_eef and eef_raw is not None:
            self._scatter_eef(out, eef_raw)

        # --- Arm joint slots ---
        left_arm_idx = groups.get("left_arm", [])
        if left_arm_idx:
            n = min(len(left_arm_idx), 8)
            out[..., LEFT_ARM.start : LEFT_ARM.start + n] = joint_raw[..., left_arm_idx[:n]]

        right_arm_idx = groups.get("right_arm", [])
        if right_arm_idx:
            n = min(len(right_arm_idx), 8)
            out[..., RIGHT_ARM.start : RIGHT_ARM.start + n] = joint_raw[..., right_arm_idx[:n]]

        # --- Gripper slots ---
        left_grip_idx = groups.get("left_gripper", [])
        if left_grip_idx:
            out[..., LEFT_GRIPPER] = joint_raw[..., left_grip_idx[0]:left_grip_idx[0] + 1]

        right_grip_idx = groups.get("right_gripper", [])
        if right_grip_idx:
            out[..., RIGHT_GRIPPER] = joint_raw[..., right_grip_idx[0]:right_grip_idx[0] + 1]

        waist_idx = groups.get("waist", [])
        if waist_idx:
            n = min(len(waist_idx), 2)
            out[..., WAIST.start : WAIST.start + n] = joint_raw[..., waist_idx[:n]]

        chassis_idx = groups.get("chassis_velocity", [])
        if chassis_idx:
            n = min(len(chassis_idx), 2)
            out[..., CHASSIS_VELOCITY.start : CHASSIS_VELOCITY.start + n] = joint_raw[..., chassis_idx[:n]]

        return out

    def _scatter_eef(self, out: np.ndarray, eef_raw: np.ndarray) -> None:
        """Fill EEF slots from raw EEF data.

        RoboCoin EEF: 12D euler [L_xyz(3) L_euler(3) R_xyz(3) R_euler(3)]
        RoboMind agilex EEF: 14D quat [L_xyz(3) L_quat(4) R_xyz(3) R_quat(4)]
        RoboMind xsens EEF: 12D euler (same as RoboCoin)
        """
        if self.eef_format == "euler":
            eef_dim = eef_raw.shape[-1]
            if eef_dim >= 12:
                left_euler = eef_raw[..., 0:6]
                right_euler = eef_raw[..., 6:12]
                left_rot6d = _euler_block_to_rot6d(left_euler)
                right_rot6d = _euler_block_to_rot6d(right_euler)
                out[..., LEFT_EEF] = left_rot6d
                out[..., RIGHT_EEF] = right_rot6d
            elif eef_dim >= 6:
                left_euler = eef_raw[..., 0:6]
                left_rot6d = _euler_block_to_rot6d(left_euler)
                out[..., LEFT_EEF] = left_rot6d
        elif self.eef_format == "quat":
            eef_dim = eef_raw.shape[-1]
            if eef_dim >= 14:
                left_quat = eef_raw[..., 0:7]
                right_quat = eef_raw[..., 7:14]
                left_rot6d = _quat_block_to_rot6d(left_quat)
                right_rot6d = _quat_block_to_rot6d(right_quat)
                out[..., LEFT_EEF] = left_rot6d
                out[..., RIGHT_EEF] = right_rot6d
            elif eef_dim >= 7:
                left_quat = eef_raw[..., 0:7]
                left_rot6d = _quat_block_to_rot6d(left_quat)
                out[..., LEFT_EEF] = left_rot6d

    _STD_FLOOR = 0.2

    def _normalize(self, values: np.ndarray, *, role: str, mask: np.ndarray | None = None) -> np.ndarray:
        """Select the per-embodiment (or global fallback) stats for this role and
        delegate the masked, std-floored math to ``stats.normalize_masked`` — the
        single normalization implementation in the pipeline. This method
        only does the *addressing* (which NormStats to use); it does no math."""
        from tau0_vla.data.stats import normalize_masked

        if role not in self._resolved_norm_stats:
            return values
        if mask is None:
            mask = self.action_mask if role == "action" else self.state_mask
        return normalize_masked(
            values,
            self._resolved_norm_stats[role],
            mask=mask,
            std_floor=self._STD_FLOOR,
        )

    def _norm_forms(self, value: np.ndarray, *, role: str, mask: np.ndarray) -> dict[str, "np.ndarray | None"]:
        """Per-role ``{raw, mean_std_norm, q_norm}`` for the ``extras`` payload,
        matching the legacy modality format (pipeline._concat_extras_per_form) so
        the same downstream transforms (FAST tokenizer) work unchanged. ``value``
        is the pre-normalize scattered/relativized 40D tensor; all forms are
        mask-zeroed on inactive slots. ``q_norm`` is None when the norm file
        carries no q01/q99 (graceful degrade, as in the legacy path)."""
        from tau0_vla.data.stats import normalize_masked

        value = np.asarray(value, dtype=np.float32)
        raw = (value * mask).astype(np.float32)
        stats = self._resolved_norm_stats.get(role)
        if stats is None:
            return {"raw": raw, "mean_std_norm": None, "q_norm": None}
        mean_std = normalize_masked(value, stats, mask=mask, std_floor=self._STD_FLOOR)
        q_norm = None
        if getattr(stats, "q01", None) is not None and getattr(stats, "q99", None) is not None:
            n = value.shape[-1]
            q01 = np.asarray(stats.q01, dtype=np.float32)[..., :n]
            q99 = np.asarray(stats.q99, dtype=np.float32)[..., :n]
            q_norm = (((value - q01) / (q99 - q01 + 1e-6)) * 2.0 - 1.0).astype(np.float32) * mask
        return {"raw": raw, "mean_std_norm": mean_std.astype(np.float32), "q_norm": q_norm}


# ─────────────────────────────────────────────────────────────────────────────
# Unified mixin for RobotConfig subclasses
# ─────────────────────────────────────────────────────────────────────────────
#
# Body-specific EEF extraction is NOT here — it lives with the robot family as
# an ``_eef_provider``. The generic ``UnifiedAssembler`` above just consumes
# whatever ``_eef_*_raw`` the provider (or declared columns) produced.


class _UnifiedMixin:
    """Mixin that wires a RobotConfig subclass to the unified 40D layout.

    Subclasses set ClassVars to declare the registry key and EEF columns.
    MRO ensures this mixin's overrides take precedence over the parent
    RobotConfig methods.
    """

    _unified_registry_key: ClassVar[str]
    _eef_state_col: ClassVar[str | None] = None
    _eef_action_col: ClassVar[str | None] = None
    _eef_present_col: ClassVar[str | None] = None
    _eef_format: ClassVar[str] = "euler"
    _is_single_arm: ClassVar[bool] = False

    def _raw_repack_structure(self) -> dict[str, Any]:
        base = super()._raw_repack_structure()
        if self._eef_state_col:
            base["_eef_state_raw"] = self._eef_state_col
        if self._eef_action_col:
            base["_eef_action_raw"] = self._eef_action_col
        if self._eef_present_col:
            base["_eef_present_raw"] = self._eef_present_col
        return base

    def _eef_provider(self):
        """Return a callable that populates ``_eef_state_raw`` / ``_eef_action_raw``
        on a sample before scatter, or ``None`` (default) when EEF comes from the
        declared ``_eef_*_col`` columns. Robot families that extract EEF from
        native raw vectors may override this so body-specific logic remains in
        the adapter."""
        return None

    def _resolve_source_delta_timestamps(self, source_kwargs):
        """Window the declared ``_eef_action_col`` with the SAME offsets as the
        joint action column.

        The base resolver only windows ``repack['action']['raw']``; a declared
        EEF action column is a top-level repack extra, so LeRobot loads it as a
        single current-frame row [D] which ``_scatter`` then silently
        numpy-broadcasts across the horizon — every chunk step becomes the same
        pose and, after ``relativize_unified_action``, the same-frame
        action−state gap instead of the growing chunk displacement (the
        2026-07-07 eef-action broadcast bug; hy trained on constant ~0.5 mm
        chunks while its norm stats describe the windowed ±49 mm quantity).

        ``_eef_state_col`` stays single-frame on purpose: state is the current
        frame by contract. Bodies without a declared column are untouched.
        """
        merged = super()._resolve_source_delta_timestamps(source_kwargs)
        if not self._eef_action_col or merged is None:
            return merged
        action_col = self.repack.get("action", {}).get("raw", "action")
        offsets = merged.get(action_col)
        if offsets is None or self._eef_action_col in merged:
            return merged
        merged = dict(merged)
        merged[self._eef_action_col] = list(offsets)
        return merged

    def _build_component_assembler(self, *, field_descriptions, **kwargs):
        disable = kwargs.get("disable_component_normalization", False)
        norm_stats = None
        per_embodiment_stats = None
        if not disable:
            norm_stats, per_embodiment_stats = self._load_unified_norm_stats()
        eef_provider = self._eef_provider()
        # EEF slots are active when EEF is sourced either from a declared native
        # column or an adapter provider.
        has_eef_state = self._eef_state_col is not None or eef_provider is not None
        has_eef_action = self._eef_action_col is not None or eef_provider is not None
        return UnifiedAssembler(
            registry_key=self._unified_registry_key,
            has_eef_action=has_eef_action,
            has_eef_state=has_eef_state,
            eef_format=self._eef_format,
            is_single_arm=self._is_single_arm,
            norm_stats=norm_stats,
            per_embodiment_stats=per_embodiment_stats,
            disable_normalization=disable,
            eef_provider=eef_provider,
            return_all_norm_forms=getattr(self, "return_all_norm_forms", False),
        )

    def _load_unified_norm_stats(self):
        """Load norm stats returning (global_stats, per_embodiment_stats)."""
        from tau0_vla.data.stats import load_file_with_per_embodiment

        if self.norm_stats_path is not None:
            return load_file_with_per_embodiment(self.norm_stats_path)
        global_stats = self._load_norm_stats()
        return global_stats, {}
