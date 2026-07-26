"""Worked finetuning example for the bundled AgiBot World subset."""

from __future__ import annotations

from pathlib import Path

from tau0_vla.adapters.g1 import G1AgibotUnified
from tau0_vla.data import FrameFilter, PromptSource, register_config
from tau0_vla.data.modalities import ArmJoint, Gripper, Image, Prompt
from tau0_vla.data.modalities.image import ColorJitter, ResizeWithPad


_REPO = str(Path(__file__).resolve().parents[2] / "example_data")
_NORM_STATS = str(Path(__file__).with_name("norm_stats.json"))

_IMAGE_TRANSFORMS = [
    ColorJitter(prob=0.33, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03),
    ResizeWithPad(224, 224),
]


def _robot_prompt(robot_type: str, control_mode: str) -> str:
    return (
        "You are controlling a robot.\n"
        f"Robot type: {robot_type}\n"
        f"Control mode: {control_mode}\n"
        "Whole-body control: disabled\n"
        "Task: {instruction}"
    )


@register_config
def agibot_world_gong_ft() -> G1AgibotUnified:
    """Post-train the released checkpoint on the public gong-striking task."""

    return G1AgibotUnified(
        repo_id=_REPO,
        images=[
            Image("head", transforms=_IMAGE_TRANSFORMS),
            Image("wrist_left", transforms=_IMAGE_TRANSFORMS),
            Image("wrist_right", transforms=_IMAGE_TRANSFORMS),
        ],
        # Read the detailed instruction from meta/info.json for the current
        # episode/frame instead of duplicating it in this config.
        prompt_source=PromptSource.from_label(source="instruction_segments"),
        prompt=Prompt(template=_robot_prompt("G1", "joint")),
        # Only annotated instruction spans are valid anchors. The action-horizon
        # tail stays within the same span, so prompt and target cannot drift.
        filter_by_segments=True,
        frame_filter=FrameFilter(positive=["l3"], negative=[]),
        state=[ArmJoint(normalize="none"), Gripper(normalize="none")],
        action=[ArmJoint(normalize="none"), Gripper(normalize="none")],
        action_horizon=30,
        state_padding_dim=40,
        action_padding_dim=40,
        norm_stats_path=_NORM_STATS,
        return_all_norm_forms=True,
    )
