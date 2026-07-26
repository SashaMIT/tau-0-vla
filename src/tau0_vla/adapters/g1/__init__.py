"""G1 embodiment adapter.

``layout.py`` owns the dataset layout and RobotConfig classes;
``deploy_io.py`` owns the SDK wire format. The public adapter stays in joint
space and does not ship a G01 URDF or URDF-backed kinematics.
"""

from tau0_vla.adapters.g1.layout import (
    G1_UNIFIED_CLASSES,
    G1A2dJointUnified,
    G1Agibot,
    G1AgibotUnified,
    G1DaasUnified,
    G1Observation,
    G1RobotConfig,
)

__all__ = [
    "G1A2dJointUnified",
    "G1Agibot",
    "G1AgibotUnified",
    "G1DaasUnified",
    "G1Observation",
    "G1RobotConfig",
    "G1_UNIFIED_CLASSES",
]
