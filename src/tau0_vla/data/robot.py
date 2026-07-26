from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Sequence

from tau0_vla.data.config import discover_config_modules, get_config
from tau0_vla.data.robots import FinchResolvedRobotConfig, RobotConfig

_LEGACY_SEMANTIC_TRANSFORMS = frozenset(
    {
        "DeltaActions",
        "TransformActionAbs2Delta",
        "TransformActionJoint2Eef",
        "TransformActionWithPadding",
        "TransformFrankaGripper",
        "TransformPadStatesAndActions",
        "TransformStateAbs2Delta",
    }
)


def resolve_finch_robot_configs(
    *,
    repo_ids: Sequence[str] | None = None,
    finch_config_names: Sequence[str] | None = None,
    shared_finch_config_name: str | None = None,
    config_modules: Sequence[str] | None = None,
) -> list[FinchResolvedRobotConfig]:
    configs = _resolve_robot_configs(
        finch_config_names=finch_config_names,
        shared_finch_config_name=shared_finch_config_name,
        config_modules=config_modules,
    )
    configs, resolved_repo_ids = _align_robot_configs(configs, repo_ids=repo_ids)
    resolved: list[FinchResolvedRobotConfig] = []
    for robot_config, repo_id in zip(configs, resolved_repo_ids, strict=True):
        resolved.append(robot_config.resolve_output_spec_for_repo(repo_id))
    return resolved


def _resolve_robot_configs(
    *,
    finch_config_names: Sequence[str] | None,
    shared_finch_config_name: str | None,
    config_modules: Sequence[str] | None,
) -> list[RobotConfig]:
    discover_config_modules(module_names=list(config_modules) if config_modules else None)
    if finch_config_names is not None:
        robot_configs = [_resolve_single_robot_config(config_name) for config_name in finch_config_names]
    elif shared_finch_config_name is not None:
        resolved = get_config(shared_finch_config_name)
        if _is_data_mix(resolved):
            robot_configs = [_resolve_single_robot_config(config_name) for config_name in resolved.configs]
        else:
            robot_config = _coerce_robot_config(resolved)
            robot_configs = [robot_config]
    else:
        raise ValueError("Finch-first path requires finch_config_names or shared_finch_config_name")
    return robot_configs


def _resolve_repo_ids(robot_configs: Sequence[RobotConfig], *, repo_ids: Sequence[str] | None) -> list[str]:
    if repo_ids is None:
        return [str(Path(robot_config._primary_repo_id()).resolve()) for robot_config in robot_configs]
    return [str(Path(repo_id).resolve()) for repo_id in repo_ids]


def _align_robot_configs(
    robot_configs: Sequence[RobotConfig], *, repo_ids: Sequence[str] | None
) -> tuple[list[RobotConfig], list[str]]:
    aligned_configs = list(robot_configs)
    resolved_repo_ids = _resolve_repo_ids(aligned_configs, repo_ids=repo_ids)
    if repo_ids is not None and len(aligned_configs) == 1 and len(resolved_repo_ids) > 1:
        aligned_configs = [aligned_configs[0] for _ in resolved_repo_ids]
    if len(aligned_configs) != len(resolved_repo_ids):
        raise ValueError("Resolved robot config count does not match repo_ids")
    return aligned_configs, resolved_repo_ids


def _resolve_single_robot_config(config_name: str) -> RobotConfig:
    return _coerce_robot_config(get_config(config_name))


def _coerce_robot_config(config: Any) -> RobotConfig:
    if _is_data_mix(config):
        raise TypeError("Expected a concrete RobotConfig, got DataMix")
    if not isinstance(config, RobotConfig):
        raise TypeError(f"Expected RobotConfig, got {type(config).__name__}")
    return config


def _is_data_mix(config: Any) -> bool:
    return (
        dataclasses.is_dataclass(config)
        and hasattr(config, "configs")
        and hasattr(config, "proportions")
        and not isinstance(config, RobotConfig)
    )


def validate_legacy_semantic_transforms(transforms: Sequence[dict[str, Any]] | None) -> None:
    if transforms is None:
        return
    legacy = [transform["type"] for transform in transforms if transform.get("type") in _LEGACY_SEMANTIC_TRANSFORMS]
    if legacy:
        joined = ", ".join(legacy)
        raise ValueError(f"Finch dataloader no longer accepts legacy semantic transforms: {joined}")


__all__ = [
    "FinchResolvedRobotConfig",
    "RobotConfig",
    "resolve_finch_robot_configs",
    "validate_legacy_semantic_transforms",
]
