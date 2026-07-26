from __future__ import annotations

import importlib
from collections.abc import Callable
from inspect import isclass
from pathlib import Path
from typing import Any

_CONFIGS: dict[str, Any] = {}
_DISCOVERED = False
_DISCOVERED_MODULES: set[str] = set()


def _normalize_config_name(name: str) -> str:
    return name.replace("_", "-")


def register_config(config: Any | None = None):
    def _store(value: Any) -> Any:
        name = _normalize_config_name(value.__name__)
        existing = _CONFIGS.get(name)
        if existing is not None:
            if getattr(existing, "__qualname__", None) == getattr(value, "__qualname__", None):
                return value
            raise ValueError(
                f"Config '{name}' is already registered by "
                f"{getattr(existing, '__module__', '<unknown>')}.{getattr(existing, '__name__', type(existing).__name__)}"
            )
        _CONFIGS[name] = value
        return value

    if config is not None:
        return _store(config)
    return _store


def register_repo_ids(
    manifest_path: str | Path,
    *,
    factory: Callable[[str], Any],
    name_from_repo_id: Callable[[str], str] | None = None,
    caller_module: str | None = None,
) -> list[str]:
    """Bulk-register one config per repo_id line in a manifest file.

    Each line (comments starting with '#' + blank lines ignored) is passed to
    ``factory`` to build a config object (typically a ``RobotConfig``). The
    result is registered under ``name_from_repo_id(repo_id)`` (defaults to
    joining the last 4 path components with underscores).

    Returns the list of **normalized** registered names. Training consumes one
    name at a time: the pipeline ships a single training route.

    Example::

        CHILDREN = register_repo_ids(
            "/path/to/repo_ids.txt",
            factory=lambda rid: G1Agibot(
                repo_id=rid,
                state=[ArmJoint(normalize="mean_std"), Gripper(normalize="mean_std")],
                action=[ArmJoint(normalize="mean_std", transforms=[RelativeToState()]),
                        Gripper(normalize="mean_std")],
                action_horizon=30,
                norm_stats_dir=f"/shared/stats/{Path(rid).name}",
            ),
        )

        # Then point data_args.config_name at any one of CHILDREN.
    """
    manifest = Path(manifest_path)
    repo_ids = [
        line.strip() for line in manifest.read_text().splitlines() if line.strip() and not line.strip().startswith("#")
    ]
    namer = name_from_repo_id or _default_name_from_repo_id
    module_name = caller_module or getattr(factory, "__module__", "__main__")

    registered: list[str] = []
    for repo_id in repo_ids:
        raw_name = namer(repo_id)
        fn = _make_factory_closure(repo_id=repo_id, raw_name=raw_name, factory=factory, module_name=module_name)
        register_config(fn)
        registered.append(_normalize_config_name(raw_name))
    return registered


def _default_name_from_repo_id(repo_id: str) -> str:
    parts = [p for p in Path(repo_id).parts if p and p != "/"]
    return "_".join(parts[-4:]) if parts else repo_id


def _make_factory_closure(*, repo_id: str, raw_name: str, factory: Callable[[str], Any], module_name: str):
    def _factory() -> Any:
        return factory(repo_id)

    _factory.__name__ = raw_name
    _factory.__qualname__ = raw_name
    _factory.__module__ = module_name
    return _factory


def get_config(name: str) -> Any:
    normalized = _normalize_config_name(name)
    try:
        entry = _CONFIGS[normalized]
    except KeyError as exc:
        available = ", ".join(sorted(_CONFIGS)) or "<empty>"
        raise KeyError(f"Unknown config '{name}'. Available configs: {available}") from exc
    if isclass(entry):
        config = entry()
    else:
        config = entry() if callable(entry) else entry
    return _attach_config_name(config, normalized)


def list_configs() -> list[str]:
    return sorted(_CONFIGS)


def _attach_config_name(config: Any, normalized_name: str) -> Any:
    """Stamp the registered name onto the config object.

    Robot Configs are frozen dataclasses, so ``object.__setattr__`` is the
    normal path; plain ``setattr`` covers anything that is not frozen.

    Failing both is not survivable and must not be swallowed: ``data_spec.py``
    reads this attribute to decide the config name recorded in a checkpoint's
    Data Spec, and falls back to ``robot_name`` when it is missing. A run would
    then train fine and write a spec naming the *robot* where the *config* name
    belongs, so inference would look up a name that was never registered.
    """
    try:
        object.__setattr__(config, "_finch_config_name", normalized_name)
    except Exception:
        try:
            setattr(config, "_finch_config_name", normalized_name)
        except Exception as exc:
            raise TypeError(
                f"cannot record the registered name on {type(config).__name__!r}: "
                f"both object.__setattr__ and setattr were refused ({exc}). "
                f"A config that cannot carry ``_finch_config_name`` would be "
                f"serialized into a checkpoint under its robot_name instead, "
                f"which inference cannot resolve."
            ) from exc
    return config


def discover_config_modules(
    *,
    module_names: list[str] | None = None,
    extra_module_names: list[str] | None = None,
) -> None:
    global _DISCOVERED

    # Merge module_names and extra_module_names for compatibility with both
    # checkpoint_spec (uses extra_module_names) and tau-0-vla (uses module_names).
    all_modules: list[str] = []
    if module_names is not None:
        all_modules.extend(module_names)
    if extra_module_names is not None:
        all_modules.extend(extra_module_names)

    if not all_modules:
        raise RuntimeError(
            "The pipeline does not auto-discover configs. "
            "Import the config module explicitly or use @register_config in the current process."
        )
    if _DISCOVERED and set(all_modules).issubset(_DISCOVERED_MODULES):
        return

    for module_name in all_modules:
        if module_name not in _DISCOVERED_MODULES:
            importlib.import_module(module_name)
            _DISCOVERED_MODULES.add(module_name)
    _DISCOVERED = True


def _reset_registry() -> None:
    _CONFIGS.clear()


def _reset_discovery() -> None:
    global _DISCOVERED
    _DISCOVERED = False
    _DISCOVERED_MODULES.clear()
