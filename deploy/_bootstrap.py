"""Deploy entry-point bootstrap. Call at the top of every ``main()`` before
any ``tau0_vla.data`` registry lookup or ``load_checkpoint_spec`` call.

Two concerns kept together because they always fire as a pair:

  1. ``ensure_configs_registered`` — walk ``configs/`` so every
     ``@register_config`` fires and the data pipeline's registry resolves
     our config names.
  2. ``ensure_policy_manifest`` — retrofit ``policy_manifest.json`` from
     ``run_spec.json`` for ckpts trained before ``save_policy_manifest``
     landed in ``train.py`` (without it, ``load_checkpoint_spec`` fails).

Neither belongs in ``policy.py``: policy is inference
(encode → forward → decode); these are process startup + ckpt format
migration.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

# ─── @register_config bootstrap ──────────────────────────────────────────
# Training runs _import_sibling_data(yaml_path); deploy has no yaml, so walk
# configs/ directly. Canonical module name (configs.<subdir>.data via PEP 420
# namespace package) matters: if ``tau0_vla.data`` later does
# importlib.import_module("configs.X.data") from a manifest's config_modules,
# sys.modules cache hits and the file isn't re-exec'd — otherwise the same
# @register_config fires twice and the registry raises "already registered".

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIGS_REGISTERED = False


# Runs on import — before sibling ``from deploy.policy import ...`` lines
# in the same file can resolve ``tau0_vla``. Deferring this into
# ``ensure_configs_registered()`` is too late: policy.py imports
# ``tau0_vla.utils.run_spec`` at its own import time.
_SRC_DIR = _REPO_ROOT / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def ensure_configs_registered() -> None:
    global _CONFIGS_REGISTERED
    if _CONFIGS_REGISTERED:
        return

    configs_dir = _REPO_ROOT / "configs"
    if configs_dir.is_dir():
        repo_root_str = str(_REPO_ROOT)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        for data_py in sorted(configs_dir.glob("*/data.py")):
            module_name = f"configs.{data_py.parent.name}.data"
            if module_name in sys.modules:
                continue
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[warn] {data_py.relative_to(_REPO_ROOT)} "
                    f"(as {module_name}): {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

    _CONFIGS_REGISTERED = True


# ─── Manifest retrofit for pre-save_policy_manifest ckpts ────────────────
# Field derivation mirrors tau0_vla/trainer/train.py:437-452; keep in lockstep.

def ensure_policy_manifest(run_dir: Path, *, verbose: bool = False) -> bool:
    """Write ``policy_manifest.json`` if missing. Idempotent. Assumes
    ``ensure_configs_registered`` has already run."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        if verbose:
            print(f"[skip] {run_dir}: not a directory")
        return False

    manifest_path = run_dir / "policy_manifest.json"
    if manifest_path.exists():
        return False

    run_spec_path = run_dir / "run_spec.json"
    if not run_spec_path.exists():
        if verbose:
            print(f"[skip] {run_dir}: no run_spec.json")
        return False

    run_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
    model_args = run_spec.get("model_args") or {}
    data_args = run_spec.get("data_args") or {}
    training_args = run_spec.get("training_args") or {}

    config_name = data_args.get("config_name")
    if not config_name:
        if verbose:
            print(f"[skip] {run_dir}: run_spec.json has no data_args.config_name")
        return False

    policy_type = (
        model_args.get("vla_model_type")
        or ("reward" if data_args.get("use_progress_regression") else "vlm")
    )
    train_config_name = (
        training_args.get("run_name") or config_name or "tau0_vla"
    )
    config_modules = list(data_args.get("config_modules") or [])
    if config_modules:
        from tau0_vla.data.config import discover_config_modules
        discover_config_modules(module_names=config_modules)

    from tau0_vla.data import save_policy_manifest
    manifest = save_policy_manifest(
        manifest_path,
        train_config_name=str(train_config_name),
        finch_config_name=str(config_name),
        policy_type=str(policy_type),
        config_modules=config_modules,
    )
    if verbose:
        print(
            f"[ok]   {run_dir}/policy_manifest.json  "
            f"policy_type={manifest['policy_type']}  "
            f"routes={manifest['routes']}"
        )
    return True


def discover_checkpoint_config_modules(run_dir: Path, *, verbose: bool = False) -> list[str]:
    """Import the config modules a checkpoint was trained with.

    Training records ``data_args.config_modules`` into ``policy_manifest.json``
    (``trainer/train.py``), so a checkpoint carries the list of modules whose
    import registers its Robot Config. Re-importing them here is what lets a
    config that lives outside ``configs/*/data.py`` resolve at serve time
    without the user naming it again — the data contract travels with the
    checkpoint, which is the same principle as the Data Spec.

    Call this after :func:`ensure_policy_manifest` and *before* building a
    policy: ``from_checkpoint`` resolves the config name, and an unregistered
    name fails there.

    ``ensure_policy_manifest`` also discovers, but only on the path where it has
    to synthesise a missing manifest. A checkpoint written by a current training
    run already has one, so that branch returns early and nothing gets imported —
    which is exactly the case this function covers. Discovery is idempotent, so
    the overlap is harmless.

    Returns the modules imported, empty when the checkpoint declares none (the
    common case: configs found through the ``configs/*/data.py`` walk).
    """
    manifest_path = Path(run_dir) / "policy_manifest.json"
    if not manifest_path.exists():
        return []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # Not fatal here: load_checkpoint_spec parses the same file moments later
        # and will report a malformed manifest far better than we can.
        if verbose:
            print(f"[warn] {manifest_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return []

    modules = [str(m) for m in (manifest.get("config_modules") or ())]
    if not modules:
        return []

    from tau0_vla.data.config import discover_config_modules

    discover_config_modules(module_names=modules)
    if verbose:
        print(f"[ok]   imported config_modules from checkpoint: {modules}")
    return modules


def resolve_deploy_io(data_spec, *, override: str | None = None):
    """Import the ``deploy_io`` module that serves this checkpoint.

    Serving needs one embodiment-specific layer: SDK payload keys, camera
    aliases, and the action column permutation. Which adapter owns it is not a
    deployment choice — it is a property of the checkpoint, so the checkpoint
    answers it. Resolution order:

    1. ``override`` — an explicit ``--adapter`` for a config the Data Spec
       cannot name (one registered outside ``tau0_vla.adapters``).
    2. The Data Spec's ``robot_cls``, resolved from the recorded ``robot_name``
       through ``_ROBOT_CLASS_PATHS``. This works for every checkpoint ever
       written, including those predating the manifest field, because the robot
       name has always been part of the Data Spec.
    3. ``tau0_vla.adapters.g1`` — the worked example, and what a checkpoint
       whose robot class lives outside the adapters package used to get
       implicitly.

    Raises when an explicitly named adapter has no ``deploy_io``: a typo there
    should stop the server rather than silently serve the G1's wire format,
    which would write gripper commands into arm joints on real hardware.
    """
    from tau0_vla.data.checkpoint_spec import adapter_module_for_config

    package = override or adapter_module_for_config(getattr(data_spec, "robot_cls", None))
    if package is None:
        package = "tau0_vla.adapters.g1"
    try:
        return importlib.import_module(f"{package}.deploy_io")
    except ImportError as exc:
        raise ImportError(
            f"Adapter '{package}' has no importable deploy_io module. "
            "Serving needs layer 3 of an adapter — see "
            "src/tau0_vla/adapters/README.md. Pass --adapter <dotted.package> "
            "to name a different one."
        ) from exc


__all__ = [
    "discover_checkpoint_config_modules",
    "ensure_configs_registered",
    "ensure_policy_manifest",
    "resolve_deploy_io",
]
