from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from tau0_vla.data.modalities.base import extract_field_descriptions_from_dataset_info

EXPECTED_BACKEND_VERSION = "0.4.1"
BACKEND_PACKAGE = "lerobot"


class _DropEmptyMultiDatasetFeatureWarning(logging.Filter):
    # LeRobot's MultiLeRobotDataset logs ``keys {extra_keys} of {repo_id} were
    # disabled ...`` once per child — including when ``extra_keys`` is an
    # empty set, which means nothing was actually disabled. For a 58-repo
    # manifest that's 58 pointless WARNING lines at startup. Drop only the
    # empty-set case; genuine mismatches still surface.
    _SENTINEL = "were disabled as they are not contained in all the other datasets"

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if self._SENTINEL not in message:
            return True
        return "keys set()" not in message


logging.getLogger().addFilter(_DropEmptyMultiDatasetFeatureWarning())


def _require_backend(expected: str = EXPECTED_BACKEND_VERSION) -> None:
    try:
        installed = metadata.version(BACKEND_PACKAGE)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"The dataset backend '{BACKEND_PACKAGE}' is not installed. "
            "Install it from tau-0-vla's requirements.txt, which pins the "
            "verified version."
        ) from exc

    # PEP 440 local-version identifiers (the bit after ``+``, e.g. ``0.4.1+agibot.1``)
    # encode provenance, not semantic compatibility. Compare the release segment
    # alone so vendored forks of the exact same upstream release pass through.
    if installed.split("+", 1)[0] != expected.split("+", 1)[0]:
        raise RuntimeError(
            f"tau-0-vla expects '{BACKEND_PACKAGE}' {expected}, but found {installed}. "
            "Reinstall from requirements.txt so the dataset reader matches the "
            "format this release was verified against."
        )


def _load_dataset_info(repo_id: str | Path) -> dict[str, Any] | None:
    path = Path(repo_id) / "meta" / "info.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


# Public alias — used by openpi openloop and checkpoint_spec workflows.
load_dataset_info = _load_dataset_info


def _load_field_descriptions(repo_id: str | Path) -> dict[str, Any]:
    return extract_field_descriptions_from_dataset_info(_load_dataset_info(repo_id))


def _load_dataset_fps(repo_id: str | Path) -> float | None:
    dataset_info = _load_dataset_info(repo_id)
    if dataset_info is None:
        return None
    fps = dataset_info.get("fps")
    return None if fps is None else float(fps)


def _episode_rows(dataset: Any) -> list[Mapping[str, Any]]:
    meta = getattr(dataset, "meta", None)
    raw_episodes = getattr(meta, "episodes", None)
    if raw_episodes is None:
        return []
    if isinstance(raw_episodes, Mapping):
        rows = list(raw_episodes.values())
    else:
        rows = list(raw_episodes)
    return [row for row in rows if isinstance(row, Mapping)]


def get_episode_index_bounds(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    # MultiLeRobotDataset holds children in ``self._datasets`` and flat-indexes
    # them by cumulative ``num_frames``. Aggregate each child's local bounds
    # with the matching frame offset so the returned indices line up with
    # ``MultiLeRobotDataset.__getitem__``.
    child_datasets = getattr(dataset, "_datasets", None)
    if child_datasets is not None:
        froms: list[np.ndarray] = []
        tos: list[np.ndarray] = []
        frame_offset = 0
        for ds in child_datasets:
            child_from, child_to = get_episode_index_bounds(ds)
            froms.append(child_from + frame_offset)
            tos.append(child_to + frame_offset)
            # ``num_frames`` is the authoritative per-child length (see
            # ``MultiLeRobotDataset.__getitem__``); fall back to the last row's
            # ``to`` if absent.
            num_frames = int(getattr(ds, "num_frames", 0))
            frame_offset += num_frames if num_frames > 0 else (int(child_to[-1]) if child_to.size else 0)
        return np.concatenate(froms), np.concatenate(tos)

    episode_data_index = getattr(dataset, "episode_data_index", None)
    if episode_data_index is not None:
        return (
            np.asarray(episode_data_index["from"], dtype=np.int64),
            np.asarray(episode_data_index["to"], dtype=np.int64),
        )

    rows = _episode_rows(dataset)
    if not rows:
        raise ValueError("LeRobotDataset exposes neither episode_data_index nor usable meta.episodes.")

    rows.sort(key=lambda row: int(row.get("episode_index", 0)))
    return (
        np.asarray([int(row["dataset_from_index"]) for row in rows], dtype=np.int64),
        np.asarray([int(row["dataset_to_index"]) for row in rows], dtype=np.int64),
    )


def get_episode_ids(dataset: Any) -> np.ndarray:
    # MultiLeRobotDataset: stack children's episode IDs and offset each child's
    # slice by the cumulative episode count of the prior children, so two
    # children never collide on the same ID.
    child_datasets = getattr(dataset, "_datasets", None)
    if child_datasets is not None:
        ids: list[np.ndarray] = []
        ep_offset = 0
        for ds in child_datasets:
            child_ids = get_episode_ids(ds)
            ids.append(child_ids + ep_offset)
            ep_offset += int(child_ids.shape[0])
        return np.concatenate(ids) if ids else np.empty(0, dtype=np.int64)

    episodes = getattr(dataset, "episodes", None)
    if episodes is not None:
        return np.asarray(episodes, dtype=np.int64)

    rows = _episode_rows(dataset)
    if not rows:
        ep_from, _ = get_episode_index_bounds(dataset)
        return np.arange(int(ep_from.shape[0]), dtype=np.int64)

    rows.sort(key=lambda row: int(row.get("episode_index", 0)))
    return np.asarray([int(row.get("episode_index", idx)) for idx, row in enumerate(rows)], dtype=np.int64)


def _attach_episode_indices(dataset: Any) -> None:
    ep_from, ep_to = get_episode_index_bounds(dataset)
    episode_ids = get_episode_ids(dataset)
    dataset.episodes = episode_ids
    dataset.episode_data_index = {
        "from": ep_from,
        "to": ep_to,
    }
    dataset._ep_idx_to_episode_idx = episode_ids


def load_aid_episode_id_mapping(repo_id: str | Path) -> dict[str, int]:
    """Build ``{episode_id: episode_index}`` from a repo's ``meta/info.json``.

    AgiBot v3.0 datasets carry an ``aid_info`` block keyed by stringified
    ``episode_index``; each entry has an external ``episode_id`` (stable
    across re-collections). This function inverts that mapping so callers
    can translate external IDs into the local 0..N-1 indices LeRobot uses.

    Returns ``{}`` when ``info.json`` is missing or has no ``aid_info``.

    On internal collisions (the same ``episode_id`` mapping to two
    different ``episode_index`` values — usually corrupt meta), logs a
    warning and keeps the last-seen mapping.
    """
    info = _load_dataset_info(repo_id)
    if not info:
        return {}
    aid_info = info.get("aid_info") or {}
    mapping: dict[str, int] = {}
    for ep_idx_str, payload in aid_info.items():
        if not isinstance(payload, Mapping):
            continue
        eid = payload.get("episode_id")
        if eid is None:
            continue
        key = str(eid)
        ep_idx = int(ep_idx_str)
        if key in mapping and mapping[key] != ep_idx:
            logging.getLogger(__name__).warning(
                "load_aid_episode_id_mapping: collision in %s/meta/info.json — "
                "episode_id %r maps to both episode_index %d and %d; keeping %d.",
                repo_id, key, mapping[key], ep_idx, ep_idx,
            )
        mapping[key] = ep_idx
    return mapping


def resolve_episodes_by_aid(
    repo_id: str | Path,
    wanted_episode_ids: Sequence[Any],
    *,
    strict: bool = False,
) -> list[int]:
    """Translate external ``aid_info`` ``episode_id`` values into local
    ``episode_index`` integers used by LeRobot and this pipeline.

    Args:
        repo_id: Path to a repo whose ``meta/info.json`` carries
            ``aid_info``.
        wanted_episode_ids: External ``episode_id`` values (str or int).
            Internally normalized via ``str(...)``. Duplicates in the
            input are deduplicated in the output (preserving first-seen
            order).
        strict: When ``True``, raise ``KeyError`` if any requested
            ``episode_id`` is absent. When ``False`` (default), log a
            warning and drop missing IDs.

    Returns:
        List of local ``episode_index`` integers, deduplicated, in
        first-seen input order. Empty if all requested IDs are missing
        under ``strict=False``.
    """
    mapping = load_aid_episode_id_mapping(repo_id)
    resolved: list[int] = []
    seen: set[int] = set()
    missing: list[str] = []
    for raw in wanted_episode_ids:
        key = str(raw)
        idx = mapping.get(key)
        if idx is None:
            missing.append(key)
            continue
        if idx in seen:
            continue
        seen.add(idx)
        resolved.append(idx)

    if missing:
        if strict:
            raise KeyError(
                f"resolve_episodes_by_aid: {len(missing)} episode_id(s) not "
                f"found in {repo_id}/meta/info.json:aid_info — "
                f"{missing[:8]}{'...' if len(missing) > 8 else ''}"
            )
        logging.getLogger(__name__).warning(
            "resolve_episodes_by_aid: dropped %d unknown episode_id(s) for "
            "%s (first 8 shown): %s%s",
            len(missing), repo_id, missing[:8],
            "..." if len(missing) > 8 else "",
        )
    return resolved


__all__ = [
    "LeRobotDataset",
    "LeRobotDatasetMetadata",
    "get_episode_ids",
    "get_episode_index_bounds",
    "load_aid_episode_id_mapping",
    "resolve_episodes_by_aid",
]
