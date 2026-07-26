from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tau0_vla.data.config import get_config
from tau0_vla.data.source import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    _load_dataset_fps,
    _load_dataset_info,
    _load_field_descriptions,
    _require_backend,
)


@dataclass
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray | None = None
    q99: np.ndarray | None = None


FORMAT_VERSION = 1
PER_EMBODIMENT_FORMAT_VERSION = 2
FINGERPRINT_VERSION = 1


def _stats_to_dict(value: NormStats) -> dict[str, Any]:
    return {
        "mean": value.mean.tolist(),
        "std": value.std.tolist(),
        "q01": None if value.q01 is None else value.q01.tolist(),
        "q99": None if value.q99 is None else value.q99.tolist(),
    }


def serialize_json(
    norm_stats: dict[str, NormStats],
    *,
    config_summary: dict[str, Any] | None = None,
    config_fingerprint: str | None = None,
    per_embodiment: dict[str, dict[str, NormStats]] | None = None,
) -> str:
    """Serialize norm stats, optionally with a per-embodiment block.

    Without ``per_embodiment`` the output is the ``format_version: 1`` document
    every existing caller writes, byte for byte.

    With it, the output is ``format_version: 2``: the same global ``norm_stats``
    plus ``per_embodiment[registry_key][role]``, which is what
    :func:`load_file_with_per_embodiment` reads and what the unified route
    normalizes against — it looks its statistics up by the robot's
    ``_unified_registry_key`` in the 40D Unified Layout, not by component. Only
    the unified tooling in ``scripts/norm_stats/`` passes this; the component
    path cannot produce it, because it fits statistics over component vectors.
    """
    if per_embodiment is None:
        payload = {
            "format_version": FORMAT_VERSION,
            "config_summary": config_summary,
            "config_fingerprint": config_fingerprint,
            "norm_stats": {key: _stats_to_dict(value) for key, value in norm_stats.items()},
        }
        return json.dumps(payload, indent=2)
    payload = {
        "format_version": PER_EMBODIMENT_FORMAT_VERSION,
        "per_embodiment": {
            reg_key: {role: _stats_to_dict(value) for role, value in roles.items()}
            for reg_key, roles in per_embodiment.items()
        },
        "norm_stats": {key: _stats_to_dict(value) for key, value in norm_stats.items()},
        "config_summary": config_summary,
    }
    if config_fingerprint is not None:
        payload["config_fingerprint"] = config_fingerprint
    return json.dumps(payload, indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    payload = json.loads(data)["norm_stats"]
    return {
        key: NormStats(
            mean=np.asarray(value["mean"]),
            std=np.asarray(value["std"]),
            q01=None if value["q01"] is None else np.asarray(value["q01"]),
            q99=None if value["q99"] is None else np.asarray(value["q99"]),
        )
        for key, value in payload.items()
    }


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())


def load_file(path: pathlib.Path | str) -> dict[str, NormStats]:
    resolved = pathlib.Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {resolved}")
    return deserialize_json(resolved.read_text())


def load_file_with_per_embodiment(
    path: pathlib.Path | str,
) -> tuple[dict[str, NormStats], dict[str, dict[str, NormStats]]]:
    """Load norm stats JSON returning both global and per-embodiment stats.

    Returns (global_stats, per_embodiment) where per_embodiment is a dict
    keyed by registry_key → {role → NormStats}. Empty dict if format_version < 2.
    """
    resolved = pathlib.Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {resolved}")
    raw = json.loads(resolved.read_text())
    global_stats = {
        key: NormStats(
            mean=np.asarray(value["mean"]),
            std=np.asarray(value["std"]),
            q01=None if value["q01"] is None else np.asarray(value["q01"]),
            q99=None if value["q99"] is None else np.asarray(value["q99"]),
        )
        for key, value in raw["norm_stats"].items()
    }
    per_embodiment: dict[str, dict[str, NormStats]] = {}
    for reg_key, roles in raw.get("per_embodiment", {}).items():
        per_embodiment[reg_key] = {
            role: NormStats(
                mean=np.asarray(v["mean"]),
                std=np.asarray(v["std"]),
                q01=None if v.get("q01") is None else np.asarray(v["q01"]),
                q99=None if v.get("q99") is None else np.asarray(v["q99"]),
            )
            for role, v in roles.items()
        }
    return global_stats, per_embodiment


def build_robot_config_summary(
    *,
    repo_id: "str | Sequence[str]",
    state: Sequence[Any] | None,
    action: Sequence[Any] | None,
    action_horizon: int | None = None,
) -> dict[str, Any]:
    # Multi-repo configs fingerprint on the sorted list of basenames so the
    # stats file (a) can't collide with a different manifest that happens to
    # share its primary repo, and (b) is order-independent across manifest
    # reorderings.
    if isinstance(repo_id, str):
        repo_field: "str | list[str]" = pathlib.Path(repo_id).name or repo_id
    else:
        repo_field = sorted(pathlib.Path(r).name or r for r in repo_id)
    summary = {
        "fingerprint_version": FINGERPRINT_VERSION,
        "type": "RobotConfig",
        "repo_id": repo_field,
        "state": [_component_summary(component) for component in (state or ())],
        "action": [_component_summary(component) for component in (action or ())],
    }
    if action_horizon is not None:
        summary["action_horizon"] = int(action_horizon)
    # NOTE: the deleted activity-filter fields were only ever added to this
    # summary when filtering was ACTIVE, so removing them leaves the emitted
    # summary — and therefore the fingerprint — bit-identical for every config
    # that did not enable them. Existing norm_stats caches stay valid.
    return summary


def build_config_summary(config: Any) -> dict[str, Any]:
    if isinstance(config, str):
        from tau0_vla.data.config import get_config
        return build_config_summary(get_config(config))
    if dataclasses.is_dataclass(config) and hasattr(config, "robot_name") and hasattr(config, "repack"):
        if hasattr(config, "_repo_id_list"):
            repo_ids = config._repo_id_list()
            repo_arg: "str | list[str]" = repo_ids[0] if len(repo_ids) == 1 else repo_ids
        else:
            repo_arg = config.repo_id
        return build_robot_config_summary(
            repo_id=repo_arg,
            state=getattr(config, "state", None),
            action=getattr(config, "action", None),
            action_horizon=getattr(config, "action_horizon", None),
        )
    raise TypeError(f"Unsupported config type for norm stats summary: {type(config).__name__}")


def config_fingerprint(config_summary: dict[str, Any]) -> str:
    # ``repo_id`` identifies *which* manifest / dataset was sampled to compute
    # the stats — useful for traceability (kept in ``config_summary``), but
    # swapping the underlying manifest shouldn't invalidate otherwise-identical
    # norm stats. Strip it before hashing.
    fingerprint_input = _strip_fingerprint_exclusions(config_summary)
    canonical = json.dumps(fingerprint_input, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _strip_fingerprint_exclusions(summary: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in summary.items() if k != "repo_id"}


def _config_storage_stem(config: Any, config_summary: dict[str, Any]) -> str:
    config_name = getattr(config, "_finch_config_name", None)
    if config_name:
        return str(config_name)
    return config_fingerprint(config_summary)[:16]


def save_for_config(directory: pathlib.Path | str, config: Any, norm_stats: dict[str, NormStats]) -> pathlib.Path:
    summary = build_config_summary(config)
    fingerprint = config_fingerprint(summary)
    path = norm_stats_path_for_config(directory, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats, config_summary=summary, config_fingerprint=fingerprint))
    return path


def norm_stats_path_for_config(directory: pathlib.Path | str, config: Any) -> pathlib.Path:
    summary = build_config_summary(config)
    return pathlib.Path(directory) / f"{_config_storage_stem(config, summary)}.json"


def _estimate_total_frames(config: Any) -> int | None:
    # Multi-repo / manifest configs: sum each child's ``total_frames``.
    # Any child without ``meta/info.json`` disqualifies the estimate, because a
    # partial sum would silently understate the corpus and skew the default
    # sampling ratio.
    if hasattr(config, "_repo_id_list"):
        repos = config._repo_id_list()
    else:
        repo_id = getattr(config, "repo_id", None)
        if not repo_id:
            return None
        repos = [repo_id]
    total = 0
    for repo in repos:
        info_path = pathlib.Path(repo) / "meta" / "info.json"
        if not info_path.exists():
            return None
        try:
            value = json.loads(info_path.read_text()).get("total_frames")
        except Exception:
            return None
        if not value:
            return None
        total += int(value)
    return total if total else None


def _auto_compute_and_save_norm_stats(directory: pathlib.Path | str, config: Any, expected_path: pathlib.Path) -> None:
    total_frames = _estimate_total_frames(config)
    frames_note = f" ({total_frames} frames)" if total_frames else ""
    print(
        f"[tau0-vla] norm_stats missing at {expected_path}; auto-computing over full dataset{frames_note}...",
        flush=True,
    )
    stats = compute_norm_stats_for_config(config, max_samples=None)
    saved = save_for_config(directory, config, stats)
    print(f"[tau0-vla] norm_stats saved to {saved}", flush=True)


def load_dir_for_summary(
    directory: pathlib.Path | str,
    config: Any,
    *,
    overwrite: bool = False,
) -> dict[str, NormStats]:
    """Load (and if missing, auto-compute + save) norm stats bound to ``config``.

    The file name is derived from the config's registered name (or a SHA256
    fingerprint prefix as a fallback); the fingerprint baked into the JSON is
    verified against what the current ``config`` would produce.

    When ``overwrite=True`` and an existing JSON's fingerprint disagrees with
    the current config, the file is recomputed and overwritten instead of
    raising ``ValueError``. Useful when you've intentionally changed the
    config (state/action components, action_horizon) and
    want to refresh the cached stats.
    """
    config_summary = build_config_summary(config)
    fingerprint = config_fingerprint(config_summary)
    path = norm_stats_path_for_config(directory, config)
    if not path.exists():
        _auto_compute_and_save_norm_stats(directory, config, path)
    payload = json.loads(path.read_text())
    if (
        payload.get("format_version") != FORMAT_VERSION
        or payload.get("config_summary") is None
        or payload.get("config_fingerprint") is None
    ):
        raise ValueError(
            _regenerate_hint(
                reason="Norm stats file missing Finch metadata",
                path=path,
                config_summary=config_summary,
            )
        )
    if payload["config_fingerprint"] != fingerprint:
        if overwrite:
            print(
                f"[tau0-vla] norm stats fingerprint mismatch at {path}; overwrite=True → recomputing and overwriting.",
                file=sys.stderr,
            )
            _auto_compute_and_save_norm_stats(directory, config, path)
            payload = json.loads(path.read_text())
        else:
            raise ValueError(
                _regenerate_hint(
                    reason="Norm stats fingerprint mismatch",
                    path=path,
                    config_summary=config_summary,
                )
            )
    return deserialize_json(json.dumps({"norm_stats": payload["norm_stats"]}))


def normalize_array(values: np.ndarray, stats: NormStats, *, use_quantiles: bool = False) -> np.ndarray:
    if use_quantiles:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("Quantile normalization requires q01 and q99 statistics")
        q01 = stats.q01[..., : values.shape[-1]]
        q99 = stats.q99[..., : values.shape[-1]]
        return (values - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    mean = stats.mean[..., : values.shape[-1]]
    std = stats.std[..., : values.shape[-1]]
    return (values - mean) / (std + 1e-6)


def unnormalize_array(values: np.ndarray, stats: NormStats, *, use_quantiles: bool = False) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if use_quantiles:
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("Quantile normalization requested but q01/q99 stats are missing")
        return ((values + 1.0) / 2.0 * (stats.q99 - stats.q01 + 1e-6) + stats.q01).astype(np.float32)
    return (values * (stats.std + 1e-6) + stats.mean).astype(np.float32)


def normalize_masked(
    values: np.ndarray,
    stats: NormStats,
    *,
    mask: np.ndarray | None = None,
    std_floor: float,
) -> np.ndarray:
    """Mean/std normalize with a hard std floor and an optional post-multiply mask.

    The cross-embodiment unified 40D layout needs two deltas over plain
    ``normalize_array``: (1) a hard ``std_floor`` clamp (``max(std, floor)``)
    instead of the ``+1e-6`` epsilon — most unified slots are zero-padded for a
    given embodiment, so their std≈0 and a floor keeps those (already
    mask-zeroed) dims from exploding; (2) a per-slot validity ``mask`` that
    zeroes inactive slots after the division. ``std_floor`` is required (no
    default) because clamping to a non-trivial floor changes results for any
    real dim with ``std < floor`` — callers must opt in explicitly.

    Lives here so unified normalization shares one implementation with the rest
    of the pipeline rather than reimplementing the math.
    """
    values = np.asarray(values, dtype=np.float32)
    # Cast stats to float32 BEFORE the arithmetic so the intermediate stays in
    # float32 (NormStats holds float64 when loaded from JSON). This matters for
    # bit-level reproducibility of the unified path, whose stored norm artifacts
    # were produced under float32 normalization.
    mean = np.asarray(stats.mean, dtype=np.float32)[..., : values.shape[-1]]
    std = np.maximum(np.asarray(stats.std, dtype=np.float32)[..., : values.shape[-1]], std_floor)
    out = (values - mean) / std
    if mask is not None:
        out = out * mask
    return out.astype(np.float32)


def unnormalize_masked(
    values: np.ndarray,
    stats: NormStats,
    *,
    mask: np.ndarray | None = None,
    std_floor: float,
) -> np.ndarray:
    """Exact inverse of :func:`normalize_masked` — ``values * max(std, floor) + mean``,
    then re-zero inactive slots via ``mask``. Same ``std_floor`` the forward used
    (must match, or active dims with ``std < floor`` won't round-trip)."""
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(stats.mean, dtype=np.float32)[..., : values.shape[-1]]
    std = np.maximum(np.asarray(stats.std, dtype=np.float32)[..., : values.shape[-1]], std_floor)
    out = values * std + mean
    if mask is not None:
        out = out * mask
    return out.astype(np.float32)


@dataclass
class RunningStats:
    total_weight: float = 0.0
    total_sum: np.ndarray | None = None
    total_sum_sq: np.ndarray | None = None
    _reservoir: list[np.ndarray] | None = None
    _reservoir_capacity: int = 100_000
    _reservoir_count: int = 0
    _collect_quantiles: bool = False

    def enable_quantiles(self, reservoir_capacity: int = 100_000) -> None:
        self._collect_quantiles = True
        self._reservoir_capacity = reservoir_capacity
        self._reservoir = []
        self._reservoir_count = 0

    def update(self, values: np.ndarray, *, weight: float = 1.0) -> None:
        values = np.asarray(values)
        values = values.reshape(-1, values.shape[-1])
        if self.total_sum is None:
            self.total_sum = np.zeros(values.shape[-1], dtype=np.float64)
            self.total_sum_sq = np.zeros(values.shape[-1], dtype=np.float64)
        # Keep ``values`` at its native dtype (usually float32) to avoid the
        # 2x-memory float64 copy; the ``dtype=np.float64`` kwargs promote the
        # accumulator so numerical precision is preserved. ``einsum`` fuses
        # square-and-sum into a single pass without a 1.97M-row temporary.
        self.total_sum += values.sum(axis=0, dtype=np.float64) * weight
        self.total_sum_sq += np.einsum("ij,ij->j", values, values, dtype=np.float64) * weight
        self.total_weight += values.shape[0] * weight

        if self._collect_quantiles and self._reservoir is not None:
            self._update_reservoir(values)

    def _update_reservoir(self, values: np.ndarray) -> None:
        n = values.shape[0]
        current_size = sum(arr.shape[0] for arr in self._reservoir)
        if current_size < self._reservoir_capacity:
            take = min(n, self._reservoir_capacity - current_size)
            self._reservoir.append(values[:take].copy())
            self._reservoir_count += take
            values = values[take:]
            n = values.shape[0]
            if n == 0:
                return
        # Vectorized reservoir sampling for the overflow portion
        if len(self._reservoir) > 1:
            self._reservoir = [np.concatenate(self._reservoir, axis=0)]
        merged = self._reservoir[0]
        indices = np.arange(self._reservoir_count + 1, self._reservoir_count + n + 1)
        rand = np.random.randint(0, indices, dtype=np.int64)
        replace_mask = rand < self._reservoir_capacity
        if np.any(replace_mask):
            replace_positions = rand[replace_mask]
            replace_values = values[replace_mask]
            merged[replace_positions] = replace_values
        self._reservoir_count += n

    def finalize(self) -> NormStats:
        if self.total_sum is None or self.total_sum_sq is None or self.total_weight == 0:
            raise ValueError("No samples were collected for normalization statistics")
        mean = self.total_sum / self.total_weight
        variance = self.total_sum_sq / self.total_weight - np.square(mean)
        std = np.sqrt(np.maximum(variance, 0.0))
        q01 = None
        q99 = None
        if self._collect_quantiles and self._reservoir:
            merged = np.concatenate(self._reservoir, axis=0)
            q01 = np.percentile(merged, 1, axis=0)
            q99 = np.percentile(merged, 99, axis=0)
        return NormStats(mean=mean, std=std, q01=q01, q99=q99)

    def merge(self, other: "RunningStats") -> None:
        """Fold ``other``'s accumulated state into ``self`` (Welford parallel merge).

        ``total_sum`` / ``total_sum_sq`` / ``total_weight`` are sufficient
        statistics that compose additively, so summing them is algebraically
        identical to a single-process pass over the union of the two inputs —
        the worker shards therefore produce bit-stable mean/std regardless of
        ``num_workers``.

        Reservoirs are merged by weighted subsample: each row gets probability
        proportional to its source's ``rows_seen / reservoir_rows_kept``, which
        preserves the "uniform sample of the union" property across shards of
        arbitrary relative size. (Exact equality vs. a single-process run is
        not guaranteed here — reservoir sampling is already stochastic.)
        """
        if other.total_weight == 0:
            return
        if self.total_sum is None:
            self.total_sum = None if other.total_sum is None else other.total_sum.copy()
            self.total_sum_sq = None if other.total_sum_sq is None else other.total_sum_sq.copy()
        else:
            if other.total_sum is not None:
                self.total_sum += other.total_sum
            if other.total_sum_sq is not None:
                self.total_sum_sq += other.total_sum_sq
        self.total_weight += other.total_weight

        if not self._collect_quantiles or self._reservoir is None:
            return
        if not other._reservoir or other._reservoir_count == 0:
            return

        n_self = sum(arr.shape[0] for arr in self._reservoir)
        n_other = sum(arr.shape[0] for arr in other._reservoir)
        N_self = self._reservoir_count
        N_other = other._reservoir_count

        all_chunks = list(self._reservoir) + list(other._reservoir)
        pooled = np.concatenate(all_chunks, axis=0) if len(all_chunks) > 1 else all_chunks[0]
        pooled_size = pooled.shape[0]
        self._reservoir_count = N_self + N_other

        if pooled_size <= self._reservoir_capacity:
            self._reservoir = [pooled if pooled is all_chunks[0] else pooled.copy()]
            return

        w_self = (N_self / n_self) if n_self else 0.0
        w_other = (N_other / n_other) if n_other else 0.0
        weights = np.concatenate(
            [
                np.full(n_self, w_self, dtype=np.float64),
                np.full(n_other, w_other, dtype=np.float64),
            ]
        )
        total_w = weights.sum()
        if total_w <= 0:
            self._reservoir = [pooled[: self._reservoir_capacity].copy()]
            return
        weights /= total_w
        rng = np.random.default_rng(seed=int(N_self + N_other) & 0xFFFFFFFF)
        idx = rng.choice(pooled_size, size=self._reservoir_capacity, replace=False, p=weights)
        idx.sort()
        self._reservoir = [pooled[idx].copy()]


def save_norm_stats(
    config_name: str,
    *,
    output_dir: str | Path | None = None,
    max_samples: int | None = None,
    max_ratio: float | None = None,
    include_video: bool = False,
    compute_quantiles: bool = True,
    num_workers: int = 1,
) -> Path:
    """Compute norm stats for a registered config and persist them.

    ``max_samples`` / ``max_ratio`` are mutually exclusive:
      - ``max_samples=N`` — hard cap on total rows sampled across all repos
        (allocated proportionally to each repo's size).
      - ``max_ratio=R`` — sample ``R × sum(total_frames)`` rows, resolved by
        reading each repo's ``meta/info.json`` up front. Useful when the
        corpus grows (e.g. switching manifest from 2k → 5k) and you want
        stats coverage to scale with it without re-tuning ``max_samples``.

    ``num_workers > 1`` parallelises the fast-path stats compute across
    worker processes. Default ``1`` keeps the behaviour identical to the
    single-process path — every existing caller is unaffected.

    Returns the path to the written JSON file.
    """
    config = get_config(config_name)
    stats = compute_norm_stats_for_config(
        config,
        max_samples=max_samples,
        max_ratio=max_ratio,
        include_video=include_video,
        compute_quantiles=compute_quantiles,
        num_workers=num_workers,
    )
    return save_for_config(_resolve_norm_stats_output_dir(config, output_dir), config, stats)


def compute_norm_stats_for_config(
    config: Any,
    *,
    max_samples: int | None = None,
    max_ratio: float | None = None,
    include_video: bool = False,
    compute_quantiles: bool = True,
    num_workers: int = 1,
) -> dict[str, NormStats]:
    if max_samples is not None and max_ratio is not None:
        raise ValueError("Pass at most one of max_samples / max_ratio, not both.")
    fast_stats = _try_compute_norm_stats_fast(
        config,
        max_samples=max_samples,
        max_ratio=max_ratio,
        include_video=include_video,
        compute_quantiles=compute_quantiles,
        num_workers=num_workers,
    )
    if fast_stats is not None:
        return fast_stats

    # Fast path bailed — slow row-wise path doesn't support frame filtering.
    # Surface this clearly rather than silently producing stats over the full
    # (unfiltered) distribution.
    frame_policy = config.frame_filter_policy() if hasattr(config, "frame_filter_policy") else None
    if frame_policy is not None:
        raise RuntimeError(
            f"compute_norm_stats_for_config: frame filtering (frame_filter={frame_policy!r}) is only supported on "
            "the hf fast path. Check why the fast path is unavailable (config has source_kwargs / "
            "include_video=True / non-RobotConfig class) and re-enable it."
        )

    pipelines, weights = _resolve_pipelines_and_weights(config, include_video=include_video)
    state_stats = RunningStats()
    action_stats = RunningStats()
    if compute_quantiles:
        state_stats.enable_quantiles()
        action_stats.enable_quantiles()
    sample_limits = _allocate_sample_limits(pipelines, weights, max_samples)

    for pipeline, weight, sample_limit in zip(pipelines, weights, sample_limits, strict=True):
        transforms = _transforms_without_normalize(pipeline)
        total = len(pipeline.dataset) if sample_limit is None else min(sample_limit, len(pipeline.dataset))
        print(
            f"[norm-stats] dataset={getattr(pipeline, 'name', 'dataset')} total_samples={total} "
            f"include_video={include_video}",
            flush=True,
        )
        started_at = time.time()
        for index in range(total):
            sample = pipeline.dataset[index]
            for transform in transforms:
                sample = transform(sample)
            state = sample.get("state")
            action = sample.get("action")
            if state is not None:
                state_stats.update(np.asarray(state), weight=weight)
            if action is not None:
                action_stats.update(np.asarray(action), weight=weight)
            if index == 0 or (index + 1) % 1000 == 0 or index + 1 == total:
                elapsed = time.time() - started_at
                processed = index + 1
                rate = processed / elapsed if elapsed > 0 else 0.0
                remaining = (total - processed) / rate if rate > 0 else float("inf")
                print(
                    f"[norm-stats] processed={processed}/{total} "
                    f"elapsed={elapsed:.1f}s rate={rate:.1f} samples/s remaining={remaining:.1f}s",
                    flush=True,
                )

    return {
        "state": state_stats.finalize(),
        "action": action_stats.finalize(),
    }


def _try_compute_norm_stats_fast(
    config: Any,
    *,
    max_samples: int | None,
    max_ratio: float | None,
    include_video: bool,
    compute_quantiles: bool = True,
    num_workers: int = 1,
) -> dict[str, NormStats] | None:
    if include_video:
        return None
    if not (dataclasses.is_dataclass(config) and hasattr(config, "robot_name") and hasattr(config, "repack")):
        return None
    source_kwargs = getattr(config, "source_kwargs", None) or {}
    if source_kwargs:
        # Multi-frame reads need explicit alignment; the fast single-frame path does not apply here.
        return None
    # Multi-repo configs (list[str] / ``.txt`` manifest) are supported by
    # iterating each child's parquet shards under one shared RunningStats pair
    # — see ``_compute_norm_stats_from_hf_dataset``.
    try:
        return _compute_norm_stats_from_hf_dataset(
            config,
            max_samples=max_samples,
            max_ratio=max_ratio,
            compute_quantiles=compute_quantiles,
            num_workers=num_workers,
        )
    except Exception as exc:
        print(
            f"[norm-stats] hf fast path unavailable, falling back to row-wise path: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


def _compute_norm_stats_from_hf_dataset(
    config: Any,
    *,
    max_samples: int | None,
    max_ratio: float | None = None,
    compute_quantiles: bool = True,
    num_workers: int = 1,
) -> dict[str, NormStats]:
    _require_backend()
    from tau0_vla.data.modalities.base import WorkflowContext

    repos = config._repo_id_list() if hasattr(config, "_repo_id_list") else [config.repo_id]

    # All children in a multi-repo config share the same modality schema
    # (``_repo_id_list`` is what distinguishes this config from a sibling),
    # so we load field descriptions and resolve components once up front.
    field_descriptions = _load_field_descriptions(repos[0])
    if not field_descriptions:
        raise ValueError("hf fast path only supports local datasets with meta/info.json")
    state_descriptions, action_descriptions = config._normalize_field_descriptions(field_descriptions)
    state_map, action_map, state_components, action_components = config._resolve_component_bundle()
    action_horizon = int(getattr(config, "action_horizon", None) or 1)
    frame_policy = config.frame_filter_policy() if hasattr(config, "frame_filter_policy") else None

    state_stats = RunningStats()
    action_stats = RunningStats()
    if compute_quantiles:
        state_stats.enable_quantiles()
        action_stats.enable_quantiles()

    # Resolve ``max_ratio`` → ``max_samples`` by summing each repo's
    # ``meta/info.json`` frame count. Keeps allocation consistent with the
    # ``max_samples`` code path: ``_allocate_repo_sample_budgets`` still
    # proportionally splits the cap across repos.
    if max_ratio is not None:
        if not (0.0 < max_ratio <= 1.0):
            raise ValueError(f"max_ratio must be in (0, 1], got {max_ratio}")
        total_frames = 0
        for rid in repos:
            info = _load_dataset_info(rid)
            total = None if info is None else info.get("total_frames")
            if not total:
                raise ValueError(f"max_ratio requires meta/info.json['total_frames'] for every repo; missing for {rid}")
            total_frames += int(total)
        resolved = max(1, int(total_frames * max_ratio))
        print(
            f"[norm-stats] max_ratio={max_ratio} → max_samples={resolved} "
            f"(total_frames={total_frames} across {len(repos)} repos)",
            flush=True,
        )
        max_samples = resolved

    budgets = _allocate_repo_sample_budgets(repos, max_samples)

    if num_workers > 1 and len(repos) > 1:
        # Shared pool across ALL repos — each worker processes one repo end-to-end.
        # Avoids the per-repo pool-spawn anti-pattern (N repos × K workers =
        # N*K interpreter/module warm-ups); here we get K total.
        _accumulate_hf_repos_parallel(
            config=config,
            repos=list(repos),
            budgets=list(budgets),
            action_horizon=action_horizon,
            num_workers=num_workers,
            compute_quantiles=compute_quantiles,
            state_stats=state_stats,
            action_stats=action_stats,
        )
    else:
        for repo_index, (repo_id, budget) in enumerate(zip(repos, budgets, strict=True)):
            _accumulate_hf_repo_into_running_stats(
                repo_id=repo_id,
                repo_index=repo_index,
                repo_total=len(repos),
                budget=budget,
                action_horizon=action_horizon,
                state_descriptions=state_descriptions,
                action_descriptions=action_descriptions,
                state_components=state_components,
                action_components=action_components,
                state_map=state_map,
                action_map=action_map,
                state_stats=state_stats,
                action_stats=action_stats,
                workflow_context_cls=WorkflowContext,
                augment_fn=getattr(config, "augment_raw_tensors", None),
                num_workers=num_workers,
                config=config,
                compute_quantiles=compute_quantiles,
                frame_filter_policy=frame_policy,
            )

    return {
        "state": state_stats.finalize(),
        "action": action_stats.finalize(),
    }


def _allocate_repo_sample_budgets(repos: Sequence[str], max_samples: int | None) -> list[int | None]:
    # Proportional allocation so a 100k-frame repo doesn't get the same budget
    # as a 10M-frame one — otherwise pooled mean/std would overweight the
    # small repo by 100x relative to its share of the corpus.
    if max_samples is None:
        return [None] * len(repos)
    lengths: list[int] = []
    for rid in repos:
        info = _load_dataset_info(rid)
        total = None if info is None else info.get("total_frames")
        lengths.append(int(total) if total else 1)
    total_len = sum(lengths) or len(repos)
    raw = [max_samples * (L / total_len) for L in lengths]
    budgets = [max(1, int(value)) for value in raw]
    while sum(budgets) < max_samples:
        for index in np.argsort([value - int(value) for value in raw])[::-1]:
            budgets[int(index)] += 1
            if sum(budgets) == max_samples:
                break
    while sum(budgets) > max_samples:
        for index in np.argsort([value - int(value) for value in raw]):
            if budgets[int(index)] > 1:
                budgets[int(index)] -= 1
            if sum(budgets) == max_samples:
                break
    return budgets


def _stats_chunk_starts(length: int, budget: int | None, chunk_size: int = 4096) -> np.ndarray:
    # Chunk-level uniform stride: each chunk reads CONTIGUOUSLY (fast on HF
    # parquet row groups), but the chunk anchors are spread across the whole
    # repo so we don't just sample the earliest episodes. Row-level striding
    # kills throughput because it forces a seek per index.
    if budget is None or budget >= length:
        return np.arange(0, length, chunk_size, dtype=np.int64)
    num_chunks = max(1, (int(budget) + chunk_size - 1) // chunk_size)
    max_start = max(0, length - chunk_size)
    if num_chunks <= 1 or max_start == 0:
        return np.array([0], dtype=np.int64)
    return np.unique(np.linspace(0, max_start, num=num_chunks, dtype=np.int64))


def _iter_hf_repo_module_windows(
    *,
    repo_id: str,
    action_horizon: int,
    state_descriptions: Mapping[str, Any],
    action_descriptions: Mapping[str, Any],
    state_components: Sequence[Any],
    action_components: Sequence[Any],
    state_map: Mapping[str, Any],
    action_map: Mapping[str, Any],
    chunk_starts: np.ndarray | None = None,
    chunk_size: int = 4096,
    yield_state: bool = True,
    augment_fn: "Callable[..., tuple[np.ndarray, np.ndarray]] | None" = None,
) -> Iterator[dict[str, Any]]:
    """Yield per-chunk projected module windows for ``repo_id``.

    Each yielded dict contains:
      - ``cs`` / ``ce`` (int)         — chunk half-open frame range.
      - ``length`` (int)              — total frames in the repo.
      - ``ep_from_arr`` / ``ep_to_arr`` (ndarray) — repo-global episode bounds
        (sorted). Both are always present (even for ``action_horizon <= 1``) so
        downstream consumers can bucket anchors by episode.
      - ``ep_idx_per_anchor`` (ndarray[B]) — episode index per anchor, clipped
        to the repo's episode count.
      - ``state_values`` (dict[str, ndarray[B, D]] | None) — per-module state
        projection; ``None`` when ``yield_state=False``.
      - ``action_values`` (dict[str, ndarray]) — per-module action projection.
        Shape is ``[B, D]`` when ``action_horizon <= 1`` and ``[B, H, D]``
        otherwise (mirrors ``_project_component_values`` / the existing stats
        path so stats consumers see identical input).
    """
    _require_backend()
    dataset = LeRobotDataset(repo_id=Path(repo_id).name, root=Path(repo_id))
    # ``with_format("numpy")`` routes parquet reads through Arrow's native
    # numpy bridge: the backing ``ListArray`` goes straight to a 2D ndarray
    # instead of the default Arrow → Python list → ``np.asarray`` double
    # conversion. For fixed-width state/action columns that's ~5x faster on
    # bulk slices.
    hf_dataset = dataset.hf_dataset.with_format("numpy")
    length = len(hf_dataset)

    # Episode bounds as sorted numpy arrays so we can locate every anchor's
    # episode in a single vectorized ``searchsorted`` call. Consumers need these
    # even when ``action_horizon <= 1``, so build unconditionally (cheap — just
    # two int64 arrays per repo).
    raw_episodes = dataset.meta.episodes
    if isinstance(raw_episodes, Mapping):
        ep_iter = list(raw_episodes.values())
    else:
        ep_iter = list(raw_episodes)
    if ep_iter:
        ep_from_arr = np.asarray([int(ep["dataset_from_index"]) for ep in ep_iter], dtype=np.int64)
        ep_to_arr = np.asarray([int(ep["dataset_to_index"]) for ep in ep_iter], dtype=np.int64)
        order = np.argsort(ep_from_arr)
        ep_from_arr = ep_from_arr[order]
        ep_to_arr = ep_to_arr[order]
    else:
        ep_from_arr = np.array([], dtype=np.int64)
        ep_to_arr = np.array([], dtype=np.int64)

    if chunk_starts is None:
        chunk_starts = np.arange(0, length, chunk_size, dtype=np.int64)
    else:
        chunk_starts = np.asarray(chunk_starts, dtype=np.int64)

    # Precompute the full chunk schedule so the prefetch thread can pipeline
    # I/O (the Arrow parquet reader releases the GIL, so a single-worker
    # background thread overlaps disk latency with the foreground compute).
    chunks: list[tuple[int, int, int]] = []
    for cs in chunk_starts:
        cs_int = int(cs)
        ce_int = min(cs_int + chunk_size, length)
        span_end = min(ce_int + action_horizon - 1, length) if action_horizon > 1 else ce_int
        chunks.append((cs_int, ce_int, span_end))

    if not chunks:
        return

    def _fetch(cs: int, ce: int, span_end: int) -> tuple[np.ndarray, np.ndarray]:
        state_raw = np.asarray(hf_dataset[cs:ce]["observation.state"], dtype=np.float32)
        if action_horizon <= 1:
            action_src = np.asarray(hf_dataset[cs:ce]["action"], dtype=np.float32)
        else:
            action_src = np.asarray(hf_dataset[cs:span_end]["action"], dtype=np.float32)
        return state_raw, action_src

    components_combined = tuple(state_components) + tuple(action_components)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="finch-module-windows-prefetch") as pool:
        pending = pool.submit(_fetch, *chunks[0])
        for i, (cs, ce, span_end) in enumerate(chunks):
            state_raw, action_src = pending.result()
            # Kick off the next read immediately so it overlaps with the
            # compute work we're about to do on this chunk.
            if i + 1 < len(chunks):
                pending = pool.submit(_fetch, *chunks[i + 1])

            # Config-level raw-tensor augmentation. We
            # run this at the SOURCE-frame layer — on the flat ``[chunk, D]``
            # ``state_raw`` and the flat ``[chunk+H-1, D]`` ``action_src`` —
            # BEFORE the per-anchor horizon window is gathered. The gather
            # below then propagates any appended columns into the windowed
            # ``action_raw`` view. If we augmented after the gather, per-frame
            # augmentation hooks would pay an H-fold cost because each source frame
            # appears in up to H overlapping action windows; doing it here
            # keeps augmentation's cost proportional to *unique* source
            # frames, which is the semantic unit. Identity by default.
            if augment_fn is not None:
                state_raw, action_src = augment_fn(
                    state_raw=state_raw,
                    action_raw=action_src,
                    state_descriptions=state_descriptions,
                    action_descriptions=action_descriptions,
                )

            anchors = np.arange(cs, ce, dtype=np.int64)
            if action_horizon <= 1:
                action_raw = action_src
            else:
                # ``action_src`` already spans [cs, span_end); we build per-anchor
                # horizon windows locally (clipped to the enclosing episode) so
                # we never list-index the parquet backend for actions.
                action_flat = action_src
                deltas = np.arange(action_horizon, dtype=np.int64)
                query = anchors[:, None] + deltas[None, :]  # (B, H) global indices
                if ep_from_arr.size and ep_to_arr.size:
                    # Each anchor's episode is ``ep_from_arr[j] <= anchor < ep_from_arr[j+1]``,
                    # which is exactly what ``searchsorted(..., side="right") - 1`` gives us.
                    ep_idx = np.searchsorted(ep_from_arr, anchors, side="right") - 1
                    ep_idx = np.clip(ep_idx, 0, ep_from_arr.shape[0] - 1)
                    lo = ep_from_arr[ep_idx][:, None]
                    hi = (ep_to_arr[ep_idx] - 1)[:, None]
                    query = np.clip(query, lo, hi)
                local_query = query - cs
                local_query = np.clip(local_query, 0, action_flat.shape[0] - 1)
                action_raw = action_flat[local_query].reshape(ce - cs, action_horizon, -1)

            if ep_from_arr.size:
                ep_idx_per_anchor = np.searchsorted(ep_from_arr, anchors, side="right") - 1
                ep_idx_per_anchor = np.clip(ep_idx_per_anchor, 0, ep_from_arr.shape[0] - 1)
            else:
                ep_idx_per_anchor = np.zeros(ce - cs, dtype=np.int64)

            if yield_state:
                state_values = _project_component_values(
                    descriptions=state_descriptions,
                    raw_values=state_raw,
                    components=components_combined,
                    field_map=state_map,
                    role="state",
                )
            else:
                state_values = None

            action_values = _project_component_values(
                descriptions=action_descriptions,
                raw_values=action_raw,
                components=components_combined,
                field_map=action_map,
                role="action",
            )

            yield {
                "cs": cs,
                "ce": ce,
                "length": length,
                "ep_from_arr": ep_from_arr,
                "ep_to_arr": ep_to_arr,
                "ep_idx_per_anchor": ep_idx_per_anchor,
                "state_values": state_values,
                "action_values": action_values,
            }


def _build_repo_frame_filter_valid_mask(
    *,
    repo_id: str,
    frame_filter_policy: Any,
    action_horizon: int,
    length: int,
) -> np.ndarray:
    """Build a per-repo ``[length]`` bool mask from the training FrameFilter policy."""
    from tau0_vla.data.sampler import build_segment_index_ranges

    dataset = LeRobotDataset(repo_id=Path(repo_id).name, root=Path(repo_id))
    meta = LeRobotDatasetMetadata(root=repo_id, repo_id=Path(repo_id).name)
    ranges, _ = build_segment_index_ranges(
        dataset_metas=[meta],
        datasets=[dataset],
        tail_buffer=max(0, int(action_horizon) - 1),
        positive_labels=frame_filter_policy.positive,
        negative_labels=frame_filter_policy.negative,
    )
    valid = np.zeros(length, dtype=bool)
    for start, end in np.asarray(ranges, dtype=np.int64):
        valid[int(start) : int(end)] = True
    return valid


def _accumulate_hf_repo_into_running_stats(
    *,
    repo_id: str,
    repo_index: int,
    repo_total: int,
    budget: int | None,
    action_horizon: int,
    state_descriptions: Mapping[str, Any],
    action_descriptions: Mapping[str, Any],
    state_components: Sequence[Any],
    action_components: Sequence[Any],
    state_map: Mapping[str, Any],
    action_map: Mapping[str, Any],
    state_stats: RunningStats,
    action_stats: RunningStats,
    workflow_context_cls: Any,
    augment_fn: "Callable[..., tuple[np.ndarray, np.ndarray]] | None" = None,
    num_workers: int = 1,  # accepted for signature-compat; parallel is orchestrated at the caller
    config: Any = None,
    compute_quantiles: bool = True,
    frame_filter_policy: Any = None,
) -> None:
    chunk_size = 4096
    # Length is required both for budget → chunk_starts planning and for the
    # progress logger's ``total`` denominator, so compute it once up front.
    dataset = LeRobotDataset(repo_id=Path(repo_id).name, root=Path(repo_id))
    length = len(dataset.hf_dataset)
    chunk_starts = _stats_chunk_starts(length, budget, chunk_size=chunk_size)
    total = int(sum(min(chunk_size, length - int(cs)) for cs in chunk_starts))

    filter_valid: np.ndarray | None = None
    if frame_filter_policy is not None:
        filter_valid = _build_repo_frame_filter_valid_mask(
            repo_id=repo_id,
            frame_filter_policy=frame_filter_policy,
            action_horizon=action_horizon,
            length=length,
        )
        print(
            f"[norm-stats] repo[{repo_index + 1}/{repo_total}] frame-filter "
            f"kept {int(filter_valid.sum())}/{length} anchors "
            f"({100.0 * filter_valid.sum() / max(length, 1):.2f}%) "
            f"positive={frame_filter_policy.positive} negative={frame_filter_policy.negative}",
            flush=True,
        )
    if filter_valid is not None and not filter_valid.any():
        raise ValueError(f"norm_stats filter kept 0 anchors for repo {repo_id}")

    print(
        f"[norm-stats] hf fast path repo[{repo_index + 1}/{repo_total}]={Path(repo_id).name} "
        f"total_samples={total} num_chunks={len(chunk_starts)} chunk_size={chunk_size}",
        flush=True,
    )
    started_at = time.time()
    processed = 0
    kept = 0

    for chunk in _iter_hf_repo_module_windows(
        repo_id=repo_id,
        action_horizon=action_horizon,
        state_descriptions=state_descriptions,
        action_descriptions=action_descriptions,
        state_components=state_components,
        action_components=action_components,
        state_map=state_map,
        action_map=action_map,
        chunk_starts=chunk_starts,
        chunk_size=chunk_size,
        yield_state=True,
        augment_fn=augment_fn,
    ):
        cs = chunk["cs"]
        ce = chunk["ce"]
        state_values = chunk["state_values"]
        action_values = chunk["action_values"]

        state_context = workflow_context_cls(state=state_values, action=action_values)
        action_state_values = dict(state_values)
        for key, state_value in state_values.items():
            action_value = action_values.get(key)
            if action_value is not None and action_value.ndim == state_value.ndim + 1:
                action_state_values[key] = np.expand_dims(state_value, axis=1)
        action_context = workflow_context_cls(state=action_state_values, action=action_values)

        state_batch = _assemble_component_batch(state_components, context=state_context, role="state")
        action_batch = _assemble_component_batch(action_components, context=action_context, role="action")
        if filter_valid is not None:
            keep = filter_valid[cs:ce]
            k = int(keep.sum())
            if k == 0:
                processed += ce - cs
                continue
            if k < keep.size:
                state_batch = state_batch[keep]
                action_batch = action_batch[keep]
            kept += k
        else:
            kept += ce - cs
        state_stats.update(state_batch)
        action_stats.update(action_batch)

        processed += ce - cs
        elapsed = time.time() - started_at
        rate = processed / elapsed if elapsed > 0 else 0.0
        remaining = (total - processed) / rate if rate > 0 else float("inf")
        kept_suffix = f" kept={kept}" if filter_valid is not None else ""
        print(
            f"[norm-stats] repo[{repo_index + 1}/{repo_total}] processed={processed}/{total}{kept_suffix} "
            f"elapsed={elapsed:.1f}s rate={rate:.1f} samples/s remaining={remaining:.1f}s",
            flush=True,
        )


def _stats_worker_shard(
    config: Any,
    repo_id: str,
    chunk_starts: np.ndarray,
    action_horizon: int,
    chunk_size: int,
    compute_quantiles: bool,
) -> tuple[RunningStats, RunningStats]:
    """Subprocess worker: accumulate local RunningStats over a shard of chunks.

    Runs the exact same per-chunk compute as the in-process path
    (``_iter_hf_repo_module_windows`` + ``_assemble_component_batch``) against
    a private ``RunningStats`` pair, then returns the pair so the main process
    can merge the Welford sufficient statistics.

    Descriptions / components are re-derived inside the worker — safer than
    shipping already-resolved objects whose picklability depends on arbitrary
    modality class layout. The small duplicated setup cost (one parquet
    ``info.json`` read, one component resolution) is absorbed by the
    per-shard compute budget.
    """
    _require_backend()
    from tau0_vla.data.modalities.base import WorkflowContext

    field_descriptions = _load_field_descriptions(repo_id)
    state_descriptions, action_descriptions = config._normalize_field_descriptions(field_descriptions)
    state_map, action_map, state_components, action_components = config._resolve_component_bundle()

    state_stats = RunningStats()
    action_stats = RunningStats()
    if compute_quantiles:
        state_stats.enable_quantiles()
        action_stats.enable_quantiles()

    frame_policy = config.frame_filter_policy() if hasattr(config, "frame_filter_policy") else None
    filter_valid: np.ndarray | None = None
    if frame_policy is not None:
        dataset = LeRobotDataset(repo_id=Path(repo_id).name, root=Path(repo_id))
        length = len(dataset.hf_dataset)
        filter_valid = _build_repo_frame_filter_valid_mask(
            repo_id=repo_id,
            frame_filter_policy=frame_policy,
            action_horizon=action_horizon,
            length=length,
        )
    if filter_valid is not None and not filter_valid.any():
        raise ValueError(f"norm_stats filter kept 0 anchors for repo {repo_id}")

    for chunk in _iter_hf_repo_module_windows(
        repo_id=repo_id,
        action_horizon=action_horizon,
        state_descriptions=state_descriptions,
        action_descriptions=action_descriptions,
        state_components=state_components,
        action_components=action_components,
        state_map=state_map,
        action_map=action_map,
        chunk_starts=chunk_starts,
        chunk_size=chunk_size,
        yield_state=True,
        augment_fn=getattr(config, "augment_raw_tensors", None),
    ):
        state_values = chunk["state_values"]
        action_values = chunk["action_values"]

        state_context = WorkflowContext(state=state_values, action=action_values)
        action_state_values = dict(state_values)
        for key, state_value in state_values.items():
            action_value = action_values.get(key)
            if action_value is not None and action_value.ndim == state_value.ndim + 1:
                action_state_values[key] = np.expand_dims(state_value, axis=1)
        action_context = WorkflowContext(state=action_state_values, action=action_values)

        state_batch = _assemble_component_batch(state_components, context=state_context, role="state")
        action_batch = _assemble_component_batch(action_components, context=action_context, role="action")
        if filter_valid is not None:
            cs = chunk["cs"]
            ce = chunk["ce"]
            keep = filter_valid[cs:ce]
            k = int(keep.sum())
            if k == 0:
                continue
            if k < keep.size:
                state_batch = state_batch[keep]
                action_batch = action_batch[keep]
        state_stats.update(state_batch)
        action_stats.update(action_batch)

    return state_stats, action_stats


def _accumulate_hf_repos_parallel(
    *,
    config: Any,
    repos: list[str],
    budgets: list[int | None],
    action_horizon: int,
    num_workers: int,
    compute_quantiles: bool,
    state_stats: RunningStats,
    action_stats: RunningStats,
    chunk_size: int = 4096,
) -> None:
    """Share one ``ProcessPoolExecutor`` across the WHOLE manifest.

    Each task = one repo, so heavy per-worker startup (interpreter warm-up and
    the pipeline import graph) runs at most
    ``num_workers`` times TOTAL — not once per repo. For a 127-repo manifest
    at ``--workers 8`` that's 8 warm-ups vs. the ~1000 a per-repo pool
    would incur.

    Repo-level granularity (vs. chunk-level) is the right knob here because
    a typical manifest has 10²–10³ repos — plenty to keep 8–32 workers busy.
    Tasks complete in episode-time rather than chunk-time (~tens of seconds
    each), so scheduling overhead is irrelevant.

    Workers return private ``RunningStats`` pairs that main merges via
    ``RunningStats.merge``; mean/std compose from Welford sufficient
    statistics so the result is bit-stable w.r.t. worker count.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import get_context

    # Precompute chunk_starts in the main process (needs parquet length).
    # Done serially here because each call is a fast ``LeRobotDataset`` open
    # + one int read — overlapping them via threads isn't worth the code.
    tasks: list[tuple[str, np.ndarray, int, int]] = []  # (repo_id, chunk_starts, total_samples, repo_index)
    grand_total = 0
    for repo_index, (repo_id, budget) in enumerate(zip(repos, budgets, strict=True)):
        dataset = LeRobotDataset(repo_id=Path(repo_id).name, root=Path(repo_id))
        length = len(dataset.hf_dataset)
        chunk_starts = _stats_chunk_starts(length, budget, chunk_size=chunk_size)
        total = int(sum(min(chunk_size, length - int(cs)) for cs in chunk_starts))
        grand_total += total
        tasks.append((repo_id, chunk_starts, total, repo_index))

    effective_workers = min(int(num_workers), len(tasks))
    print(
        f"[norm-stats] parallel fast path: {len(tasks)} repos, "
        f"{grand_total} total samples, workers={effective_workers}",
        flush=True,
    )
    started_at = time.time()

    mp_ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=effective_workers, mp_context=mp_ctx) as pool:
        future_to_task = {
            pool.submit(
                _stats_worker_shard,
                config,
                repo_id,
                chunk_starts,
                action_horizon,
                chunk_size,
                compute_quantiles,
            ): (repo_id, total, repo_index)
            for repo_id, chunk_starts, total, repo_index in tasks
        }
        completed = 0
        done_samples = 0
        for future in as_completed(future_to_task):
            repo_id, total, repo_index = future_to_task[future]
            worker_state, worker_action = future.result()
            state_stats.merge(worker_state)
            action_stats.merge(worker_action)
            completed += 1
            done_samples += total
            elapsed = time.time() - started_at
            # ETA must be repo-based, not sample-based: 8 workers process 8
            # repos in parallel, but ``done_samples`` only accounts for the
            # COMPLETED ones — the 8 in-flight repos don't contribute until
            # they finish, so a naïve samples/elapsed rate understates real
            # throughput by the parallel factor. Wall-clock per completed
            # repo × remaining repos is the right extrapolation.
            eta = (elapsed / completed) * (len(tasks) - completed) if completed else float("inf")
            rate = done_samples / elapsed if elapsed > 0 else 0.0
            print(
                f"[norm-stats] done repo[{repo_index + 1}/{len(tasks)}]={Path(repo_id).name} "
                f"samples={total} ({completed}/{len(tasks)} repos) "
                f"elapsed={elapsed:.1f}s rate={rate:.1f} samples/s eta={eta:.1f}s",
                flush=True,
            )


def _project_component_values(
    *,
    descriptions: Mapping[str, Any],
    raw_values: np.ndarray,
    components: Sequence[Any],
    field_map: Mapping[str, str | list[str]],
    role: str,
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    prefix = f"{role}/"
    for component in components:
        if component.key in values:
            continue
        field_key = component.field_key
        synthetic_field_key = f"{role}/{component.key}"
        if (
            field_key is None
            or field_key == synthetic_field_key
            or not _has_description(descriptions, field_key, prefix)
        ):
            # Component may be single-role (e.g. chassis_velocity lives only on
            # the action side). When resolving values for the opposite role
            # there's no field_map entry, so skip rather than KeyError.
            if component.key not in field_map:
                continue
            field_key = field_map[component.key]
        if isinstance(field_key, list):
            projected = []
            for key in field_key:
                description = _lookup_description(descriptions, key, prefix)
                try:
                    value = description.project(raw_values)
                except (IndexError, ValueError) as exc:
                    raise ValueError(
                        f"Failed to project {role} component {component.key!r} from field {key!r}: "
                        f"raw_values.shape={tuple(raw_values.shape)}, indices={getattr(description, 'indices', None)!r}"
                    ) from exc
                projected.append(np.asarray(value, dtype=np.float32))
            values[component.key] = np.concatenate(projected, axis=-1)
            continue
        description = _lookup_description(descriptions, field_key, prefix)
        try:
            value = description.project(raw_values)
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"Failed to project {role} component {component.key!r} from field {field_key!r}: "
                f"raw_values.shape={tuple(raw_values.shape)}, indices={getattr(description, 'indices', None)!r}"
            ) from exc
        values[component.key] = np.asarray(value, dtype=np.float32)
    return values


def _lookup_description(descriptions: Mapping[str, Any], key: str, prefix: str) -> Any:
    if key in descriptions:
        return descriptions[key]
    stripped = key[len(prefix) :] if key.startswith(prefix) else key
    if stripped in descriptions:
        return descriptions[stripped]
    raise KeyError(stripped)


def _has_description(descriptions: Mapping[str, Any], key: str, prefix: str) -> bool:
    if key in descriptions:
        return True
    stripped = key[len(prefix) :] if key.startswith(prefix) else key
    return stripped in descriptions


def _assemble_component_batch(components: Sequence[Any], *, context: Any, role: str) -> np.ndarray:
    batches: list[np.ndarray] = []
    for component in components:
        batches.append(np.asarray(component.transform(context, role=role, mode="pre"), dtype=np.float32))
    if not batches:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(batches, axis=-1) if len(batches) > 1 else batches[0]


def _resolve_pipelines_and_weights(config: Any, *, include_video: bool) -> tuple[list[Any], list[float]]:
    if dataclasses.is_dataclass(config) and hasattr(config, "robot_name") and hasattr(config, "repack"):
        return [_build_pipeline_for_stats(config, include_video=include_video)], [1.0]
    return [config.build_pipeline()], [1.0]


def _allocate_sample_limits(pipelines: list[Any], weights: list[float], max_samples: int | None) -> list[int | None]:
    if max_samples is None:
        return [None] * len(pipelines)
    total_weight = sum(weights)
    raw_limits = [max_samples * (weight / total_weight) for weight in weights]
    limits = [max(1, int(value)) for value in raw_limits]
    while sum(limits) < max_samples:
        for index in np.argsort([value - int(value) for value in raw_limits])[::-1]:
            limits[int(index)] += 1
            if sum(limits) == max_samples:
                break
    while sum(limits) > max_samples:
        for index in np.argsort([value - int(value) for value in raw_limits]):
            if limits[int(index)] > 1:
                limits[int(index)] -= 1
            if sum(limits) == max_samples:
                break
    return limits


def _transforms_without_normalize(pipeline: Any) -> tuple:
    return tuple(pipeline.transforms or ())


def _component_summary(component: Any) -> dict[str, Any]:
    return {
        "type": type(component).__name__,
        "key": getattr(component, "key", None),
        "pipeline": [_transform_summary_name(transform) for transform in getattr(component, "transforms", ())],
    }


def _transform_summary_name(transform: Any) -> str:
    if hasattr(transform, "__name__"):
        return str(transform.__name__)
    name = type(transform).__name__
    if hasattr(transform, "quat_order"):
        order = getattr(transform, "resolved_quat_order", transform.quat_order)
        name = f"{name}(quat_order={order!r})"
    return name


def _build_pipeline_for_stats(config: Any, *, include_video: bool) -> Any:
    # fps / delta_timestamps probing needs a filesystem dataset root, not the
    # manifest-string or the list form. Use the primary repo; all repos in a
    # multi-repo config share schema (see ``_primary_repo_id`` docstring).
    primary_repo = config._primary_repo_id() if hasattr(config, "_primary_repo_id") else config.repo_id
    source_kwargs = _build_stats_source_kwargs(
        config.source_kwargs,
        repo_id=primary_repo,
        action_horizon=getattr(config, "action_horizon", None),
        include_video=include_video,
    )
    stat_config = dataclasses.replace(
        config,
        source_kwargs=source_kwargs,
        state_padding_dim=None,
        action_padding_dim=None,
    )
    return stat_config.build_pipeline(disable_component_normalization=True)


def _build_stats_source_kwargs(
    source_kwargs: Mapping[str, Any] | None,
    *,
    repo_id: str,
    action_horizon: int | None,
    include_video: bool,
) -> dict[str, Any]:
    merged = dict(source_kwargs or {})
    if action_horizon is not None and action_horizon > 1 and "delta_timestamps" not in merged:
        merged["delta_timestamps"] = {"action": [t / _infer_dataset_fps(repo_id) for t in range(action_horizon)]}
    if include_video:
        return merged

    # Norm stats only consume state/action, so avoid MP4/image access on the raw dataset.
    merged["download_videos"] = False
    merged["video_backend"] = None
    return merged


def _infer_dataset_fps(repo_id: str) -> float:
    dataset_fps = _load_dataset_fps(repo_id)
    if dataset_fps is not None:
        return dataset_fps

    _require_backend()

    local_path = Path(repo_id)
    if local_path.exists():
        dataset_meta = LeRobotDatasetMetadata(local_path.name, root=local_path)
    else:
        dataset_meta = LeRobotDatasetMetadata(repo_id)
    return float(dataset_meta.fps)


def _resolve_norm_stats_output_dir(config: Any, output_dir: str | Path | None) -> str | Path:
    if output_dir is not None:
        return output_dir

    config_output_dir = getattr(config, "norm_stats_dir", None)
    if config_output_dir:
        return config_output_dir
    repo_id = getattr(config, "repo_id", None)
    if repo_id:
        return Path(repo_id) / "norm_stats"
    raise ValueError(
        "Norm stats output dir is not set. Pass --output-dir or define repo_id/norm_stats_dir in the config."
    )


def _stable_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _stable_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stable_jsonable(item) for item in value]
    return value


def _regenerate_hint(*, reason: str, path: pathlib.Path, config_summary: dict[str, Any]) -> str:
    return (
        f"{reason}: {path}\n"
        f"Config summary: {_compact_summary(config_summary)}\n"
        f"Regenerate with: "
        f"python -m tau0_vla.data compute_norm <config-name> --output-dir {path.parent}"
    )


def _compact_summary(config_summary: dict[str, Any]) -> str:
    return json.dumps(config_summary, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
