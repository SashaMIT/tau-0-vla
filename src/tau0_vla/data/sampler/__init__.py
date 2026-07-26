from __future__ import annotations

import itertools
import os
import random
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from tau0_vla.data.source import get_episode_ids, get_episode_index_bounds

_ANNOTATION_FILTER_ALIASES = {
    "l3": "l3",
    "instruction_segments": "l3",
    "instruction_segment": "l3",
    "segment": "l3",
    "segments": "l3",
    "l2": "l2",
    "subtask_frame": "l2",
    "subtask_frames": "l2",
    "subtask": "l2",
    "key_frame": "l2",
    "key_frames": "l2",
    "error_frame": "error_frame",
    "error_frames": "error_frame",
}


def select_physical_shard(
    weights: Sequence[int],
    world_size: int,
    local_index: int,
    *aligned_lists: Sequence,
) -> dict[str, Any]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if not (0 <= local_index < world_size):
        raise ValueError(f"local_index must be in [0, {world_size}), got {local_index}")

    n = len(weights)
    for i, lst in enumerate(aligned_lists):
        if len(lst) != n:
            raise ValueError(f"aligned_lists[{i}] has length {len(lst)}, expected {n}")

    items = list(enumerate(weights))
    items.sort(key=lambda x: (-x[1], x[0]))
    bucket_sums: list[int] = [0] * world_size
    bucket_indices: list[list[int]] = [[] for _ in range(world_size)]

    for idx, weight in items:
        best = min(range(world_size), key=lambda bucket: (bucket_sums[bucket], bucket))
        bucket_indices[best].append(idx)
        bucket_sums[best] += weight

    selected = sorted(bucket_indices[local_index])
    return {
        "indices": selected,
        "weights": [weights[i] for i in selected],
        "sum": bucket_sums[local_index],
        "aligned": [[lst[i] for i in selected] for lst in aligned_lists],
    }


def select_episode_ids(
    meta: Any,
    *,
    debug_one_episode: bool,
    filter_by_stats: bool,
    random_ratio: float,
    valid_range: tuple[float, float] = (-3.4, 3.6),
) -> list[int]:
    if debug_one_episode:
        return [0]

    episodes = list(range(meta.total_episodes))
    if filter_by_stats:
        bad_eps = find_bad_episode_indices(meta, valid_range)
        if bad_eps:
            episodes = [ep for ep in episodes if ep not in bad_eps]

    if random_ratio >= 1.0:
        return episodes

    selected_episode_num = int(len(episodes) * random_ratio)
    if selected_episode_num == 0:
        return []
    return random.sample(episodes, k=selected_episode_num)


def find_bad_episode_indices(meta: Any, valid_range: tuple[float, float]) -> set[int]:
    raw_stats = getattr(meta, "episodes_stats", None)
    if raw_stats is None:
        return set()

    if isinstance(raw_stats, Mapping):
        items = sorted(raw_stats.items())
    else:
        items = []
        for row in raw_stats:
            if isinstance(row, Mapping):
                episode_index = int(row.get("episode_index", len(items)))
                items.append((episode_index, row))

    lower, upper = valid_range
    bad_eps: set[int] = set()
    for episode_index, row in items:
        state_min = row.get("stats/observation.state/min")
        state_max = row.get("stats/observation.state/max")
        action_min = row.get("stats/action/min")
        action_max = row.get("stats/action/max")
        bounds = [arr for arr in (state_min, state_max, action_min, action_max) if arr is not None]
        if not bounds:
            continue
        min_value = min(float(np.min(np.asarray(arr, dtype=np.float32))) for arr in bounds[::2] or bounds)
        max_value = max(float(np.max(np.asarray(arr, dtype=np.float32))) for arr in bounds[1::2] or bounds)
        if min_value < lower or max_value > upper:
            bad_eps.add(int(episode_index))
    return bad_eps


def build_segment_index_ranges(
    *,
    dataset_metas: Sequence[Any],
    datasets: Sequence[Any],
    tail_buffer: int,
    positive_labels: Sequence[str] = ("l3", "l2"),
    negative_labels: Sequence[str] = ("error_frame",),
) -> tuple[np.ndarray, np.ndarray]:
    valid_ranges: list[list[int]] = []
    base_offsets = list(itertools.accumulate((0, *(len(ds) for ds in datasets[:-1]))))
    report_rows: list[tuple[str, int, int, int]] = []

    total_raw = 0
    total_kept = 0
    for meta, ds, base_offset in zip(dataset_metas, datasets, base_offsets, strict=True):
        ep_from, ep_to = get_episode_index_bounds(ds)
        ds_kept = 0
        ds_segments = 0
        positive = {_normalize_annotation_filter_label(v) for v in positive_labels}
        negative = {_normalize_annotation_filter_label(v) for v in negative_labels}
        # filter_by_segments=True promises "only anchors inside declared
        # instruction_segments intervals survive". If the annotation policy
        # requests ``l3`` but a repo doesn't declare any segments at all, it
        # can't honor that promise — silently keeping its frames would sneak
        # past prompt_source=instruction_segments and fail later, or worse,
        # contaminate training with unannotated data. Refuse loudly.
        if "l3" in positive and not _repo_declares_segments(meta):
            repo_label = getattr(meta, "root", None) or getattr(meta, "repo_id", None) or "<unknown>"
            raise ValueError(
                f"filter_by_segments=True but repo {repo_label!r} has no non-empty "
                f"'instruction_segments' in meta/info.json. Either annotate the repo, "
                f"exclude it from this config, or drop filter_by_segments."
            )
        # `ep_from` / `ep_to` are indexed by absolute episode_index (full list from
        # meta.episodes); `get_episode_ids` returns only the episodes actually loaded.
        for episode_index in get_episode_ids(ds):
            episode_index = int(episode_index)
            ep_start = int(ep_from[episode_index])
            ep_end = int(ep_to[episode_index])
            if ep_end <= ep_start:
                continue
            ep_len = ep_end - ep_start
            pieces = [(0, ep_len)]
            for label in sorted(positive):
                intervals = _annotation_intervals(meta, episode_index, ep_len=ep_len, label=label)
                if not intervals:
                    pieces = []
                    break
                pieces = _intersect_many_with_intervals(pieces, intervals)
                if not pieces:
                    break
            if not pieces:
                continue
            for label in sorted(negative):
                intervals = _annotation_intervals(meta, episode_index, ep_len=ep_len, label=label)
                if not intervals:
                    continue
                refined: list[tuple[int, int]] = []
                for piece in pieces:
                    refined.extend(_subtract_intervals(piece, intervals))
                pieces = refined
                if not pieces:
                    break
            for s_local, e_local in pieces:
                start = ep_start + s_local
                end = ep_start + e_local - tail_buffer
                if start < end:
                    valid_ranges.append([base_offset + start, base_offset + end])
                    ds_kept += end - start
                    ds_segments += 1

        ds_total = len(ds)
        total_raw += ds_total
        total_kept += ds_kept
        label = getattr(meta, "root", None) or getattr(meta, "repo_id", None) or "<unknown>"
        report_rows.append((str(label), ds_kept, ds_total, ds_segments))

    _print_filter_report(
        title="segment filter",
        rows=report_rows,
        total_kept=total_kept,
        total_raw=total_raw,
        total_segments=len(valid_ranges),
    )
    return _ranges_with_cumsum(valid_ranges)


def _print_filter_report(
    *,
    title: str,
    rows: Sequence[tuple[str, int, int, int]],
    total_kept: int,
    total_raw: int,
    total_segments: int,
) -> None:
    if not rows:
        _print_filter_block(f"[sampler] {title}: kept 0/0 frames (0.00%), filtered 0.00%, segments=0")
        return
    lines = [f"[sampler] {title}"]
    for label, kept, raw, segments in rows:
        lines.append(label)
        lines.append("  " + _format_filter_stats(kept, raw, segments))
    if len(rows) > 1:
        lines.append("-" * 80)
        lines.append("TOTAL")
        lines.append("  " + _format_filter_stats(total_kept, total_raw, total_segments))
    _print_filter_block("\n".join(lines))


def _format_filter_stats(kept: int, raw: int, segments: int) -> str:
    keep_pct, filtered_pct = _filter_percentages(kept, raw)
    return (
        f"kept {kept:,}/{raw:,} frames ({keep_pct:.2f}%), "
        f"filtered {filtered_pct:.2f}%, segments={segments:,}"
    )


def _filter_percentages(kept: int, raw: int) -> tuple[float, float]:
    filtered_pct = 0.0 if raw == 0 else 100.0 * (raw - kept) / raw
    return 100.0 - filtered_pct, filtered_pct


def build_constant_action_index_ranges(
    *,
    datasets: Sequence[Any],
    raw_action_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    valid_ranges: list[list[int]] = []
    base_offsets = list(itertools.accumulate((0, *(len(ds) for ds in datasets[:-1]))))

    for ds_idx, ds in enumerate(datasets):
        base = base_offsets[ds_idx]
        ep_from, ep_to = get_episode_index_bounds(ds)

        for ep_idx in range(ep_from.shape[0]):
            ep_start = int(ep_from[ep_idx])
            ep_end = int(ep_to[ep_idx])
            n_frames = ep_end - ep_start
            if n_frames <= 0:
                continue
            actions = ds.hf_dataset[ep_start:ep_end][raw_action_key]
            if isinstance(actions, list):
                actions = torch.stack(actions)
            elif not isinstance(actions, torch.Tensor):
                actions = torch.tensor(actions)

            if actions.ndim == 3:
                first = actions[:, :1, :]
                is_const = torch.all(actions == first, dim=1).all(dim=1)
            elif actions.ndim == 2:
                first = actions[:, :1]
                is_const = torch.all(actions == first, dim=1)
            else:
                valid_ranges.append([base + ep_start, base + ep_end])
                continue

            ep_filtered = int(is_const.sum().item())
            if ep_filtered == 0:
                valid_ranges.append([base + ep_start, base + ep_end])
            elif ep_filtered != n_frames:
                is_valid = ~is_const
                diff = torch.diff(is_valid.to(torch.int8), prepend=torch.tensor([0], dtype=torch.int8))
                starts = torch.where(diff == 1)[0]
                ends_diff = torch.diff(is_valid.to(torch.int8), append=torch.tensor([0], dtype=torch.int8))
                ends = torch.where(ends_diff == -1)[0] + 1
                for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
                    valid_ranges.append([base + ep_start + start, base + ep_start + end])

    return _ranges_with_cumsum(valid_ranges)


def _print_filter_block(message: str) -> None:
    width = max(len(line) for line in message.splitlines())
    line = "=" * min(120, max(80, width))
    print(line)
    print(message)
    print(line)


def intersect_index_ranges(ranges_a: np.ndarray, ranges_b: np.ndarray) -> np.ndarray:
    if ranges_a.size == 0 or ranges_b.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    result = []
    i, j = 0, 0
    while i < len(ranges_a) and j < len(ranges_b):
        start = max(int(ranges_a[i][0]), int(ranges_b[j][0]))
        end = min(int(ranges_a[i][1]), int(ranges_b[j][1]))
        if start < end:
            result.append([start, end])
        if ranges_a[i][1] < ranges_b[j][1]:
            i += 1
        else:
            j += 1
    return np.asarray(result, dtype=np.int64) if result else np.empty((0, 2), dtype=np.int64)


def map_selected_index(filtered_ranges: np.ndarray, filtered_cumsum: np.ndarray, filtered_idx: int) -> int:
    j = int(np.searchsorted(filtered_cumsum, filtered_idx, side="right"))
    prev = int(filtered_cumsum[j - 1]) if j > 0 else 0
    offset = filtered_idx - prev
    start = int(filtered_ranges[j, 0])
    return start + offset


# Drop pathologically long instruction_segments: a handful of arx/g2a/g1
# auto-annotations are corrupted (repetition-collapse walls of '#'/'a'/garbage
# Chinese, 5k-40k chars vs a normal ~60-200 char task string). If anchored they
# blow the VLA sequence length -> rank OOM or, after the vla_max_token_len
# re-roll, a dataloader straggler that times out the gradient allreduce (the
# step-227 crash). Excluding the segment here means its frames are never
# selected as l3 anchors, so the corruption never reaches the model. Threshold
# is generous (normal max ~200, corrupt min ~4900); override via env, 0 = off.
_MAX_INSTRUCTION_CHARS = int(os.environ.get("TAU0_MAX_INSTRUCTION_CHARS", "1024"))
_warned_long_instr: set[str] = set()


def _instruction_too_long(segment: Mapping) -> bool:
    if _MAX_INSTRUCTION_CHARS <= 0:
        return False
    instr = segment.get("instruction")
    if instr is None or len(str(instr)) <= _MAX_INSTRUCTION_CHARS:
        return False
    key = str(instr)[:48]
    if key not in _warned_long_instr:
        _warned_long_instr.add(key)
        print(
            f"[finch] dropping corrupted instruction_segment "
            f"({len(str(instr))} chars > {_MAX_INSTRUCTION_CHARS}); instr[:48]={key!r}",
            flush=True,
        )
    return True


def _episode_segments(meta: Any, episode_index: int) -> list[dict[str, Any]]:
    info = getattr(meta, "info", None) or {}
    segments_map = info.get("instruction_segments") or {}
    if isinstance(segments_map, Mapping):
        segments = segments_map.get(str(episode_index), segments_map.get(episode_index, ()))
    else:
        segments = ()
    return [
        segment
        for segment in segments
        if isinstance(segment, Mapping) and not _instruction_too_long(segment)
    ]


def _annotation_intervals(
    meta: Any,
    episode_index: int,
    *,
    ep_len: int,
    label: str,
) -> list[tuple[int, int]]:
    normalized = _normalize_annotation_filter_label(label)
    if normalized == "l3":
        intervals = []
        for segment in _episode_segments(meta, episode_index):
            seg_s = max(0, int(segment.get("start_frame_index", 0)))
            seg_e = min(max(0, int(segment.get("end_frame_index", ep_len))), ep_len)
            if seg_e > seg_s:
                intervals.append((seg_s, seg_e))
        return intervals
    if normalized == "l2":
        intervals = _episode_subtask_key_frames(meta, episode_index)
        if intervals:
            return intervals
        # Episode-level fallback: if no valid subtask key-frame ranges are
        # present at all, treat l2 as l3 rather than collapsing the whole
        # episode to empty. This preserves older repos whose instruction
        # segments exist but key_frame annotations were never filled in.
        return _annotation_intervals(meta, episode_index, ep_len=ep_len, label="l3")
    if normalized == "error_frame":
        return _episode_error_key_frames(meta, episode_index)
    raise ValueError(f"Unsupported annotation filter label: {label!r}")


def _normalize_annotation_filter_label(label: Any) -> str:
    key = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _ANNOTATION_FILTER_ALIASES.get(key)
    if normalized is None:
        raise ValueError(f"Unknown annotation filter label: {label!r}")
    return normalized


_SUBTASK_FRAME_LABEL = "subtaskframe"
_ERROR_FRAME_LABEL = "errorframe"
_SUBTASK_FRAME_TYPE_ALIASES = frozenset(
    {
        "subtaskframe",
        "subtaskframes",
        "subtaskframe.",
        "subtaskframe1",
        "subtaskframe2",
        "subtaskframe3",
        "subtaskframe4",
        "taskframe",
        "taskframes",
        "taskframe.",
        "taskframe1",
        "taskframe2",
        "taskframe3",
        "taskframe4",
    }
)
_ERROR_FRAME_TYPE_ALIASES = frozenset(
    {
        "error",
        "errorframe",
        "errorframes",
        "errorframe.",
        "errorframe2",
    }
)


def _normalize_frame_type_name(name: Any) -> str:
    """Normalize ``frame_type_name`` for matching.

    Keeps only ASCII letters / digits and lowercases the result, so variants
    with spaces / hyphens / periods collapse to the same token.
    """
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _canonical_key_frame_type(name: Any) -> str | None:
    normalized = _normalize_frame_type_name(name)
    if normalized in _SUBTASK_FRAME_TYPE_ALIASES:
        return _SUBTASK_FRAME_LABEL
    if normalized in _ERROR_FRAME_TYPE_ALIASES:
        return _ERROR_FRAME_LABEL
    return None


def _episode_subtask_key_frames(meta: Any, episode_index: int) -> list[tuple[int, int]]:
    """Return merged sub-task ``key_frame`` ranges ``(start, end)`` (end
    exclusive) for one episode, in **episode-local** frame coordinates.

    Handles both info.json shapes seen in the wild:
      * ``kf[ep]`` is a ``list`` of items.
      * ``kf[ep]`` is a ``dict`` keyed by track (e.g.
        ``{"single": [...], "dual": [...]}``) — all tracks flattened.

    Only items whose normalized ``frame_type_name`` resolves to the sub-task
    alias family are kept; error/unrecognized labels are dropped. Overlapping
    ranges are merged so downstream intersection math is well-defined.
    """
    items = _episode_key_frame_items(meta, episode_index)
    ranges: list[tuple[int, int]] = []
    for it in items:
        if not isinstance(it, Mapping):
            continue
        if _canonical_key_frame_type(it.get("frame_type_name", "")) != _SUBTASK_FRAME_LABEL:
            continue
        try:
            start = int(it["start"])
            end = int(it["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            ranges.append((start, end))
    return _merge_intervals(ranges)


def _episode_error_key_frames(meta: Any, episode_index: int) -> list[tuple[int, int]]:
    """Return merged error ``key_frame`` intervals.

    Accepts either an explicit truthy ``error_frame`` flag or an error-like
    ``frame_type_name`` alias such as ``"Error Frame"`` / ``"ErrorFrame"``.
    """
    items = _episode_key_frame_items(meta, episode_index)
    ranges: list[tuple[int, int]] = []
    for it in items:
        if not isinstance(it, Mapping):
            continue
        is_error_flag = _is_truthy_flag(it.get("error_frame"))
        is_error_type = _canonical_key_frame_type(it.get("frame_type_name", "")) == _ERROR_FRAME_LABEL
        if not (is_error_flag or is_error_type):
            continue
        try:
            start = int(it["start"])
            end = int(it["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end > start:
            ranges.append((start, end))
    return _merge_intervals(ranges)


def _episode_key_frame_items(meta: Any, episode_index: int) -> list[Any]:
    info = getattr(meta, "info", None) or {}
    kf_map = info.get("key_frame")
    if not isinstance(kf_map, Mapping):
        return []
    ep_val = kf_map.get(str(episode_index))
    if ep_val is None:
        ep_val = kf_map.get(episode_index)
    if ep_val is None:
        return []

    if isinstance(ep_val, list):
        return list(ep_val)
    if isinstance(ep_val, Mapping):
        items: list[Any] = []
        for sub_list in ep_val.values():
            if isinstance(sub_list, list):
                items.extend(sub_list)
        return items
    return []


def _is_truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _subtract_intervals(
    interval: tuple[int, int],
    blocked: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    start, end = interval
    pieces = [(start, end)]
    for block_start, block_end in blocked:
        next_pieces: list[tuple[int, int]] = []
        for piece_start, piece_end in pieces:
            overlap_start = max(piece_start, block_start)
            overlap_end = min(piece_end, block_end)
            if overlap_end <= overlap_start:
                next_pieces.append((piece_start, piece_end))
                continue
            if piece_start < overlap_start:
                next_pieces.append((piece_start, overlap_start))
            if overlap_end < piece_end:
                next_pieces.append((overlap_end, piece_end))
        pieces = next_pieces
        if not pieces:
            break
    return pieces


def _intersect_many_with_intervals(
    base: list[tuple[int, int]],
    other: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for base_start, base_end in base:
        for other_start, other_end in other:
            start = max(base_start, other_start)
            end = min(base_end, other_end)
            if end > start:
                out.append((start, end))
    return _merge_intervals(out)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged: list[tuple[int, int]] = [ordered[0]]
    for s, e in ordered[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def _repo_declares_segments(meta: Any) -> bool:
    """True iff ``info.json['instruction_segments']`` is a non-empty mapping.

    Used by ``build_segment_index_ranges`` to tell "repo has no segments
    declared" (filter is a no-op) from "repo has segments declared but THIS
    episode's entry is empty" (unannotated episode — must be filtered out).
    """
    info = getattr(meta, "info", None) or {}
    segments_map = info.get("instruction_segments")
    return isinstance(segments_map, Mapping) and len(segments_map) > 0


def _ranges_with_cumsum(ranges: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    if not ranges:
        return np.empty((0, 2), dtype=np.int64), np.array([], dtype=np.int64)
    ranges_arr = np.asarray(ranges, dtype=np.int64)
    lengths = ranges_arr[:, 1] - ranges_arr[:, 0]
    cumsum = np.cumsum(lengths, dtype=np.int64)
    return ranges_arr, cumsum


__all__ = [
    "build_constant_action_index_ranges",
    "build_segment_index_ranges",
    "find_bad_episode_indices",
    "intersect_index_ranges",
    "map_selected_index",
    "select_episode_ids",
    "select_physical_shard",
]
