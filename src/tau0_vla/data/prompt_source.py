from __future__ import annotations

import dataclasses
import functools
import random
from collections.abc import Mapping, Sequence
from typing import Any

from tau0_vla.data.modalities.language import Prompt
from tau0_vla.data.source import load_dataset_info

_LABEL_SOURCE_ALIASES = {
    "parquet": "parquet",
    "l1": "high_level_instruction",
    "high_level_instruction": "high_level_instruction",
    "high_level_instructions": "high_level_instruction",
    "l2": "subtask_frame",
    "subtask_frame": "subtask_frame",
    "subtask_frames": "subtask_frame",
    "subtask": "subtask_frame",
    "key_frame": "subtask_frame",
    "key_frames": "subtask_frame",
    "l3": "instruction_segments",
    "instruction_segments": "instruction_segments",
    "instruction_segment": "instruction_segments",
    "segment": "instruction_segments",
    "segments": "instruction_segments",
    "data_studio": "data_studio",
}
_SUPPORTED_LABEL_SOURCES = tuple(_LABEL_SOURCE_ALIASES)


@dataclasses.dataclass(frozen=True)
class PromptSource:
    """Declarative prompt source for a RobotConfig / data_config.

    Use the ``fix`` / ``from_label`` / ``random`` factories rather than the raw
    constructor.

    ``fix``        — every sample gets the same ``fix_prompt``.
    ``from_label`` — per-sample prompt drawn from one of:
        * ``"parquet"``              — column ``parquet_key`` (defaults to the
          robot's ``repack['prompt']`` column, e.g. ``"task"``).
        * ``"high_level_instruction"`` / ``"l1"`` — episode-level instruction
          from ``meta/info.json['high_level_instruction'][episode]``.
        * ``"subtask_frame"`` / ``"l2"`` — sub-task string from
          ``meta/info.json['subtask_frame']`` or compatible ``key_frame``
          subtask intervals.
        * ``"instruction_segments"`` / ``"l3"`` — sub-task string from
          ``meta/info.json['instruction_segments'][episode][frame]``. Missing
          or empty entries raise loudly; use ``filter_by_segments=True`` on
          the ``RobotConfig`` to exclude unannotated frames at anchor time.
        * ``"data_studio"``          — placeholder, raises ``NotImplementedError``.
      Optional ``prompt_template`` wraps only this source's resolved
      instruction before it reaches the global ``RobotConfig.prompt`` layer.
    ``random``     — per-sample pick one of ``sources`` with the given weights,
      then delegate ``.resolve(...)`` to it. Enables mixing (e.g. 50% a fixed
      task-level prompt, 50% segment-level). Nestable.
    """

    kind: str
    fix_prompt: str | None = None
    source: str | None = None
    parquet_key: str | None = None
    prompt_template: str | None = None
    sources: tuple["PromptSource", ...] | None = None
    probabilities: tuple[float, ...] | None = None

    @classmethod
    def fix(cls, fix_prompt: str, *, prompt_template: str | None = None) -> "PromptSource":
        return cls(kind="fix", fix_prompt=fix_prompt, prompt_template=prompt_template)

    @classmethod
    def from_label(
        cls,
        source: str,
        *,
        parquet_key: str | None = None,
        prompt_template: str | None = None,
    ) -> "PromptSource":
        return cls(kind="from_label", source=source, parquet_key=parquet_key, prompt_template=prompt_template)

    @classmethod
    def random(
        cls,
        sources: Sequence["PromptSource"],
        *,
        probabilities: Sequence[float] | None = None,
    ) -> "PromptSource":
        """Stochastic mix of ``PromptSource``s.

        At each sample, one of ``sources`` is picked with probability
        proportional to the matching entry in ``probabilities`` (weights are
        auto-normalized — ``[1, 1]`` is equivalent to ``[0.5, 0.5]``).
        ``probabilities=None`` → uniform.

        Example::

            PromptSource.random(
                [
                    PromptSource.fix("put the red block into the cup"),
                    PromptSource.from_label(source="instruction_segments"),
                ],
                probabilities=[0.5, 0.5],
            )
        """
        sources_tup = tuple(sources)
        if probabilities is None:
            probs_tup: tuple[float, ...] = tuple(1.0 for _ in sources_tup)
        else:
            probs_tup = tuple(float(p) for p in probabilities)
        return cls(kind="random", sources=sources_tup, probabilities=probs_tup)

    def __post_init__(self) -> None:
        if self.kind == "fix":
            if not self.fix_prompt:
                raise ValueError("PromptSource.fix requires a non-empty fix_prompt")
            if self.source is not None or self.parquet_key is not None:
                raise ValueError("PromptSource(kind='fix') must not set source / parquet_key")
            if self.sources is not None or self.probabilities is not None:
                raise ValueError("PromptSource(kind='fix') must not set sources / probabilities")
            return
        if self.kind == "from_label":
            source = _canonical_label_source(self.source)
            if self.fix_prompt is not None:
                raise ValueError("PromptSource(kind='from_label') must not set fix_prompt")
            if self.parquet_key is not None and source != "parquet":
                raise ValueError("parquet_key is only valid with source='parquet'")
            if self.sources is not None or self.probabilities is not None:
                raise ValueError("PromptSource(kind='from_label') must not set sources / probabilities")
            object.__setattr__(self, "source", source)
            return
        if self.kind == "random":
            if (
                self.fix_prompt is not None
                or self.source is not None
                or self.parquet_key is not None
                or self.prompt_template is not None
            ):
                raise ValueError(
                    "PromptSource(kind='random') must not set fix_prompt / source / parquet_key / prompt_template"
                )
            if not self.sources:
                raise ValueError("PromptSource.random requires a non-empty sources list")
            for i, s in enumerate(self.sources):
                if not isinstance(s, PromptSource):
                    raise TypeError(f"PromptSource.random: sources[{i}] must be a PromptSource, got {type(s).__name__}")
            if self.probabilities is None or len(self.probabilities) != len(self.sources):
                raise ValueError(
                    f"PromptSource.random: probabilities (got {self.probabilities}) must have "
                    f"the same length as sources ({len(self.sources)})"
                )
            if any(p < 0 for p in self.probabilities):
                raise ValueError("PromptSource.random: all probabilities must be non-negative")
            if sum(self.probabilities) <= 0:
                raise ValueError("PromptSource.random: probabilities must sum to a positive number")
            return
        raise ValueError(f"Invalid PromptSource kind: {self.kind!r}")

    def uses_instruction_segments(self) -> bool:
        """True iff this source (or any nested source under ``random``) reads
        from ``meta/info.json['instruction_segments']`` — used by callers to
        emit a config-time warning when ``filter_by_segments`` isn't enabled
        alongside."""
        if self.kind == "from_label" and self.source == "instruction_segments":
            return True
        if self.kind == "random" and self.sources:
            return any(s.uses_instruction_segments() for s in self.sources)
        return False

    def resolve(
        self,
        raw: Mapping[str, Any],
        *,
        repo_id: str,
        default_parquet_key: str,
    ) -> str:
        """Return the prompt string for one raw sample."""
        if self.kind == "fix":
            return self._apply_template(str(self.fix_prompt))
        if self.kind == "random":
            chosen = random.choices(self.sources, weights=self.probabilities, k=1)[0]
            return chosen.resolve(raw, repo_id=repo_id, default_parquet_key=default_parquet_key)
        if self.source == "parquet":
            key = self.parquet_key or default_parquet_key
            if key not in raw:
                raise KeyError(f"prompt_source=parquet: column '{key}' not in raw sample")
            return self._apply_template(_stringify(raw[key]))
        if self.source == "instruction_segments":
            if "episode_index" not in raw or "frame_index" not in raw:
                raise KeyError("prompt_source=instruction_segments requires episode_index / frame_index")
            return self._apply_template(
                _lookup_segment_instruction(
                    _load_instruction_segments(repo_id),
                    _as_scalar(raw["episode_index"]),
                    _as_scalar(raw["frame_index"]),
                    repo_id=repo_id,
                )
            )
        if self.source == "high_level_instruction":
            if "episode_index" not in raw:
                raise KeyError("prompt_source=high_level_instruction requires episode_index")
            return self._apply_template(
                _lookup_high_level_instruction(
                    _load_prompt_info(repo_id),
                    _as_scalar(raw["episode_index"]),
                    repo_id=repo_id,
                )
            )
        if self.source == "subtask_frame":
            if "episode_index" not in raw or "frame_index" not in raw:
                raise KeyError("prompt_source=subtask_frame requires episode_index / frame_index")
            return self._apply_template(
                _lookup_subtask_frame_instruction(
                    _load_prompt_info(repo_id),
                    _as_scalar(raw["episode_index"]),
                    _as_scalar(raw["frame_index"]),
                    repo_id=repo_id,
                )
            )
        if self.source == "data_studio":
            raise NotImplementedError("PromptSource.from_label(source='data_studio') is not implemented yet")
        raise AssertionError(f"unreachable PromptSource: {self!r}")

    def _apply_template(self, instruction: str) -> str:
        if self.prompt_template is None:
            return instruction
        return Prompt(template=self.prompt_template).format(instruction)


def _canonical_label_source(source: Any) -> str:
    key = str(source).strip().lower().replace("-", "_").replace(" ", "_")
    canonical = _LABEL_SOURCE_ALIASES.get(key)
    if canonical is None:
        raise ValueError(f"from_label source must be one of {_SUPPORTED_LABEL_SOURCES}, got {source!r}")
    return canonical


@functools.lru_cache(maxsize=128)
def _load_prompt_info(repo_id: str) -> Mapping[str, Any]:
    info = load_dataset_info(repo_id) or {}
    if not isinstance(info, Mapping):
        raise TypeError(f"meta/info.json must be a mapping, got {type(info).__name__}")
    return info


@functools.lru_cache(maxsize=128)
def _load_instruction_segments(repo_id: str) -> Mapping[str, Any]:
    segments = _load_prompt_info(repo_id).get("instruction_segments") or {}
    if not isinstance(segments, Mapping):
        raise TypeError(f"meta/info.json['instruction_segments'] must be a mapping, got {type(segments).__name__}")
    return segments


def _lookup_segment_instruction(
    segments_map: Mapping[str, Any],
    episode_index: Any,
    frame_index: Any,
    *,
    repo_id: str,
) -> str:
    raw_segments = segments_map.get(str(episode_index))
    if raw_segments is None:
        raw_segments = segments_map.get(episode_index)
    segments = [seg for seg in (raw_segments or ()) if isinstance(seg, Mapping)]
    if not segments:
        raise KeyError(
            f"prompt_source=instruction_segments: episode_index={episode_index!r} has no "
            f"usable entry in meta/info.json['instruction_segments'] of repo_id={repo_id!r}. "
            f"Set filter_by_segments=True on the RobotConfig to exclude unannotated episodes."
        )
    frame = int(frame_index)
    for i, seg in enumerate(segments):
        # Force the first segment to start at 0 so early frames always hit a segment (matches openpi).
        start = 0 if i == 0 else int(seg["start_frame_index"])
        if start <= frame < int(seg["end_frame_index"]):
            return str(seg["instruction"])
    return str(segments[-1]["instruction"])


def _lookup_high_level_instruction(
    info: Mapping[str, Any],
    episode_index: Any,
    *,
    repo_id: str,
) -> str:
    raw = info.get("high_level_instruction")
    if raw is None:
        raw = info.get("high_level_instructions")
    value = _lookup_episode_value(raw, episode_index)
    text = _maybe_instruction_text(value)
    if text is not None:
        return text

    fallback_segments = _lookup_episode_value(info.get("instruction_segments"), episode_index)
    for segment in _flatten_frame_entries(fallback_segments):
        text = _maybe_instruction_text(segment)
        if text is not None:
            return text
    if value is None:
        raise KeyError(
            f"prompt_source=high_level_instruction: episode_index={episode_index!r} has no usable "
            f"entry in meta/info.json['high_level_instruction'] of repo_id={repo_id!r}."
        )
    raise KeyError(
        f"prompt_source=high_level_instruction: episode_index={episode_index!r} has only blank "
        f"high-level text in meta/info.json of repo_id={repo_id!r}."
    )


def _lookup_subtask_frame_instruction(
    info: Mapping[str, Any],
    episode_index: Any,
    frame_index: Any,
    *,
    repo_id: str,
) -> str:
    raw_entries = _lookup_episode_value(info.get("subtask_frame") or info.get("subtask_frames"), episode_index)
    entries = _flatten_frame_entries(raw_entries)
    if not entries:
        entries = [
            entry
            for entry in _flatten_frame_entries(_lookup_episode_value(info.get("key_frame"), episode_index))
            if _is_subtask_frame(entry)
        ]
    if not entries:
        raise KeyError(
            f"prompt_source=subtask_frame: episode_index={episode_index!r} has no usable subtask-frame "
            f"entry in meta/info.json of repo_id={repo_id!r}."
        )

    frame = int(frame_index)
    last_before: Mapping[str, Any] | None = None
    for entry in sorted(entries, key=lambda item: int(item.get("start", item.get("start_frame_index", 0)))):
        start = int(entry.get("start", entry.get("start_frame_index", 0)))
        end = int(entry.get("end", entry.get("end_frame_index", start + 1)))
        if start <= frame < end:
            text = _maybe_instruction_text(entry)
            if text is not None:
                return text
            return _lookup_segment_instruction(
                _load_instruction_segments(repo_id),
                episode_index,
                frame_index,
                repo_id=repo_id,
            )
        if start <= frame:
            last_before = entry
    if last_before is not None:
        text = _maybe_instruction_text(last_before)
        if text is not None:
            return text
        return _lookup_segment_instruction(
            _load_instruction_segments(repo_id),
            episode_index,
            frame_index,
            repo_id=repo_id,
        )
    text = _maybe_instruction_text(entries[0])
    if text is not None:
        return text
    return _lookup_segment_instruction(
        _load_instruction_segments(repo_id),
        episode_index,
        frame_index,
        repo_id=repo_id,
    )


def _lookup_episode_value(raw: Any, episode_index: Any) -> Any:
    if isinstance(raw, Mapping):
        value = raw.get(str(episode_index))
        if value is None:
            value = raw.get(episode_index)
        return value
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        try:
            return raw[int(episode_index)]
        except (IndexError, TypeError, ValueError):
            return None
    return raw


def _flatten_frame_entries(raw: Any) -> list[Mapping[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, Mapping):
        if any(key in raw for key in ("start", "end", "start_frame_index", "end_frame_index")):
            return [raw]
        out: list[Mapping[str, Any]] = []
        for value in raw.values():
            out.extend(_flatten_frame_entries(value))
        return out
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        return [item for item in raw if isinstance(item, Mapping)]
    return []


def _is_subtask_frame(entry: Mapping[str, Any]) -> bool:
    frame_type = entry.get("frame_type_name")
    if frame_type is None:
        return True
    normalized = "".join(ch for ch in str(frame_type).lower() if ch.isalnum())
    return normalized in {
        "subtaskframe",
        "subtaskframes",
        "subtaskframe1",
        "subtaskframe2",
        "subtaskframe3",
        "subtaskframe4",
        "taskframe",
        "taskframes",
        "taskframe1",
        "taskframe2",
        "taskframe3",
        "taskframe4",
    }


def _maybe_instruction_text(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in (
            "instruction",
            "high_level_instruction",
            "subtask_instruction",
            "sub_task_instruction",
            "task",
            "label",
            "description",
            "text",
            "comment",
        ):
            if key in value and value[key] is not None:
                text = _stringify(value[key]).strip()
                if text:
                    return text
        detail = value.get("frame_detail")
        if isinstance(detail, Mapping):
            return _maybe_instruction_text(detail)
        return None
    text = _stringify(value).strip() if value is not None else ""
    return text or None


def _as_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _stringify(value: Any) -> str:
    if value is None:
        raise ValueError("prompt source returned None")
    value = _as_scalar(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value if isinstance(value, str) else str(value)


__all__ = ["PromptSource"]
