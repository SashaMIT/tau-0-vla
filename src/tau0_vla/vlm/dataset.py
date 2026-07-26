"""Finch-backed VLA dataset adapter.

This module is the only bridge between ``tau0_vla.data`` (data domain) and the
tau-0-vla policy layer. Responsibilities:

1. Take a ``data_args`` dataclass, resolve the Finch ``@register_config``
   named ``data_args.config_name`` via ``tau0_vla.data.FinchDataLoader``, pull
   the underlying ``torch.utils.data.Dataset`` that yields
   ``canonical_payload`` samples (``{prompt, images, state, action, meta}``).
2. Wrap each sample with VLM-specific encoding (Qwen-VL tokenization,
   state/action tensors) via the existing
   ``tau0_vla.vlm.qwenvl_utils.build_vla_inference_data_dict``.
3. Expose a ``save_data_spec(run_dir)`` helper so the trainer writes the
   Finch data contract into the checkpoint directory — the same file
   ``deploy/policy.py::Tau0VLAPolicy.from_checkpoint`` reads back via
   ``tau0_vla.data.load_data_spec``.

Design principles (embodiment-specific knowledge lives in the adapter
for the general-pipeline / VLM-side split):
- Only the public API surface re-exported in ``tau0_vla/data/__init__.py`` is
  consumed. Nothing reaches into ``tau0_vla.data.pipeline._*``.
- Semantic action/state transforms (Normalize, Delta, Joint2Eef, etc.)
  live *declaratively* inside the Finch ``@register_config`` — this module
  never constructs per-sample transform instances.
- VLM encoding is kept in tau-0-vla, never pushed into ``tau0_vla.data``:
  a VLM-only concern does not belong in the general data pipeline.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image as PILImage

from tau0_vla.data import FinchDataLoader, build_transforms, compose
from tau0_vla.data.config import discover_config_modules

# Importing policy_transforms fires the ``@register_transform`` decorators that
# ``tau0_vla.data.build_transforms`` needs when yaml specs reference them
# (``ColorJitterImages`` and any future policy-only transform).
from tau0_vla.vlm import policy_transforms as _policy_transforms  # noqa: F401
from tau0_vla.vlm._image_utils import to_uint8_hwc
from tau0_vla.vlm.collator import DataCollatorForVLMSupervisedDataset
from tau0_vla.vlm.qwenvl_utils import build_vla_inference_data_dict
from tau0_vla.vlm.sharding import attach_shard_info, get_dataset_shard_info

_CANONICAL_CAM_PRIORITY = ("head", "front", "ego", "wrist_left", "wrist_right", "left", "right")
# Corruption guard ONLY — do not lower to trim distribution tails. Temporal-shape
# analysis of every >5 sample in the stage-2 occupancy probe (2026-07-07) showed
# 100% smooth anchor-relative growth at plausible speeds (genrobot chunk-end
# 0.16-1.5 m/s), i.e. the 3-10 band is the FASTEST REAL MOTION, not glitches;
# a 5.0 cutoff silently dropped 11.2% of genrobot / 8% SOP / 6% yam-joint.
_NORM_ABS_MAX_THRESHOLD = 10.0


def _canonical_cam_order(keys: list[str]) -> list[str]:
    """Sort camera keys by semantic priority for deterministic ordering."""

    def _priority(k: str) -> int:
        for i, prefix in enumerate(_CANONICAL_CAM_PRIORITY):
            if prefix in k:
                return i
        return len(_CANONICAL_CAM_PRIORITY)

    return sorted(keys, key=_priority)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


class FinchVLADataset(torch.utils.data.Dataset):
    """Finch-backed supervised VLA dataset.

    The heavy lifting (frame decoding, normalization, padding, prompt
    resolution, chunking) is done by the underlying
    ``tau0_vla.data``-built ``torch.utils.data.Dataset``. This class only adds
    VLM-side encoding on top of each canonical payload.
    """

    # Max re-rolls when ``vla_max_token_len`` filtering rejects a long sample
    # before giving up (and passing one through) so a too-aggressive threshold
    # can't recurse forever.
    _VLA_LEN_FILTER_MAX_REROLL: int = 10

    def __init__(self, processor, data_args) -> None:
        super().__init__()
        self.processor = processor
        self.data_args = data_args
        self.vlm_model_type = data_args.vlm_model_type

        self.shard_info = get_dataset_shard_info(data_args)
        self._loader = _build_finch_loader(data_args, self.shard_info)
        self._underlying: torch.utils.data.Dataset = self._loader.dataset
        self._source_config = getattr(self._loader, "_source_config", None)
        # Mirror the legacy ``LeRobotDataset.physically_split`` flag so that
        # ``trainer.py`` can see ``getattr(ds, "physically_split", False)``
        # and skip ``accelerator.prepare()`` to avoid double-sharding.
        attach_shard_info(self, self.shard_info)
        self._validate_non_empty_dataset()

        # Cameras come from the Finch config's canonical payload
        # ("head" / "wrist_left" / "wrist_right" for G1Agibot). If the yaml
        # pins a subset, honor that order; otherwise probe the first sample.
        self.cam_keys: List[str] = _resolve_cam_keys(
            self._underlying,
            data_args.camera_keys,
            source_config=self._source_config,
        )
        self._camera_keys_pinned = bool(data_args.camera_keys)
        self.cam_view_template = getattr(data_args, "cam_view_template", None)
        self.cam_view_names = getattr(data_args, "cam_view_names", None) or {}
        if self.cam_view_template is not None:
            assert self.cam_view_template.count("<image>") == 1, (
                f"cam_view_template must contain exactly one '<image>', got: {self.cam_view_template!r}"
            )

        # Policy-side VLM-aux transforms declared in ``data_args.transforms`` (yaml).
        # ``policy_transforms`` has already registered them via ``@register_transform``
        # at module import time; ``build_transforms`` instantiates each spec by name.
        self.vlm_aux_transforms = compose(build_transforms(data_args.transforms)) if data_args.transforms else None

        # Per-sample safety clamp for VLM context length control.
        self.max_images_per_sample: Optional[int] = data_args.max_images_per_sample

        # Opt-in VLA length filter: drop samples whose tokenized prefix exceeds
        # this, before they pad the micro-batch and OOM a rank. None = disabled.
        self.vla_max_token_len: Optional[int] = getattr(data_args, "vla_max_token_len", None)

        # Per-route corruption-guard override (see DataArguments.norm_abs_max_threshold).
        override = getattr(data_args, "norm_abs_max_threshold", None)
        self.norm_abs_max_threshold: float = float(override) if override is not None else _NORM_ABS_MAX_THRESHOLD

    def __len__(self) -> int:
        return len(self._underlying)

    def __getitem__(self, i: int, _len_filter_depth: int = 0) -> Dict[str, torch.Tensor]:
        # ``_len_filter_depth`` is internal (DataLoader only ever passes ``i``):
        # it bounds the re-roll recursion of the ``vla_max_token_len`` filter.
        # Retry on transient video-decode failures. torchcodec on h265 has
        # been observed to raise "Could not push packet to decoder: Invalid
        # data found when processing input" on rare random-access patterns
        # under multi-worker load (ffmpeg decodes the same files cleanly, so
        # this is a decoder-state hiccup, not corrupt data). Falling through
        # to a different sample is safer than crashing the whole DataLoader.
        last_exc: Optional[BaseException] = None
        attempts = 50
        n = len(self._underlying)
        for attempt in range(attempts):
            try:
                raw = self._underlying[i]
                # Skip corrupt samples with extreme normalized values
                state_arr = raw.get("state")
                action_arr = raw.get("action")
                if state_arr is not None and np.abs(np.asarray(state_arr)).max() > self.norm_abs_max_threshold:
                    i = random.randint(0, n - 1)
                    continue
                if action_arr is not None and np.abs(np.asarray(action_arr)).max() > self.norm_abs_max_threshold:
                    i = random.randint(0, n - 1)
                    continue
                break
            except RuntimeError as exc:
                last_exc = exc
                i = random.randint(0, n - 1)
        else:
            if last_exc is not None:
                raise last_exc
            return self.__getitem__(random.randint(0, n - 1))

        # Policy-side transforms run on the raw payload BEFORE VLM encoding.
        if self.vlm_aux_transforms is not None:
            raw = self.vlm_aux_transforms(raw)

        instruction = str(raw["prompt"])
        images_dict: Dict[str, np.ndarray] = raw["images"]
        sample_cam_keys = self._resolve_sample_cam_keys(images_dict)
        if self.max_images_per_sample is not None:
            sample_cam_keys = sample_cam_keys[: self.max_images_per_sample]
        images_np = [np.asarray(images_dict[key]) for key in sample_cam_keys]
        images_pil = [PILImage.fromarray(to_uint8_hwc(img)) for img in images_np]

        data_dict = build_vla_inference_data_dict(
            processor=self.processor,
            images_pil=images_pil,
            instruction=instruction,
            vlm_model_type=self.vlm_model_type,
            cam_view_labels=self._render_cam_view_labels(sample_cam_keys),
        )

        # Tensor payload consumed by the VLA action expert.
        state = _as_float_tensor(raw["state"]).reshape(1, -1)
        action = _as_float_tensor(raw["action"])
        data_dict["state"] = state
        data_dict["action"] = action
        data_dict["state_mask"] = _as_float_tensor(raw["state_mask"]) if raw.get("state_mask") is not None else None
        data_dict["action_mask"] = _as_float_tensor(raw["action_mask"]) if raw.get("action_mask") is not None else None
        data_dict["task_ids"] = torch.tensor([0], dtype=torch.int32)
        # Best-effort source id for oversized-sample diagnostics (consumed by
        # DataCollatorForVLMSupervisedDataset). The data pipeline attaches a
        # ``meta`` payload whose schema varies by source, so stringify defensively.
        try:
            data_dict["profile_source"] = str(raw.get("meta"))[:200]
        except Exception:
            data_dict["profile_source"] = None

        # Opt-in VLA length filter (see data_args.vla_max_token_len). Drop samples
        # whose tokenized prefix exceeds the threshold by re-rolling to another
        # index, so one long sample can't pad the bs=28 micro-batch up and OOM a
        # rank. Compared against input_ids length -- the same quantity the
        # [LONG_SAMPLE] collator log reports. Bounded re-rolls avoid infinite
        # recursion if the threshold is set too aggressively for this shard.
        if self.vla_max_token_len is not None:
            seq_len = int(data_dict["input_ids"].shape[-1])
            if seq_len > self.vla_max_token_len:
                if _len_filter_depth < self._VLA_LEN_FILTER_MAX_REROLL:
                    return self.__getitem__(
                        random.randint(0, n - 1), _len_filter_depth=_len_filter_depth + 1
                    )
                # Could not find a short enough sample -- pass this one through
                # rather than recurse forever, but make it loud.
                print(
                    f"[VLA_LEN_FILTER] rank={os.environ.get('RANK', '?')} no sample "
                    f"<= {self.vla_max_token_len} tokens after {self._VLA_LEN_FILTER_MAX_REROLL} "
                    f"re-rolls (last seq_len={seq_len}); passing it through",
                    flush=True,
                )
        return data_dict

    def save_data_spec(self, run_dir: str) -> None:
        """Write ``<run_dir>/finch_data_spec/<config_name>/`` so that deploy
        can reconstruct the exact data contract used at training time."""
        self._loader.save_data_spec(
            run_dir,
            config_modules=tuple(self.data_args.config_modules or ()),
            vlm_model_type=str(self.vlm_model_type),
            vlm_aux_transforms=tuple(self.data_args.transforms or ()),
            cam_keys=tuple(self.cam_keys),
            cam_view_template=self.cam_view_template,
            cam_view_names=dict(self.cam_view_names or {}),
            max_images_per_sample=self.max_images_per_sample,
        )

    def _render_cam_view_labels(self, cam_keys: list[str]) -> list[str] | None:
        if self.cam_view_template is None:
            return None
        return [self.cam_view_template.format(self.cam_view_names.get(key, key.rsplit(".", 1)[-1])) for key in cam_keys]

    def _resolve_sample_cam_keys(self, images_dict: Dict[str, np.ndarray]) -> list[str]:
        if self._camera_keys_pinned:
            sample_keys = [key for key in self.cam_keys if key in images_dict]
            if not sample_keys:
                raise KeyError(
                    f"Finch sample has no cameras matching pinned camera_keys={self.cam_keys}; "
                    f"available={list(images_dict)}"
                )
            return sample_keys

        return _canonical_cam_order(list(images_dict.keys())) if images_dict else list(self.cam_keys)

    def _validate_non_empty_dataset(self) -> None:
        dataset_len = len(self._underlying)
        if dataset_len > 0:
            return

        source_config = self._source_config
        prompt_source = getattr(source_config, "prompt_source", None)
        uses_instruction_segments = (
            bool(prompt_source.uses_instruction_segments()) if prompt_source is not None else False
        )
        uses_segment_filter = bool(getattr(source_config, "filter_by_segments", False))

        raise ValueError(
            "FinchVLADataset resolved to zero samples. "
            f"config_name={getattr(self.data_args, 'config_name', None)!r}, "
            f"physically_split={self.physically_split}, "
            f"config.filter_by_segments={uses_segment_filter}, "
            f"prompt_source_uses_instruction_segments={uses_instruction_segments}. "
            "This usually means all frames were filtered out by segment filtering or the selected repos "
            "contain no valid annotated training windows."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Collators
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class DataCollatorForVLASupervisedDataset(DataCollatorForVLMSupervisedDataset):
    """Stacks VLA-specific per-sample keys on top of the VLM collation."""

    def _process_action_data(self, instances: Sequence[Dict], batch: Dict) -> Dict:
        for key in ("action", "task_ids", "state", "ctrl_freqs"):
            if key in instances[0]:
                batch[key] = torch.stack([instance[key] for instance in instances])
        for mask_key in ("state_mask", "action_mask"):
            if instances[0].get(mask_key) is not None:
                batch[mask_key] = torch.stack([instance[mask_key] for instance in instances])
        return batch

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(instances)
        batch = self._process_action_data(instances, batch)
        return batch


# ─────────────────────────────────────────────────────────────────────────────
# Public entry points consumed by trainer.train
# ─────────────────────────────────────────────────────────────────────────────


def make_supervised_finch_data_module(processor, data_args) -> Dict:
    """Build the VLA data module that ``trainer.train`` registers under the
    ``"vla"`` key in its ``DATA_MODULE_REGISTRY``."""
    train_dataset = FinchVLADataset(processor, data_args=data_args)
    data_collator = DataCollatorForVLASupervisedDataset(processor.tokenizer)
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": data_collator,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_finch_loader(data_args, shard_info=None) -> FinchDataLoader:
    """Resolve ``data_args.config_name`` through ``tau0_vla.data`` and return a
    loader. We only use its ``.dataset`` (for HF Trainer to wrap in its own
    DataLoader) and its ``save_data_spec`` helper."""
    if not data_args.config_name:
        raise ValueError(
            "FinchVLADataset requires data_args.config_name to be set "
            "(tau0_vla.data @register_config name). None given."
        )
    if data_args.config_modules:
        discover_config_modules(module_names=list(data_args.config_modules))
    if shard_info is None:
        shard_info = get_dataset_shard_info(data_args)

    # batch_size=None + num_workers=0 because HF Trainer will wrap the
    # underlying dataset with its own DataLoader. We only need the Dataset
    # plumbed correctly.
    return FinchDataLoader.from_config_name(
        data_args.config_name,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        physically_split=shard_info.physically_split,
        physical_rank=shard_info.physical_rank if shard_info.physically_split else None,
        physical_world_size=shard_info.physical_world_size if shard_info.physically_split else None,
    )


def _resolve_cam_keys(
    underlying: torch.utils.data.Dataset,
    override: Optional[List[str]],
    *,
    source_config: Any | None = None,
) -> List[str]:
    """Pick the cam_keys to surface in ``__getitem__``.

    - If the yaml pinned ``camera_keys``, honor that order.
    - Otherwise prefer the finch config's canonical ``repack["images"]``
      declaration, which is stable even when segment filtering leaves
      a child repo with zero valid frames.
    - Only as a last resort, probe the first sample if the dataset is non-empty.
    """
    if override:
        return list(override)

    camera_map = getattr(source_config, "repack", {}).get("images", {}) if source_config is not None else {}
    if camera_map:
        return list(camera_map.keys())

    if len(underlying) <= 0:
        raise ValueError(
            "Unable to resolve finch camera keys: dataset is empty after filtering and no camera_keys/config repack "
            "declaration was provided."
        )

    probe = underlying[0]
    return list(probe["images"].keys())


def _as_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)
