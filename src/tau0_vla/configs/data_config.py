from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DataArguments:
    """Data-layer configuration for the finch-backed training pipeline.

    Fields are grouped:
      - VLM image/video/token knobs (shared across all dataset kinds).
      - Finch dataset knobs. These drive ``FinchVLADataset``, which builds its
        pipeline through ``RobotConfig.build_pipeline``. Concrete state /
        action semantics live in the ``@register_config`` referenced by
        ``config_name`` — the fields here only control dataset wiring
        (filtering, action horizon, camera overrides, etc.).
      - VLM-aux transforms list. Pure ``type``-dispatched specs consumed by
        ``tau0_vla.data.build_transforms``; semantic transforms (Normalize,
        Joint2Eef, Abs2Delta, FrankaGripper, …) are expressed declaratively
        inside the Finch config and MUST NOT appear here.
    """

    # ── VLM encoding knobs ───────────────────────────────────────────────────
    # vlm_model_type is owned by ModelArguments; ModelBuilder.__init__ copies
    # it onto the data_args instance as a runtime attribute so dataset code
    # can read ``data_args.vlm_model_type``. Declaring it here would clash
    # with ModelArguments' CLI flag under HfArgumentParser.

    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    image_resize_hw: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": (
                "Resize images to [H, W] before the Qwen3VL image processor. "
                "Ensures train/inference resolution parity. None = defer to processor's "
                "min/max_pixels based dynamic resize."
            )
        },
    )
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=64 * 28 * 28)
    video_min_pixels: int = field(default=64 * 28 * 28)
    video_fps: float = 2

    max_images_per_sample: Optional[int] = field(
        default=None,
        metadata={"help": "Per-sample image cap; samples exceeding it are truncated to prevent OOM. None = no limit."},
    )
    vla_max_token_len: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "VLA-only length filter: drop (re-roll) any VLA sample whose tokenized input_ids prefix exceeds "
                "this many tokens, before it can pad the whole micro-batch and OOM a rank. Compared against the "
                "same length the [LONG_SAMPLE] collator log reports. None = disabled (no filtering). Set after "
                "inspecting the [LONG_SAMPLE] length distribution; usually < model_max_length."
            )
        },
    )
    norm_abs_max_threshold: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Per-route override of the normalized state/action corruption guard (|z| cutoff, default 10.0). "
                "Raise it when a legitimate sparse dim exceeds the default — e.g. franka_dresser ft gripper "
                "toggles normalize to z≈11.6 under the reused mean_std stats, so the default silently drops "
                "every chunk containing a right-gripper event. Keep it a corruption guard: only as high as the "
                "route's fastest real motion requires."
            )
        },
    )
    vlm_random_fallback_retries: int = field(
        default=12,
        metadata={
            "help": (
                "Number of random replacement samples to try when a VLM sample is corrupted, text/vision-truncated, "
                "or otherwise invalid. Default 12 preserves the historical num_base_retries * 4 behavior."
            )
        },
    )

    supervise_thinking: bool = field(
        default=True,
        metadata={"help": "Whether <think>...</think> tokens are supervised. Only relevant for Qwen3.5 / CoT data."},
    )

    padding_side: str = field(default="right")

    # ── Finch dataset: config resolution ─────────────────────────────────────
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of a @register_config registered with tau0_vla.data. Single-config case."},
    )
    config_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Multiple @register_config names. Mutually exclusive with config_name."},
    )
    config_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Python module dotted paths that call @register_config at import time."},
    )

    # ── Finch dataset: chunking / filtering ─────────────────────────────────
    camera_keys: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Video keys to decode. Defaults to whatever the Finch config declares."},
    )
    cam_view_template: Optional[str] = field(
        default=None,
        metadata={
            "help": "Template for per-view image labels, e.g. '{} view: <image>'. "
            "Must contain exactly one <image>. When None, uses bare '<image>' tokens."
        },
    )
    cam_view_names: Optional[Dict[str, str]] = field(
        default=None,
        metadata={
            "help": "Map camera_key -> human-readable view name, e.g. {'head': 'head', 'wrist_right': 'right hand'}."
        },
    )
    action_horizon: int = field(default=30, metadata={"help": "Action chunk size (horizon, in frames)."})
    state_dim: int = field(default=16, metadata={"help": "Padded state dim the model head expects."})
    action_dim: int = field(default=16, metadata={"help": "Padded action dim the model head expects."})

    # TODO: migrate stats-based episode filtering into the data pipeline's
    # RobotConfig / build_pipeline path, then remove this tau-0-vla-side
    # compatibility knob.
    filter_dataset_by_stats: bool = field(default=True)

    physically_split: bool = field(
        default=False,
        metadata={
            "help": "Each rank loads a disjoint subset of datasets (LPT-balanced by total_frames). "
            "Avoids OOM when one config spans many dataset repos."
        },
    )
    num_physical_shards: int = field(
        default=0,
        metadata={
            "help": "Number of repo-level physical shards when physically_split=True. "
            "0 keeps legacy behavior: split by the full distributed world_size. "
            "A positive value enables grouped physical splitting; ranks with the same "
            "rank % num_physical_shards share one physical shard and are divided by a sampler."
        },
    )

    image_resize: Optional[int] = field(
        default=None,
        metadata={"help": "Optional square resize applied at FinchVLADataset payload time."},
    )

    resolution_kwargs: Optional[Dict[str, Any]] = field(
        default=None,
        metadata={"help": "Extra kwargs forwarded to tau0_vla.data.robot.resolve_finch_robot_configs."},
    )

    data_task_default_prompt: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Per-dataset fallback prompt; consumed by GeneratePrompt when present."},
    )

    # ── Finch dataset: VLM-aux transforms ───────────────────────────────────
    transforms: Optional[List[Dict[str, Any]]] = field(
        default_factory=list,
        metadata={
            "help": "VLM-aux transform specs, applied to the payload after the pipeline "
            "and before VLM encoding. Leave empty for image augmentation: declare "
            "ColorJitter on RobotConfig.images instead, which is deploy-symmetric. "
            "Adding ColorJitterImages here on top of that jitters twice. Semantic "
            "transforms (Normalize, Delta, ...) live in the Finch config, not here."
        },
    )

    # ── Runtime knobs ───────────────────────────────────────────────────────
    # data_seed is mirrored from training_args.seed by trainer/train.py as a
    # runtime attribute; declaring it as a field would clash with
    # TrainingArguments' built-in data_seed flag under HfArgumentParser.
