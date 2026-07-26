"""``TrainingArguments`` — extends the HuggingFace dataclass with LoRA, separate
learning rates per module group, and the CUDA cache-clearing interval.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import transformers

logger = logging.getLogger(__name__)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """tau-0-vla training hyperparameters.

    Extends ``transformers.TrainingArguments`` with:

    - **LoRA finetuning**: enabled by ``lora_enable`` and configured through
      ``lora_r``, ``lora_alpha`` and ``lora_dropout``.
    - **Differential multimodal learning rates**: ``mm_projector_lr`` and
      ``vision_tower_lr`` give the vision-language projector and the vision
      encoder learning rates of their own.
    - **Memory management**: ``disable_python_gc`` turns off automatic Python GC
      in the training process, and ``torch_empty_cache_steps`` sets how often
      collection and CUDA cache clearing happen manually instead.

    Attributes:
        cache_dir: Hugging Face model cache directory.
        optim: Optimizer type. Defaults to ``"adamw_torch"``.
        model_max_length: Maximum tokenizer sequence length; longer sequences are
            truncated or right-padded.
        disable_python_gc: Whether to disable automatic Python GC during training
            and call ``gc.collect()`` on a fixed step interval instead.
        torch_empty_cache_steps: How many global steps between manual
            ``gc.collect()`` and ``torch.cuda.empty_cache()`` calls, which reduce
            GC jitter and ease memory fragmentation. ``None`` disables the manual
            cleanup entirely.
        ddp_find_unused_parameters: Whether DDP searches for unused parameters.
            ``False`` is recommended with LoRA and partially frozen parameters.
        mm_projector_lr: Learning rate for the vision-language projector (the
            merger). ``None`` uses the global learning rate.
        vision_tower_lr: Learning rate for the vision encoder. ``None`` uses the
            global learning rate.
        lora_enable: Whether to finetune with LoRA. Calls ``peft.get_peft_model``
            when enabled.
        lora_r: LoRA rank, setting the low-rank matrices' dimension. Defaults to 64.
        lora_alpha: LoRA scaling factor, conventionally ``lora_r * 2``. Defaults to 128.
        lora_dropout: LoRA dropout probability. Defaults to 0.0.
        lora_weight_path: Optional path to preloaded LoRA weights.
        lora_bias: How LoRA handles biases. Defaults to ``"none"``.
        gradient_checkpointing: Whether to trade compute for memory with gradient
            checkpointing. Defaults to ``false``.
        verbose_logging: Whether to emit detailed debug logs.
        attn_implementation: Attention backend; ``"flash_attention_2"`` recommended.
        remove_unused_columns: Whether to drop unused columns from the batch.
            Defaults to ``False``.

    Example:
        Through a YAML file (recommended)::

            training_args:
              output_dir: /tmp/my_experiment
              per_device_train_batch_size: 2
              lora_enable: true
              lora_r: 64

        Or instantiated directly in Python::

            >>> from tau0_vla.configs.training_config import TrainingArguments
            >>> args = TrainingArguments(
            ...     output_dir="/tmp/output",
            ...     lora_enable=True,
            ... )
    """

    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    disable_python_gc: bool = field(
        default=True,
        metadata={
            "help": (
                "Disable Python automatic garbage collection during training_step in the main training process. "
                "Manual gc.collect() is run together with torch.cuda.empty_cache() based on torch_empty_cache_steps."
            )
        },
    )
    torch_empty_cache_steps: Optional[int] = field(
        default=1000,
        metadata={
            "help": (
                "Number of global_steps between manual memory cleanup calls. "
                "When set, the trainer runs gc.collect() and torch.cuda.empty_cache() once per interval. "
                "Set to None to disable manual cleanup."
            )
        },
    )
    ddp_find_unused_parameters: bool = field(default=False)
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    vla_action_head_lr: Optional[float] = field(
        default=None,
        metadata={
            "help": "Independent learning rate for the VLA action head (DiT, action_encoder, etc.). When None, uses the global learning rate."
        },
    )

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
    lora_weight_path: str = ""
    lora_bias: str = "none"

    ## training config
    gradient_checkpointing: bool = field(default=False)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "Use transformers attention implementation."},
    )

    remove_unused_columns: bool = field(default=False)

    log_metrics_sync_across_ranks: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to all_reduce per-dataset loss / epoch metrics across ranks in log(). "
                "False skips the collective and reports rank0's local values, saving sync latency. "
                "Recommended False when logging_steps is small and world_size is large."
            )
        },
    )
    # DataLoader config
    dataloader_num_workers: int = field(default=4, metadata={"help": "Number of subprocesses to use for data loading."})
    dataloader_pin_memory: bool = field(default=True, metadata={"help": "Pin memory in DataLoader."})
    dataloader_persistent_workers: bool = field(
        default=True, metadata={"help": "Use persistent workers to avoid recreating workers each epoch."}
    )
    dataloader_drop_last: bool = field(default=True, metadata={"help": "Drop last incomplete batch."})
    dataloader_prefetch_factor: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of batches to prefetch per worker. None leaves the "
                "DataLoader default in place and is required when "
                "dataloader_num_workers=0."
            )
        },
    )
    enable_step_profiling: bool = field(
        default=False,
        metadata={
            "help": (
                "Enable lightweight per-step timing logs for data loading and model execution. "
                "Metrics are aggregated over the normal logging window."
            )
        },
    )
    step_profiling_cuda_sync: bool = field(
        default=False,
        metadata={
            "help": (
                "Synchronize CUDA around profiled regions for more accurate forward/backward/optimizer timings. "
                "This improves timing fidelity but adds profiling overhead."
            )
        },
    )
    step_profiling_all_ranks_csv: bool = field(
        default=False,
        metadata={
            "help": (
                "With enable_step_profiling, EVERY rank appends its per-step timings to "
                "output_dir/step_profile/step_profile_rank{RANK}.csv (the logging window in the "
                "console only shows rank0). Use to locate which rank stalls a synchronized step. "
                "optimizer/lr_scheduler times run after training_step and land on the next row."
            )
        },
    )
    vlm_first_batch_pad_to_max: bool = field(
        default=False,
        metadata={
            "help": (
                "CUDA-allocator max-shape warmup: right-pad each rank's FIRST vlm batch to "
                "model_max_length (pad_token/-100/mask-0, loss unchanged) so the allocator grows "
                "to its peak activation footprint at step 1 instead of stalling ~5s whenever a "
                "rank first meets a record-length sample. In synchronous data parallel those "
                "per-rank stalls gate every step at world_size/step_n expected spikes, which is "
                "what made fresh 768-GPU runs ramp for ~2000 steps."
            )
        },
    )

    def __post_init__(self):
        # Fix incompatible DataLoader params BEFORE parent __post_init__ validates them.
        # Parent class raises ValueError if num_workers=0 and prefetch_factor is not None.
        if self.dataloader_num_workers == 0:
            if self.dataloader_prefetch_factor is not None:
                logger.warning(
                    f"dataloader_num_workers=0: setting dataloader_prefetch_factor={self.dataloader_prefetch_factor} -> None"
                )
                self.dataloader_prefetch_factor = None
            if self.dataloader_persistent_workers:
                logger.warning("dataloader_num_workers=0: setting dataloader_persistent_workers=True -> False")
                self.dataloader_persistent_workers = False
        super().__post_init__()
