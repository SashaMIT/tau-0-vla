"""Qwen3.5 VLM wrapper for Tau VLA joint attention.

Adapted from transformers.models.qwen3_5.modeling_qwen3_5.
Key modifications:
  - Qwen3_5AttentionForVLM: adds compute_kqv / output_atten two-phase interface
  - Qwen3_5DecoderLayerForVLM: routes full_attention layers through two-phase path,
    linear_attention layers through standard GatedDeltaNet
  - Qwen3_5VLMTextModel: uses Qwen3_5DecoderLayerForVLM
  - Qwen3_5_VLForConditionalGeneration: top-level VLM with custom text model
"""

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    BaseModelOutputWithPooling,
    CausalLMOutputWithPast,
    ModelOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
    can_return_tuple,
    logging,
)
from transformers.utils.generic import is_flash_attention_requested, maybe_autocast, merge_with_config_defaults
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available
from transformers.utils.output_capturing import capture_outputs

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

if is_flash_linear_attention_available():
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
else:
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
    FusedRMSNormGated = None

logger = logging.get_logger(__name__)


# ===========================================================================
# Re-export unchanged classes from transformers.models.qwen3_5
# ===========================================================================
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5PreTrainedModel,
    Qwen3_5RMSNorm,
    Qwen3_5RMSNormGated,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionAttention,
    Qwen3_5VisionBlock,
    Qwen3_5VisionMLP,
    Qwen3_5VisionModel,
    Qwen3_5VisionPatchEmbed,
    Qwen3_5VisionPatchMerger,
    Qwen3_5VisionRotaryEmbedding,
    apply_mask_to_padding_states,
    apply_rotary_pos_emb,
    eager_attention_forward,
    is_fast_path_available,
    repeat_kv,
    rotate_half,
)

# ===========================================================================
# Modified Attention for VLM (two-phase: compute_kqv / output_atten)
# ===========================================================================


class Qwen3_5AttentionForVLM(nn.Module):
    """Qwen3.5 attention adapted for VLA joint attention.

    Supports two modes beyond standard forward:
      - compute_kqv=True: returns (Q, K, V, gate) without computing attention
      - output_atten=True: takes pre-computed attention output, applies gate and o_proj
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim * 2, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Standard forward (not used in joint attention path)."""
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ===========================================================================
# Modified DecoderLayer for VLM (two-phase support)
# ===========================================================================


class Qwen3_5DecoderLayerForVLM(GradientCheckpointingLayer):
    """Qwen3.5 decoder layer adapted for VLA.

    For full_attention layers: supports compute_kqv / output_atten two-phase calls.
    For linear_attention layers: uses GatedDeltaNet normally.
    """

    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_type = config.layer_types[layer_idx]
        self.layer_idx = layer_idx

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_5AttentionForVLM(config, layer_idx)

        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        att_output: torch.Tensor | None = None,
        start: int = 0,
        end: int = 0,
        compute_kqv: bool = False,
        output_atten: bool = False,
        gate: torch.Tensor | None = None,
        # Standard args for linear_attention / normal forward
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ):
        if compute_kqv:
            assert self.layer_type == "full_attention", (
                f"compute_kqv called on {self.layer_type} layer {self.layer_idx}"
            )

            normed = self.input_layernorm(hidden_states)
            input_shape = normed.shape[:-1]
            hidden_shape = (*input_shape, -1, self.self_attn.head_dim)

            # q_proj outputs [Q, gate] interleaved per head
            query_states, gate_out = torch.chunk(
                self.self_attn.q_proj(normed).view(*input_shape, -1, self.self_attn.head_dim * 2),
                2,
                dim=-1,
            )
            gate_out = gate_out.reshape(*input_shape, -1)  # [B, L, num_heads * head_dim]

            query_states = self.self_attn.q_norm(query_states.view(hidden_shape))  # [B, L, H, D]
            key_states = self.self_attn.k_norm(self.self_attn.k_proj(normed).view(hidden_shape))  # [B, L, Hkv, D]
            value_states = self.self_attn.v_proj(normed).view(hidden_shape)  # [B, L, Hkv, D]

            return query_states, key_states, value_states, gate_out

        elif output_atten:
            assert self.layer_type == "full_attention", (
                f"output_atten called on {self.layer_type} layer {self.layer_idx}"
            )

            if att_output.dtype != self.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(self.self_attn.o_proj.weight.dtype)

            # Apply sigmoid(gate) BEFORE o_proj (gate is in attention dim, not hidden dim)
            sliced = att_output[:, start:end]
            if gate is not None:
                sliced = sliced * torch.sigmoid(gate)
            out_emb = self.self_attn.o_proj(sliced)
            out_emb = out_emb + hidden_states

            # MLP
            residual = out_emb
            out_emb = self.post_attention_layernorm(out_emb)
            out_emb = self.mlp(out_emb)
            out_emb = residual + out_emb

            # Ensure output dtype matches layer parameters (softmax/sigmoid may upcast to float32)
            target_dtype = self.self_attn.o_proj.weight.dtype
            if out_emb.dtype != target_dtype:
                out_emb = out_emb.to(target_dtype)

            return out_emb

        else:
            # Standard forward (used for linear_attention layers)
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            if self.layer_type == "linear_attention":
                # Cast to match linear_attn weight dtype (float32 can leak from
                # softmax/sigmoid in the preceding full_attention output_atten path)
                param_dtype = next(self.linear_attn.parameters()).dtype
                if hidden_states.dtype != param_dtype:
                    hidden_states = hidden_states.to(param_dtype)
                    residual = residual.to(param_dtype)
                hidden_states = self.linear_attn(
                    hidden_states=hidden_states,
                    cache_params=past_key_values,
                    attention_mask=attention_mask,
                )
            elif self.layer_type == "full_attention":
                hidden_states, _ = self.self_attn(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = residual + hidden_states

            return hidden_states


# ===========================================================================
# VLM Text Model using modified decoder layers
# ===========================================================================


class Qwen3_5VLMTextModel(Qwen3_5PreTrainedModel):
    """Qwen3.5 text model with Qwen3_5DecoderLayerForVLM layers."""

    config_class = Qwen3_5TextConfig

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayerForVLM(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()


# ===========================================================================
# Top-level VLM (Qwen3_5 VL for Conditional Generation)
# ===========================================================================


class Qwen3_5_VLModel(Qwen3_5PreTrainedModel):
    """Composite model: visual + language_model."""

    base_model_prefix = "model"
    config_class = Qwen3_5Config

    def __init__(self, config: Qwen3_5Config):
        super().__init__(config)
        self.visual = Qwen3_5VisionModel._from_config(config.vision_config)
        self.language_model = Qwen3_5VLMTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: list[int, int, int] | torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None,
    ):
        # Kept byte-for-byte identical to transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5VLModel.get_vision_position_ids.
        """
        Compute 3D positional indices for vision tokens derived from a single image or video input.

        The positions are generated from the input grid defined by temporal (T), height (H), and
        width (W) dimensions. Temporal and spatial dimensions can be downscaled according to the
        merge sizes used in the vision backbone. The resulting positions are offset by `start_position`.

        Args:
            start_position (`int`):
                Offset added to all computed positional indices.
            grid_thw (`Sequence[int]` or `torch.Tensor` of shape `(3,)`):
                The (T, H, W) grid representing the feature layout of the current image or video after patch embedding.
            temp_merge_size (`int`, *optional*):
                Factor by which the temporal dimension is reduced in the backbone. The temporal grid size is divided
                by this value. Defaults to 1.
            spatial_merge_size (`int`, *optional*):
                Factor by which the spatial dimensions (H and W) are reduced in the backbone. Both H and W are divided
                by this value. Defaults to 1.
            time_interval (`int`, *optional*):
                Spacing factor applied between consecutive temporal position indices.Defaults to 1.
            device (`str` or `torch.device`, *optional*):
                Device on which the resulting tensor is allocated. If `None`, uses the current default device.

        Returns:
            torch.LongTensor of shape (3, sequence_length):
                Positional indices for temporal, height, and width dimensions,
                flattened into sequence form and offset by `start_position`.
        """
        llm_grid_t, llm_grid_h, llm_grid_w = (
            grid_thw[0].item() // temp_merge_size,
            grid_thw[1].item() // spatial_merge_size,
            grid_thw[2].item() // spatial_merge_size,
        )

        image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
        position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(
            llm_grid_h * llm_grid_t
        )
        position_height = torch.arange(start_position, start_position + llm_grid_h, device=device).repeat_interleave(
            llm_grid_w * llm_grid_t
        )
        position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long)
        position_temporal = position_temporal * time_interval
        vision_position_ids = torch.stack([position_temporal, position_height, position_width], dim=0)

        return vision_position_ids

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Qwen3.5 uses timestamp-separated video placeholders:
        # <t1><vision_start><frame1_pads><vision_end><t2><vision_start><frame2_pads>...
        # so mm_token_type_ids produces T separate "video" groups per video.
        # Expand each [T, H, W] row into T rows of [1, H, W] to match.
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        spatial_merge_size = self.config.vision_config.spatial_merge_size

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                else:
                    grid_thw = next(grid_iters[modality_type])
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor | None = None,
        **kwargs,
    ):
        # Use patch_embed weight to determine dtype (robust under DataParallel replication)
        target_dtype = self.visual.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.type(target_dtype)
        vision_output: BaseModelOutputWithPooling = self.visual(
            pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs
        )
        image_embeds = vision_output.pooler_output
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor | None = None,
        **kwargs,
    ):
        """Returns flat [total_video_tokens, hidden_dim] so callers can mask-assign directly.

        Unlike get_image_features (which splits per image and stack is used downstream),
        videos can have different T across batch so we keep the flat pooler_output.
        """
        target_dtype = self.visual.patch_embed.proj.weight.dtype
        pixel_values_videos = pixel_values_videos.type(target_dtype)
        vision_output: BaseModelOutputWithPooling = self.visual(
            pixel_values_videos, grid_thw=video_grid_thw, return_dict=True, **kwargs
        )
        return vision_output.pooler_output


class Qwen3_5_VLForConditionalGeneration(Qwen3_5PreTrainedModel, GenerationMixin):
    """Qwen3.5 VL model using custom VLM text layers for VLA joint attention."""

    config_class = Qwen3_5Config
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    def __init__(self, config: Qwen3_5Config):
        super().__init__(config)
        self.model = Qwen3_5_VLModel(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(config.text_config.hidden_size, self.vocab_size, bias=False)
        self.post_init()

    @property
    def visual(self):
        return self.model.visual

    @property
    def language_model(self):
        return self.model.language_model

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_image_features(self, pixel_values, image_grid_thw=None, **kwargs):
        return self.model.get_image_features(pixel_values, image_grid_thw, **kwargs)

    def get_video_features(self, pixel_values_videos, video_grid_thw=None, **kwargs):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw, **kwargs)

    def _build_mm_token_type_ids(self, input_ids: torch.LongTensor) -> torch.IntTensor:
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int)
        image_token_id = getattr(self.config, "image_token_id", None)
        if image_token_id is not None:
            mm_token_type_ids[input_ids == image_token_id] = 1
        video_token_id = getattr(self.config, "video_token_id", None)
        if video_token_id is not None:
            mm_token_type_ids[input_ids == video_token_id] = 2
        return mm_token_type_ids

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if not is_first_iteration and use_cache:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _prepare_position_ids_for_generation(self, inputs_tensor, model_kwargs):
        text_positions = super()._prepare_position_ids_for_generation(inputs_tensor, model_kwargs)

        past_length = 0
        if (cache := model_kwargs.get("past_key_values")) is not None:
            past_length = cache.get_seq_length()
        if past_length != 0 and self.model.rope_deltas is not None:
            position_ids = text_positions[None, ...] + self.model.rope_deltas
            return position_ids

        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
        has_mm = model_kwargs.get("image_grid_thw") is not None or model_kwargs.get("video_grid_thw") is not None

        if is_input_ids and has_mm:
            mm_token_type_ids = model_kwargs.get("mm_token_type_ids")
            if mm_token_type_ids is None:
                mm_token_type_ids = self._build_mm_token_type_ids(inputs_tensor)
            vision_positions, rope_deltas = self.model.get_rope_index(
                inputs_tensor,
                mm_token_type_ids,
                image_grid_thw=model_kwargs.get("image_grid_thw"),
                video_grid_thw=model_kwargs.get("video_grid_thw"),
                attention_mask=model_kwargs.get("attention_mask"),
            )
            self.model.rope_deltas = rope_deltas
        else:
            vision_positions = text_positions.unsqueeze(0).expand(3, -1, -1)
            self.model.rope_deltas = torch.zeros(
                inputs_tensor.shape[0], 1, dtype=torch.long, device=inputs_tensor.device
            )

        text_positions = text_positions[None, ...]
        position_ids = torch.cat([text_positions, vision_positions], dim=0)
        return position_ids

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Standard VLM forward for VQA / FAST co-training (no joint attention).

        Mirrors the HF Qwen3_5ForConditionalGeneration.forward() logic:
        vision encoding → embed merge → language model layers → lm_head → CE loss.
        """
        lang_model = self.model.language_model

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config.text_config)

        # 1. Build input embeddings (merge vision tokens into text embeddings)
        inputs_embeds = lang_model.embed_tokens(input_ids)

        if pixel_values is not None and image_grid_thw is not None:
            image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
            # image_embeds is a list of tensors, one per image; concat into single tensor
            image_embeds = torch.cat(image_embeds, dim=0)
            image_token_id = getattr(self.config, "image_token_id", None)
            if image_token_id is not None:
                image_mask = input_ids == image_token_id
                n_image_tokens = image_mask.sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    expected_features = (
                        (image_grid_thw.prod(-1) // (self.model.visual.spatial_merge_size**2)).sum().item()
                    )
                    raise ValueError(
                        "Image features and image tokens do not match: "
                        f"tokens={n_image_tokens}, features={n_image_features}, "
                        f"expected_features_from_grid={expected_features}, "
                        f"num_images={image_grid_thw.shape[0]}, input_ids_shape={tuple(input_ids.shape)}"
                    )
                image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, image_embeds.to(inputs_embeds.dtype))

        if pixel_values_videos is not None and video_grid_thw is not None:
            video_embeds = self.model.get_video_features(pixel_values_videos, video_grid_thw)
            # flat [total_video_tokens, D]
            video_token_id = getattr(self.config, "video_token_id", None)
            if video_token_id is not None:
                video_mask = input_ids == video_token_id
                n_video_tokens = video_mask.sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    expected_features = (
                        (video_grid_thw.prod(-1) // (self.model.visual.spatial_merge_size**2)).sum().item()
                    )
                    raise ValueError(
                        "Video features and video tokens do not match: "
                        f"tokens={n_video_tokens}, features={n_video_features}, "
                        f"expected_features_from_grid={expected_features}, "
                        f"num_videos={video_grid_thw.shape[0]}, input_ids_shape={tuple(input_ids.shape)}"
                    )
                video_mask_expanded = video_mask.unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask_expanded, video_embeds.to(inputs_embeds.dtype))

        # 2. Compute MRoPE position_ids if not provided
        if position_ids is None:
            if mm_token_type_ids is None:
                mm_token_type_ids = self._build_mm_token_type_ids(input_ids)
            position_ids, _ = self.model.get_rope_index(
                input_ids, mm_token_type_ids, image_grid_thw, video_grid_thw, attention_mask
            )

        # 3. Decompose position_ids and compute RoPE
        hidden_states = inputs_embeds

        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            pos_3d = position_ids[1:]
        elif position_ids.ndim == 3 and position_ids.shape[0] == 3:
            text_position_ids = None
            pos_3d = position_ids
        elif position_ids.ndim == 3:
            text_position_ids = None
            pos_3d = position_ids.expand(3, -1, -1)
        else:
            text_position_ids = None
            pos_3d = position_ids.unsqueeze(0).expand(3, -1, -1)

        pos_1d = pos_3d[0] if text_position_ids is None else text_position_ids
        rope = lang_model.rotary_emb(hidden_states, pos_3d)

        # 4. Build masks
        text_config = self.config.text_config if hasattr(self.config, "text_config") else self.config

        causal = create_causal_mask(
            config=text_config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=pos_1d,
        )

        linear_attn_mask = attention_mask
        if (past_key_values is not None and past_key_values.has_previous_state()) or (
            attention_mask is not None and torch.all(attention_mask == 1)
        ):
            linear_attn_mask = None

        # 5. Run through language model layers
        for layer in lang_model.layers:
            layer_mask = linear_attn_mask if layer.layer_type == "linear_attention" else causal
            layer_out = layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=pos_1d,
                position_embeddings=rope,
                past_key_values=past_key_values,
            )
            hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        hidden_states = lang_model.norm(hidden_states)

        # 6. Compute logits and loss
        loss = None
        if labels is not None:
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            flat_hidden = shift_hidden.view(-1, shift_hidden.size(-1))
            flat_labels = shift_labels.view(-1)
            valid_mask = flat_labels != -100
            if valid_mask.any():
                logits = F.linear(
                    flat_hidden[valid_mask],
                    self.lm_head.weight,
                    self.lm_head.bias if self.lm_head.bias is not None else None,
                )
                loss = F.cross_entropy(logits, flat_labels[valid_mask])
            else:
                loss = hidden_states.sum() * 0.0  # no valid tokens; zero loss with grad
            logits = None
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        return CausalLMOutputWithPast(loss=loss, logits=logits)
