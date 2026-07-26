import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image as PILImage

# PIL emits a UserWarning when converting palette images (mode "P") with
# byte-encoded transparency directly to "RGB".  This happens inside the
# official qwen_vl_utils.fetch_image → to_rgb path which we cannot modify.
# Suppress it globally; we handle palette images correctly in our own code.
warnings.filterwarnings(
    "ignore",
    message="Palette images with Transparency expressed in bytes",
    category=UserWarning,
)


def _pil_to_rgb(img: PILImage.Image) -> PILImage.Image:
    """Convert any PIL image to RGB without triggering palette-transparency warnings.

    PIL warns when converting a mode-'P' image whose transparency is stored as
    a plain integer/bytes rather than a full alpha channel.  The fix is to
    promote it to RGBA first, which properly expands the transparency, and then
    flatten to RGB.
    """
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    return img.convert("RGB")


from tau0_vla.configs.constants import IGNORE_INDEX
from tau0_vla.utils.utils import rank0_print

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-model-family special token IDs for vision tokens.
# Qwen2-VL / Qwen2.5-VL / Qwen3-VL share the same 151k vocabulary.
# Qwen3.5 uses an enlarged 248k vocabulary with different token IDs.
# ---------------------------------------------------------------------------
VISION_TOKEN_IDS = {
    "qwen2": {
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
    },
    "qwen3.5": {
        "image_token_id": 248056,
        "video_token_id": 248057,
        "vision_start_token_id": 248053,
    },
}
# Qwen2.5-VL and Qwen3-VL share the same vocab as Qwen2-VL
VISION_TOKEN_IDS["qwen2.5"] = VISION_TOKEN_IDS["qwen2"]
VISION_TOKEN_IDS["qwen3"] = VISION_TOKEN_IDS["qwen2"]


def _get_vision_token_ids(model_family: str) -> tuple:
    """Return (image_token_id, video_token_id, vision_start_token_id) for a model family."""
    ids = VISION_TOKEN_IDS.get(model_family)
    if ids is None:
        raise ValueError(
            f"Unknown model family '{model_family}' for vision token IDs. Supported: {list(VISION_TOKEN_IDS.keys())}"
        )
    return ids["image_token_id"], ids["video_token_id"], ids["vision_start_token_id"]


def get_rope_index_3(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Qwen3-VL MRoPE helper aligned with current HF timestamp semantics.

    Current Hugging Face Qwen3-VL separates video frames with timestamp text, so
    each [T, H, W] video grid is split into T frame-level [1, H, W] grids before
    RoPE positions are assigned. This local helper keeps the repo's collator
    interface and scans input_ids by vision token ids instead of using upstream's
    mm_token_type_ids grouping, but the timestamp/video-grid split semantics are
    the same as the official implementation.

    Token IDs use the Qwen2/2.5/3 vocabulary family (151k).
    """
    # Timestamp-separated video layout:
    # <t1> <vision_start> <frame1> <vision_end>
    # <t2> <vision_start> <frame2> <vision_end>
    # Each original [T, H, W] video grid is therefore expanded into T separate
    # [1, H, W] frame grids before scanning the placeholder tokens.
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    image_token_id, video_token_id, vision_start_token_id = _get_vision_token_ids("qwen3")
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            # Use the de-padded input_ids length rather than total_input_ids[i], matching HF Qwen3 get_rope_index
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def get_rope_index_35(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Qwen3.5 MRoPE (aligned with HF upstream).

    The processor emits timestamp-separated video placeholders
    (<t1><vision_start><frame1><vision_end><t2>...), so each [T, H, W] video grid
    is expanded into T rows of [1, H, W] and every frame is handled as its own
    vision block. See transformers.models.qwen3_5.modeling_qwen3_5.get_rope_index.
    """
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    image_token_id, video_token_id, vision_start_token_id = _get_vision_token_ids("qwen3.5")
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            # Use the de-padded input_ids length rather than total_input_ids[i], matching HF Qwen3.5 get_rope_index
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def get_rope_index_25(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103, 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    image_token_id, video_token_id, vision_start_token_id = _get_vision_token_ids("qwen2.5")
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * 2

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def get_rope_index_2(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embeddin for text part.
        Examples:
            Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [3, 4, 5, 6, 7]
            text height position_ids: [3, 4, 5, 6, 7]
            text width position_ids: [3, 4, 5, 6, 7]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    image_token_id, video_token_id, vision_start_token_id = _get_vision_token_ids("qwen2")
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas


def update_qwenvl_processor_pixels(processor, data_args):
    # --- Image Processor ---
    ip = processor.image_processor
    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"ip.size: {ip.size}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    # Fast image processors (transformers >= 5.x) store limits in `ip.size`
    # as a `SizeDict` dataclass — not a `dict` subclass — but it still
    # supports both `__setitem__` and attribute access.
    if hasattr(ip, "size") and hasattr(ip.size, "__setitem__"):
        ip.size["shortest_edge"] = data_args.min_pixels
        ip.size["longest_edge"] = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor size['shortest_edge'] to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor size['longest_edge'] to {data_args.max_pixels}")

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
    rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}")
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}")
            rank0_print(f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}")

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(f"✅ Updated video_processor min_frames to {data_args.video_min_frames}")
            rank0_print(f"✅ Updated video_processor max_frames to {data_args.video_max_frames}")

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if hasattr(vp, "size") and hasattr(vp.size, "__setitem__"):
            vp.size["shortest_edge"] = data_args.video_min_pixels
            vp.size["longest_edge"] = data_args.video_max_pixels
            rank0_print(f"✅ Updated Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}")
            rank0_print(f"✅ Updated Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}")
        rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")

    return processor


def _make_abs_paths(base: str, files):
    if isinstance(files, list):
        return [os.path.realpath(os.path.join(base, f)) for f in files]
    return os.path.realpath(os.path.join(base, files))


def _uniform_subsample_indices(n: int, k: int) -> List[int]:
    # Uniformly pick k indices from [0, n-1] inclusive of endpoints.
    if k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    step = (n - 1) / (k - 1)
    return [round(i * step) for i in range(k)]


def _build_messages(
    item: Dict[str, Any],
    base_path: str | None,
    max_images: int | None = None,
    max_video_frames: int | None = None,
) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Parquet-only compatibility:
    # parquet schema may wrap video-file entries as list[list[str]], e.g. [["a.mp4"]].
    # Expand inner lists containing only video-file paths back to video entries.
    if item.get("_backend") == "parquet" and isinstance(videos, list):
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"}
        normalized_videos = []
        for v in videos:
            if isinstance(v, list) and all(isinstance(x, str) for x in v):
                if len(v) > 0 and all(Path(x).suffix.lower() in video_exts for x in v):
                    normalized_videos.extend(v)
                else:
                    normalized_videos.append(v)
            else:
                normalized_videos.append(v)
        videos = normalized_videos

    # Cap images at max_images to prevent OOM.  Uniformly subsample (keeping
    # the first AND last frames) rather than taking the head: over-cap samples
    # are temporal frame sequences whose labels ("at the moment", "last task
    # completed", "next event") depend on the final frames, so head-truncation
    # trains against states the model never sees.
    if max_images is not None and len(images) > max_images:
        print(
            f"[Warning] Sample has {len(images)} images, subsampling to {max_images}. "
            f"Conversations[0]: {item.get('conversations', [{}])[0].get('value', '')[:100]}"
        )
        images = [images[i] for i in _uniform_subsample_indices(len(images), max_images)]

    # Subsample frame-list videos before they reach the processor.
    # The Qwen video processor refuses to sample frames from list inputs
    # (do_sample_frames=False is required), so any cap must be applied here.
    # Single-file videos (str) are left to the processor's own sampler.
    if max_video_frames is not None and max_video_frames > 0:
        capped = []
        for v in videos:
            if isinstance(v, list) and len(v) > max_video_frames:
                idxs = _uniform_subsample_indices(len(v), max_video_frames)
                print(f"[Warning] Frame-list video has {len(v)} frames, subsampling to {max_video_frames}.")
                capped.append([v[i] for i in idxs])
            else:
                capped.append(v)
        videos = capped

    # Build media pools with absolute paths
    image_pool = [
        {
            "type": "image",
            "image": _make_abs_paths(base_path, img) if base_path is not None else img,
        }
        for img in images
    ]
    video_pool = [
        {
            "type": "video",
            "video": _make_abs_paths(base_path, vid) if base_path is not None else vid,
        }
        for vid in videos
    ]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        if max_images is not None:
                            continue  # skip truncated image placeholder
                        raise ValueError("Number of <image> placeholders exceeds the number of provided images")
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError("Number of <video> placeholders exceeds the number of provided videos")
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files — warn and skip instead of crashing training
    if image_pool:
        logger.warning(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders). "
            f"Item id={item.get('id', '?')}. Skipping this sample."
        )
        return None
    if video_pool:
        logger.warning(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders). "
            f"Item id={item.get('id', '?')}. Skipping this sample."
        )
        return None

    return messages


def _resize_images_in_messages(messages: List[Dict], resize_hw: Tuple[int, int]) -> List[Dict]:
    """Replace every ``{"type": "image", "image": path_or_pil}`` entry in
    ``messages`` with a resized PIL Image, leaving all other fields (``type``,
    video entries, and so on) untouched.

    Args:
        messages: The message list ``_build_messages`` returned.
        resize_hw: Target size as ``(H, W)``, matching ``ResizeImages.size``.

    Returns:
        The same ``messages`` object, modified in place.
    """
    H, W = resize_hw
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") != "image":
                continue
            img = item["image"]
            if isinstance(img, str):
                pil = _pil_to_rgb(PILImage.open(img))
            elif isinstance(img, PILImage.Image):
                pil = img
            else:
                continue
            item["image"] = pil.resize((W, H), PILImage.BILINEAR)
    return messages


def _apply_chat_template_with_processor_kwargs(processor, messages, processor_kwargs=None, **kwargs):
    """Call apply_chat_template across transformers versions.

    Transformers >=5.5 expects kwargs for ``processor.__call__`` (for example
    ``do_sample_frames``) to be passed through ``processor_kwargs`` instead of
    directly in ``**kwargs``. Older versions do not accept ``processor_kwargs``.
    """
    processor_kwargs = processor_kwargs or {}
    if not processor_kwargs:
        return processor.apply_chat_template(messages, **kwargs)

    # Try the new-style API first (transformers >=5.5 with processor_kwargs support).
    # Detect whether the processor actually supports processor_kwargs by checking
    # the apply_chat_template signature — if it doesn't list processor_kwargs
    # as an explicit parameter and accepts **kwargs, the argument will be silently
    # ignored, causing video frame-list processing to fail.
    import inspect

    sig = inspect.signature(processor.apply_chat_template)
    has_processor_kwargs_param = "processor_kwargs" in sig.parameters

    if has_processor_kwargs_param:
        return processor.apply_chat_template(messages, processor_kwargs=processor_kwargs, **kwargs)
    else:
        # Old-style: merge processor_kwargs into kwargs directly
        return processor.apply_chat_template(messages, **kwargs, **processor_kwargs)


def preprocess_qwen_visual(
    sources,
    processor,
    max_images: int | None = None,
    max_video_frames: int | None = None,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
    supervise_thinking: bool = True,
    resize_hw: Optional[Tuple[int, int]] = None,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = source.get("data_root", "")
    messages = _build_messages(source, base_path, max_images=max_images, max_video_frames=max_video_frames)
    if messages is None:
        raise ValueError("Bad sample: unused media files (image/video placeholders mismatch)")

    if resize_hw is not None:
        _resize_images_in_messages(messages, resize_hw)

    # Detect if any video is a list of image paths (frame list format),
    # which requires disabling frame sampling in the video processor.
    videos_raw = source.get("video") or []
    if isinstance(videos_raw, str):
        videos_raw = [videos_raw]
    has_frame_list_video = any(isinstance(v, list) for v in videos_raw)

    extra_kwargs = {}
    if has_frame_list_video:
        default_fps = getattr(getattr(processor, "video_processor", None), "fps", None) or 24
        extra_kwargs["do_sample_frames"] = False
        extra_kwargs["video_metadata"] = [
            {
                "total_num_frames": len(video),
                "fps": default_fps,
                "duration": len(video) / default_fps,
                "frames_indices": list(range(len(video))),
            }
            for video in videos_raw
            if isinstance(video, list)
        ]

    full_result = _apply_chat_template_with_processor_kwargs(
        processor,
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        processor_kwargs=extra_kwargs,
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    assistant_id = tokenizer.convert_tokens_to_ids("assistant")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")

    while pos < L:
        if input_ids_flat[pos] == assistant_id:
            ans_start = pos + 2  # skip "assistant" + "\n"
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != im_end_id:
                ans_end += 1
            if ans_end < L:
                # Optionally skip <think>...</think> from supervised labels
                label_start = ans_start
                if not supervise_thinking:
                    # Find </think> within [ans_start, ans_end) and start after it
                    for j in range(ans_start, ans_end):
                        if input_ids_flat[j] == think_end_id:
                            label_start = j + 1
                            break
                labels[0, label_start : ans_end + 2] = input_ids[0, label_start : ans_end + 2]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


def update_qwenvl_data_dict_for_training(data_dict, processor, model_type, merge_size=2):
    seq_len = data_dict["input_ids"][0].size(0)

    if "image_grid_thw" in data_dict:
        grid_thw = data_dict.get("image_grid_thw")
        if not isinstance(grid_thw, Sequence):
            grid_thw = [grid_thw]
    else:
        grid_thw = None

    if "video_grid_thw" in data_dict:
        video_grid_thw = data_dict.get("video_grid_thw")
        if not isinstance(video_grid_thw, Sequence):
            video_grid_thw = [video_grid_thw]
        second_per_grid_ts = [processor.video_processor.temporal_patch_size / processor.video_processor.fps] * len(
            video_grid_thw
        )
    else:
        video_grid_thw = None
        second_per_grid_ts = None

    # Select appropriate RoPE index function based on model type
    if model_type in ["qwen3vl", "qwen3vl_moe"]:
        get_rope_index = get_rope_index_3
    elif model_type in ["qwen3.5"]:
        get_rope_index = get_rope_index_35
    elif model_type in ["qwen2.5vl"]:
        get_rope_index = get_rope_index_25
    elif model_type == "qwen2vl":
        get_rope_index = get_rope_index_2
    else:
        raise ValueError(f"model_type: {model_type} not supported")

    position_ids, _ = get_rope_index(
        merge_size,
        data_dict["input_ids"],
        image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
        video_grid_thw=(torch.cat(video_grid_thw, dim=0) if video_grid_thw else None),
        second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
    )

    data_dict["position_ids"] = position_ids
    data_dict["attention_mask"] = [seq_len]
    data_dict["profile_seq_len"] = int(seq_len)

    profile_num_images = 0
    profile_num_videos = 0
    profile_vision_tokens = 0
    profile_grid_t_mean = 0.0
    profile_grid_h_mean = 0.0
    profile_grid_w_mean = 0.0
    profile_grid_t_max = 0
    profile_grid_h_max = 0
    profile_grid_w_max = 0

    if grid_thw:
        flat_image_grids = [tuple(int(v) for v in row) for g in grid_thw for row in g.tolist()]
        profile_num_images = int(len(flat_image_grids))
        profile_vision_tokens += int(
            sum(int(t) * (int(h) // merge_size) * (int(w) // merge_size) for t, h, w in flat_image_grids)
        )
        profile_grid_t_mean = sum(t for t, _, _ in flat_image_grids) / len(flat_image_grids)
        profile_grid_h_mean = sum(h for _, h, _ in flat_image_grids) / len(flat_image_grids)
        profile_grid_w_mean = sum(w for _, _, w in flat_image_grids) / len(flat_image_grids)
        profile_grid_t_max = max(t for t, _, _ in flat_image_grids)
        profile_grid_h_max = max(h for _, h, _ in flat_image_grids)
        profile_grid_w_max = max(w for _, _, w in flat_image_grids)

    if video_grid_thw:
        profile_num_videos = int(sum(int(g.shape[0]) for g in video_grid_thw))
        profile_vision_tokens += int(
            sum(
                int(t) * (int(h) // merge_size) * (int(w) // merge_size)
                for g in video_grid_thw
                for t, h, w in g.tolist()
            )
        )

    data_dict["profile_num_images"] = profile_num_images
    data_dict["profile_num_videos"] = profile_num_videos
    data_dict["profile_vision_tokens"] = profile_vision_tokens
    data_dict["profile_grid_t_mean"] = float(profile_grid_t_mean)
    data_dict["profile_grid_h_mean"] = float(profile_grid_h_mean)
    data_dict["profile_grid_w_mean"] = float(profile_grid_w_mean)
    data_dict["profile_grid_t_max"] = int(profile_grid_t_max)
    data_dict["profile_grid_h_max"] = int(profile_grid_h_max)
    data_dict["profile_grid_w_max"] = int(profile_grid_w_max)

    # text = processor.tokenizer.decode(data_dict["input_ids"][0], skip_special_tokens=False)

    # labels = data_dict["labels"][0]
    # labels = [tid if tid != -100 else processor.tokenizer.pad_token_id for tid in labels]
    # label = processor.tokenizer.decode(labels, skip_special_tokens=False)

    # return data_dict, text, label
    return data_dict, None, None


def build_vla_inference_data_dict(
    processor,
    images_pil: list,
    instruction: str,
    vlm_model_type: str = "qwen3vl",
    enable_thinking: bool = False,
    cam_view_labels: list[str] | None = None,
) -> dict:
    """Build a preprocessed data dict for VLA inference using the **training** preprocessing path.

    Uses ``preprocess_qwen_visual`` (same backend as training) to ensure that image
    token counts and sequence structure are identical between training and inference.
    Call this in servers/inference scripts instead of ``processor(text=..., images=...)``
    directly.

    Args:
        processor: The VLM processor loaded from the checkpoint.
        images_pil: Observation images as ``PIL.Image`` objects.
        instruction: Task instruction string.
        vlm_model_type: VLM architecture key (``"qwen3vl"``, ``"qwen2.5vl"``, …).
        enable_thinking: If True, the generation prompt ends with an open ``<think>\\n``
            so the model can produce chain-of-thought reasoning.  If False (default),
            the prompt contains a closed ``<think>\\n\\n</think>\\n\\n`` block.
        cam_view_labels: Pre-rendered per-image label strings, each containing exactly
            one ``<image>`` placeholder (e.g. ``["head view: <image>", ...]``).
            When None, uses bare ``<image>`` tokens (legacy behavior).

    Returns:
        dict with keys matching the training collator output:
        ``input_ids``, ``labels``, ``pixel_values``, ``image_grid_thw``,
        ``position_ids`` (3-D RoPE), ``attention_mask`` (2-D bool tensor [1, seq_len]).
    """
    if cam_view_labels is not None:
        assert len(cam_view_labels) == len(images_pil), (
            f"cam_view_labels length ({len(cam_view_labels)}) must match images_pil length ({len(images_pil)})"
        )
        image_text = "\n".join(cam_view_labels) + "\n"
    else:
        image_text = "<image>\n" * len(images_pil)

    # Observation turn only; ``add_generation_prompt`` appends the assistant
    # prefix so the prompt matches the training-time prefix exactly (the
    # Qwen3.5 chat_template adds <think>\n\n</think>\n\n to the assistant turn).
    source = [
        {
            "data_root": None,
            "image": images_pil,
            "conversations": [
                {
                    "from": "human",
                    "value": f"{image_text}{instruction}",
                },
            ],
        }
    ]

    data_dict = preprocess_qwen_visual(
        source,
        processor,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    # Compatibility guard: warn if enable_thinking was requested but the
    # chat_template (e.g. Qwen2.5-VL, Qwen3-VL) doesn't support it.
    if enable_thinking:
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        think_token_ids = tokenizer.encode("<think>", add_special_tokens=False)
        input_ids_flat = (
            data_dict["input_ids"][0].tolist()
            if hasattr(data_dict["input_ids"], "tolist")
            else data_dict["input_ids"][0]
        )
        if think_token_ids and think_token_ids[0] not in input_ids_flat:
            logger.warning(
                "enable_thinking=True was requested but the chat_template did not "
                "produce <think> tokens. This model likely does not support thinking "
                "mode (only Qwen3.5 templates do). Falling back to standard generation."
            )

    merge_size = getattr(processor.image_processor, "merge_size", 2)
    data_dict, _, _ = update_qwenvl_data_dict_for_training(
        data_dict,
        processor=processor,
        model_type=vlm_model_type,
        merge_size=merge_size,
    )

    if images_pil:
        raw_widths = [int(img.size[0]) for img in images_pil]
        raw_heights = [int(img.size[1]) for img in images_pil]
        raw_pixels = [w * h for w, h in zip(raw_widths, raw_heights)]
        data_dict["profile_raw_num_images"] = len(images_pil)
        data_dict["profile_raw_image_width_mean"] = float(sum(raw_widths) / len(raw_widths))
        data_dict["profile_raw_image_height_mean"] = float(sum(raw_heights) / len(raw_heights))
        data_dict["profile_raw_image_width_max"] = int(max(raw_widths))
        data_dict["profile_raw_image_height_max"] = int(max(raw_heights))
        data_dict["profile_raw_image_pixels_mean"] = float(sum(raw_pixels) / len(raw_pixels))
        data_dict["profile_raw_image_pixels_max"] = int(max(raw_pixels))
    else:
        data_dict["profile_raw_num_images"] = 0
        data_dict["profile_raw_image_width_mean"] = 0.0
        data_dict["profile_raw_image_height_mean"] = 0.0
        data_dict["profile_raw_image_width_max"] = 0
        data_dict["profile_raw_image_height_max"] = 0
        data_dict["profile_raw_image_pixels_mean"] = 0.0
        data_dict["profile_raw_image_pixels_max"] = 0

    # ``update_qwenvl_data_dict_for_training`` converts attention_mask to the packed
    # format ``[seq_len]`` (a list with a single integer).  For inference we need a
    # standard 2-D boolean tensor ``[1, seq_len]`` that the unpatched Qwen attention
    # can process, and that ``_build_dit_inputs`` in MiBoT can index with ``[:, None, :]``.
    seq_len = data_dict["input_ids"].shape[1]
    data_dict["attention_mask"] = torch.ones(1, seq_len, dtype=torch.bool)

    return data_dict
