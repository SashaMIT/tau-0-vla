"""Batch collation for Qwen-VL supervised training.

``DataCollatorForVLMSupervisedDataset`` pads the per-sample Qwen-VL encodings
produced by ``qwenvl_utils.build_vla_inference_data_dict`` into a batch: token
ids and labels are right-padded, the 3-D MRoPE ``position_ids`` are padded and
concatenated, and the variable-length vision tensors (``pixel_values`` /
``pixel_values_videos`` and their grids) are concatenated along the patch axis.

``dataset.DataCollatorForVLASupervisedDataset`` subclasses it to stack the
action-expert tensors (state / action / masks) on top.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers

from tau0_vla.configs.constants import IGNORE_INDEX


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 0)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForVLMSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    _padding_side_checked: bool = False

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]

        # --- Oversized-sample diagnostics -------------------------------------
        # The micro-batch is padded to the LONGEST sample in it (VLA bs=28),
        # so one long sample multiplies activation memory for the whole batch
        # and OOMs the unlucky rank -- which then stalls every other rank at the
        # next allreduce until the NCCL watchdog (600s) SIGABRTs the whole job,
        # looking deceptively like a cluster-wide "node OOM". Log the offender's
        # length / source so its distribution can be evaluated for a filter.
        # Threshold defaults to 75% of model_max_length; override via env.
        _long_thr = int(os.environ.get("VLA_LOG_LONG_SAMPLE_LEN", str(int(0.75 * self.tokenizer.model_max_length))))
        _lens = [int(ids.shape[0]) for ids in input_ids]
        if _lens and max(_lens) >= _long_thr:
            _argmax = max(range(len(_lens)), key=lambda j: _lens[j])
            _n_trunc = sum(1 for _L in _lens if _L > self.tokenizer.model_max_length)
            _peak = instances[_argmax]

            # Split the peak length into vision vs text so a future long sample
            # self-diagnoses: huge image_grid (uncapped images) vs huge text
            # (corrupted/over-long instruction). Vision patches come straight
            # from the grids; text ~= max_len - vision_tokens (post-merge).
            def _vision_patches(inst):
                n = 0
                for _k in ("image_grid_thw", "video_grid_thw"):
                    g = inst.get(_k)
                    if g is None:
                        continue
                    try:
                        for _t, _h, _w in g.reshape(-1, 3).tolist():
                            n += int(_t) * int(_h) * int(_w)
                    except Exception:
                        pass
                return n

            _merge = int(getattr(self.tokenizer, "_long_sample_merge", 2)) ** 2 or 1
            _vis_patches = _vision_patches(_peak)
            _vis_tokens = _vis_patches // _merge
            _n_img = _peak.get("profile_raw_num_images", _peak.get("profile_num_images"))
            print(
                f"[LONG_SAMPLE] rank={os.environ.get('RANK', '?')} "
                f"max_len={max(_lens)} model_max_length={self.tokenizer.model_max_length} "
                f"would_truncate={_n_trunc}/{len(_lens)} "
                f"peak_num_images={_n_img} peak_vision_tokens~={_vis_tokens} "
                f"peak_text_tokens~={max(_lens) - _vis_tokens} "
                f"source={str(_peak.get('profile_source'))[:200]} "
                f"all_lens={sorted(_lens, reverse=True)}",
                flush=True,
            )
        # ----------------------------------------------------------------------

        # torch.nn.utils.rnn.pad_sequence defaults to padding_side='right', which
        # Tau0VLA / HF FA2 / joint-attention prefix masks all rely on. The one-shot
        # check below verifies this contract on the first batch only.
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        if not self._padding_side_checked:
            mask = batch["attention_mask"]
            if mask.shape[1] > 1:
                has_valid_after_pad = ((~mask[:, :-1]) & mask[:, 1:]).any().item()
                if has_valid_after_pad:
                    raise ValueError(
                        "Collator produced non-right-padded attention_mask. Tau0VLA requires "
                        "right-padding (HF FA2 cu_seqlens, joint-attention prefix masks, RoPE "
                        "position_ids all assume it). Check the dataset's tokenize step."
                    )
            self._padding_side_checked = True
        images = list(instance["pixel_values"] for instance in instances if "pixel_values" in instance)
        videos = list(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids

        batch["profile_batch_seq_len"] = int(batch["input_ids"].shape[1])
        batch["profile_batch_num_images"] = sum(int(instance.get("profile_num_images", 0)) for instance in instances)
        batch["profile_batch_num_videos"] = sum(int(instance.get("profile_num_videos", 0)) for instance in instances)
        batch["profile_batch_vision_tokens"] = sum(
            int(instance.get("profile_vision_tokens", 0)) for instance in instances
        )
        batch["profile_batch_raw_num_images"] = sum(
            int(instance.get("profile_raw_num_images", 0)) for instance in instances
        )
        batch["profile_batch_raw_image_width_mean"] = sum(
            float(instance.get("profile_raw_image_width_mean", 0.0)) for instance in instances
        )
        batch["profile_batch_raw_image_height_mean"] = sum(
            float(instance.get("profile_raw_image_height_mean", 0.0)) for instance in instances
        )
        batch["profile_batch_raw_image_width_max"] = max(
            (int(instance.get("profile_raw_image_width_max", 0)) for instance in instances),
            default=0,
        )
        batch["profile_batch_raw_image_height_max"] = max(
            (int(instance.get("profile_raw_image_height_max", 0)) for instance in instances),
            default=0,
        )
        batch["profile_batch_raw_image_pixels_mean"] = sum(
            float(instance.get("profile_raw_image_pixels_mean", 0.0)) for instance in instances
        )
        batch["profile_batch_raw_image_pixels_max"] = max(
            (int(instance.get("profile_raw_image_pixels_max", 0)) for instance in instances),
            default=0,
        )
        return batch


@dataclass
class FlattenedDataCollatorForVLMSupervisedDataset(DataCollatorForVLMSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(*(instance["attention_mask"] for instance in instances if "attention_mask" in instance))
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(instance["pixel_values"] for instance in instances if "pixel_values" in instance)
        videos = list(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        return batch
