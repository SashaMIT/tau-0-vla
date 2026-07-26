import copy
from pathlib import Path
from typing import Dict, List, Optional

import torch
import transformers
from torch import nn
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from tau0_vla.vlm.qwenvl_utils import update_qwenvl_processor_pixels


class QwenVLInterfacewrapper(nn.Module):
    """
    Interface wrapper of Qwen-VL models to unify the API for Agibot-VLA.
    Support Qwen2VL, Qwen2.5VL and Qwen3VL (Dense and MoE).
    """

    def __init__(self, model_args, training_args, data_args):
        super().__init__()
        self.model_args = model_args
        self.training_args = training_args
        self.data_args = data_args

        self.model = self.init_model(attn_implementation="flash_attention_2")
        self.processor = self.init_tokenizer_and_processor()

    def init_model(self, attn_implementation="flash_attention_2"):
        if self.model_args.vlm_model_type == "qwen3vl_moe":
            # Qwen3-VL-MoE
            model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                self.model_args.model_name_or_path,
                cache_dir=self.training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=(torch.bfloat16 if self.training_args.bf16 else None),
            )

        elif self.model_args.vlm_model_type == "qwen3vl":
            # Qwen3-VL-Dense
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_args.model_name_or_path,
                cache_dir=self.training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=(torch.bfloat16 if self.training_args.bf16 else None),
            )
        elif self.model_args.vlm_model_type == "qwen2.5vl":
            # Qwen2.5-VL
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_args.model_name_or_path,
                cache_dir=self.training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=(torch.bfloat16 if self.training_args.bf16 else None),
            )
        elif self.model_args.vlm_model_type == "qwen2vl":
            # Qwen2-VL
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_args.model_name_or_path,
                cache_dir=self.training_args.cache_dir,
                attn_implementation=attn_implementation,
                dtype=(torch.bfloat16 if self.training_args.bf16 else None),
            )
        else:
            raise NotImplementedError("Only support qwen models for now.")
        model.config.use_cache = False

        return model

    def init_tokenizer_and_processor(self):
        processor = AutoProcessor.from_pretrained(
            self.model_args.model_name_or_path,
        )

        # update qwenvl processor
        if self.model_args.vlm_model_type in ["qwen3vl", "qwen2.5vl", "qwen2vl", "qwen3vl_moe"]:
            processor = update_qwenvl_processor_pixels(processor, self.data_args)

        # tokenizer = transformers.AutoTokenizer.from_pretrained(
        #     self.model_args.model_name_or_path,
        #     cache_dir=self.training_args.cache_dir,
        #     model_max_length=self.training_args.model_max_length,
        #     padding_side="right",
        #     use_fast=False,
        # )

        # self.tokenizer = tokenizer
        return processor

    def forward(self, **kwargs) -> CausalLMOutputWithPast:
        return self.model(**kwargs)

    def generate(
        self,
        **kwargs,
    ):
        generation_output = self.model.generate(**kwargs)
        return generation_output
