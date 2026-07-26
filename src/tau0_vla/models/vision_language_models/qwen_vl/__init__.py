"""Qwen-VL loading and attention patches.

``qwen_vl.QwenVLInterfacewrapper`` loads a backbone across Qwen3-VL / Qwen2.5-VL
/ Qwen2-VL; ``qwen_vl_patches`` holds the variable-length attention replacement,
which the shipped configs deliberately do not enable.
"""

__all__: list[str] = []
