"""Visual modality subpackage.

- ``Image`` class for per-camera declarative specs (`base.py`).
- Pixel-domain transforms: deterministic (``ResizeWithPad`` / ``Identity``) and
  random (``ColorJitter``) — `transforms.py`.
- Numpy reference implementations of image ops — `resize_ops.py`.
"""

from tau0_vla.data.modalities.image.base import Image, SyntheticImage
from tau0_vla.data.modalities.image.resize_ops import resize_with_pad_numpy
from tau0_vla.data.modalities.image.transforms import (
    ColorJitter,
    Identity,
    ImageTransform,
    ResizeWithPad,
)

__all__ = [
    "ColorJitter",
    "Identity",
    "Image",
    "ImageTransform",
    "ResizeWithPad",
    "SyntheticImage",
    "resize_with_pad_numpy",
]
