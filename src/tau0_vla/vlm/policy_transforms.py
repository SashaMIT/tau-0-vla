"""Policy-side ``@register_transform`` entries for the finch pipeline.

These are tau-0-vla-internal transforms that piggyback on the data
pipeline's ``build_transforms`` registry. They MUST NOT be moved into
``tau0_vla.data``: that would add a VLM-only concern to the general
pipeline, which the adapter boundary deliberately keeps out of it.
They encode VLM-side conventions the pipeline has no business knowing about.

Importing this module is sufficient to register the transforms.
``FinchVLADataset.__init__`` does that import lazily — the registrations
fire on first import, and subsequent yaml-driven
``tau0_vla.data.build_transforms(specs)`` calls resolve ``{"type": "Name"}``
entries against the populated registry.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Any

import numpy as np

from tau0_vla.data import DataDict, DataTransformFn, register_transform

# ─────────────────────────────────────────────────────────────────────────────
# Image color jitter (VLM-side augmentation)
# ─────────────────────────────────────────────────────────────────────────────


@register_transform("ColorJitterImages")
@dataclasses.dataclass(frozen=True)
class ColorJitterImages(DataTransformFn):
    """Per-camera brightness / contrast / saturation / hue jitter.

    **Legacy — no shipped config enables this.** Prefer declaring
    ``ColorJitter`` on ``RobotConfig.images``
    (``tau0_vla.data.modalities.image``): that path is per-camera,
    deploy-symmetric, and runs inside the pipeline where the deploy side
    resolves the same spec. This one runs *after* the pipeline, so enabling
    both jitters every frame twice. It is kept because the ``@register_transform``
    registry is the documented extension point for policy-side transforms
    and this is its worked example.

    One behavioural difference if you do switch: this class uses
    ``torchvision.transforms.ColorJitter``, which applies ``hue``. The
    ``RobotConfig.images`` version is PIL-based and silently ignores ``hue``
    (``PIL.ImageEnhance`` has no Hue operation) — true of the pre-migration
    baseline too, so it is upstream behaviour rather than a migration artifact.

    Operates on the finch payload's ``raw["images"]`` dict (camera name →
    uint8 HWC ``np.ndarray``). Each image is independently Bernoulli-gated
    by ``prob``; gated-on images get a fresh random draw from torchvision's
    ``ColorJitter`` and are written back in place. Non-dict / non-HWC
    entries are left untouched.

    Defaults mirror the baseline ``ColorJitterImages(prob=0.3, brightness=0.3,
    contrast=0.4, saturation=0.5, hue=0.03)`` so yaml specs porting from the
    old repo need only pass overrides (e.g. ``prob: 0.33``).
    """

    prob: float = 0.3
    brightness: float = 0.3
    contrast: float = 0.4
    saturation: float = 0.5
    hue: float = 0.03
    images_key: str = "images"

    def __post_init__(self) -> None:
        import torchvision.transforms as tv_transforms

        object.__setattr__(
            self,
            "_jitter",
            tv_transforms.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
            ),
        )

    def __call__(self, data: DataDict) -> DataDict:
        images = data.get(self.images_key)
        if not isinstance(images, dict):
            return data
        from PIL import Image as _PILImage

        updated = dict(images)
        any_changed = False
        for k, v in images.items():
            arr = np.asarray(v)
            if arr.ndim != 3 or arr.shape[-1] not in (1, 3):
                continue
            if random.random() >= self.prob:
                continue
            src = arr if arr.dtype == np.uint8 else np.clip(arr, 0, 255).astype(np.uint8)
            pil = _PILImage.fromarray(src)
            jittered = np.asarray(self._jitter(pil))
            updated[k] = jittered if arr.dtype == np.uint8 else jittered.astype(arr.dtype, copy=False)
            any_changed = True
        if any_changed:
            data[self.images_key] = updated
        return data


__all__ = ["ColorJitterImages"]
