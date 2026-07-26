"""Pixel-domain transforms for `Image` modalities.

Convention: any class whose name starts with ``Random`` is a training-time
augmentation; at inference it is replaced by the result of ``eval_fallback()``
(a deterministic transform, typically ``ResizeWithPad`` or ``Identity``).
Deterministic transforms (``ResizeWithPad``, ``Identity``) behave the same in
both train and eval.

These classes carry declarative spec only. The numeric ops live in
`tau0_vla.data.modalities.image.resize_ops` (numpy/PIL reference
implementation). Framework-specific accelerated implementations (JAX-jitted,
torch) live inside each policy repo's training pipeline and read the same specs.
"""

from __future__ import annotations

import dataclasses
import random

import numpy as np

from tau0_vla.data.modalities.image.resize_ops import resize_with_pad_numpy


class ImageTransform:
    """Base class for pixel-domain transforms.

    Subclasses implement ``apply(image, train=bool)`` — returns a new image.
    ``Random*`` subclasses additionally implement ``eval_fallback()`` — return
    the deterministic transform applied at inference.
    """

    def apply(self, image: np.ndarray, *, train: bool = True) -> np.ndarray:
        raise NotImplementedError

    @property
    def is_random(self) -> bool:
        return type(self).__name__.startswith("Random") or type(self).__name__ == "ColorJitter"

    def eval_fallback(self) -> "ImageTransform":
        """Deterministic counterpart used at inference. Default: self."""
        return self


# ---------------------------------------------------------------------------
# Deterministic transforms
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Identity(ImageTransform):
    """No-op. Used as a default eval_fallback for random transforms that have
    no deterministic equivalent (e.g. `ColorJitter`)."""

    def apply(self, image: np.ndarray, *, train: bool = True) -> np.ndarray:
        return image

    @property
    def is_random(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class ResizeWithPad(ImageTransform):
    """Resize to (height, width) preserving aspect ratio; pad with black.

    Mirrors the semantics of openpi's `image_tools.resize_with_pad`:
    - uint8 → pad 0
    - float32 → pad -1.0 (values in [-1, 1])
    """

    height: int
    width: int

    def apply(self, image: np.ndarray, *, train: bool = True) -> np.ndarray:
        return resize_with_pad_numpy(image, self.height, self.width)

    @property
    def is_random(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Random transforms (train-only; replaced by eval_fallback at inference)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ColorJitter(ImageTransform):
    """Training-time brightness / contrast / saturation jitter.
    Inference fallback: `Identity` (no color modification).

    Args mirror torchvision: each magnitude is a max delta applied as
    ``1 + U(-v, +v)`` multiplicative. ``prob`` adds an independent Bernoulli
    gate per frame — default 1.0 means "always apply" (every frame gets a small
    random perturbation); set e.g. 0.33 to match baseline
    ``ColorJitterImages(prob=0.33)`` where a third of frames get full-strength
    jitter and the rest pass through untouched.

    ``hue`` is accepted for signature compatibility with torchvision but is
    **not applied** — ``PIL.ImageEnhance`` has no Hue operation. This matches
    the pre-migration baseline exactly (verified byte-identical), so it is
    upstream behaviour, not a migration artifact. The torchvision-backed
    ``ColorJitterImages`` in ``tau0_vla.vlm.policy_transforms`` does apply
    hue; see its docstring before switching, and note that enabling both jitters
    each frame twice.
    """

    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0
    prob: float = 1.0

    def apply(self, image: np.ndarray, *, train: bool = True) -> np.ndarray:
        if not train:
            return self.eval_fallback().apply(image, train=False)
        if self.prob < 1.0 and random.random() >= self.prob:
            return np.asarray(image)
        arr = np.asarray(image)
        from PIL import Image as _PILImage
        from PIL import ImageEnhance

        was_float = arr.dtype != np.uint8
        data = arr if not was_float else ((arr + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        img = _PILImage.fromarray(data).convert("RGB")
        # Draw order is load-bearing: each enhance() consumes one value from the
        # shared `random` stream, so reordering these silently changes every
        # augmented image for a given seed.
        if self.brightness > 0:
            img = ImageEnhance.Brightness(img).enhance(1.0 + random.uniform(-self.brightness, self.brightness))
        if self.contrast > 0:
            img = ImageEnhance.Contrast(img).enhance(1.0 + random.uniform(-self.contrast, self.contrast))
        if self.saturation > 0:
            img = ImageEnhance.Color(img).enhance(1.0 + random.uniform(-self.saturation, self.saturation))
        out = np.asarray(img)
        return out if not was_float else (out.astype(np.float32) / 255.0 * 2.0 - 1.0)

    def eval_fallback(self) -> Identity:
        return Identity()


__all__ = [
    "ColorJitter",
    "Identity",
    "ImageTransform",
    "ResizeWithPad",
]
