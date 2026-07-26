"""The `Image` modality — per-camera declarative spec.

An ``Image("head", transforms=[ResizeWithPad(224, 224), ColorJitter(0.2)])``
instance declares:

- the canonical-payload key for this camera (``"head"``),
- the pixel-domain transform chain to apply.

Transforms are executed via `Image.apply(raw_image, train=bool)`. At inference
(``train=False``), any ``Random*`` transform is auto-replaced by its
``eval_fallback()`` — deterministic.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

import numpy as np

from tau0_vla.data.modalities.image.transforms import ImageTransform


@dataclasses.dataclass(frozen=True)
class Image:
    """Per-camera declaration.

    Args:
        name: canonical-payload key for this camera ("head", "wrist_left", ...).
        transforms: pixel-domain transform chain (`ResizeWithPad` / random
            augmentations / etc.). Empty list means "pass-through".
    """

    name: str
    transforms: tuple[ImageTransform, ...] = ()

    def __init__(
        self,
        name: str,
        *,
        transforms: Iterable[ImageTransform] | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "transforms", tuple(transforms or ()))

    def apply(self, image: np.ndarray, *, train: bool = True) -> np.ndarray:
        """Run the transform chain. At inference, ``Random*`` transforms are
        swapped for their ``eval_fallback()`` (typically a deterministic
        `ResizeWithPad` or `Identity`).
        """
        out = np.asarray(image)
        for t in self.transforms:
            step = t if (train or not t.is_random) else t.eval_fallback()
            out = step.apply(out, train=train)
        return out

    @property
    def output_size(self) -> tuple[int, int] | None:
        """Infer final (height, width) from the transform chain.

        Looks for the last transform that declares `height`/`width` attributes
        (e.g. `ResizeWithPad`). Returns None if nothing in the chain sets a
        fixed size — in that case the image is passed through at its native
        resolution.
        """
        size: tuple[int, int] | None = None
        for t in self.transforms:
            h = getattr(t, "height", None)
            w = getattr(t, "width", None)
            if h is not None and w is not None:
                size = (int(h), int(w))
        return size


@dataclasses.dataclass(frozen=True)
class SyntheticImage(Image):
    """Declarative synthetic camera spec.

    ``SyntheticImage("head", "black", transforms=[ResizeWithPad(224, 224)])``
    means the model contract includes a ``head`` camera, but the payload should
    be synthesized as a black image instead of read from a real sensor.

    ``mode`` currently only supports ``"black"``. Synthetic images are
    materialized directly at their final output resolution and do not apply
    random color transforms; this keeps the placeholder semantically "missing"
    instead of becoming augmented noise.
    """

    mode: str = "black"

    def __init__(
        self,
        name: str,
        mode: str = "black",
        *,
        transforms: Iterable[ImageTransform] | None = None,
    ) -> None:
        super().__init__(name, transforms=transforms)
        object.__setattr__(self, "mode", str(mode))

    def apply(self, image: np.ndarray | None = None, *, train: bool = True) -> np.ndarray:
        del image, train
        return self.materialize()

    def materialize(self, *, target_size: tuple[int, int] | None = None) -> np.ndarray:
        if self.mode != "black":
            raise ValueError(f"Unsupported SyntheticImage mode {self.mode!r}; expected 'black'")
        size = target_size or self.output_size
        if size is None:
            raise ValueError(
                f"SyntheticImage(name={self.name!r}, mode='black') requires a fixed output size. "
                "Declare a size-setting transform such as ResizeWithPad."
            )
        height, width = int(size[0]), int(size[1])
        return np.zeros((height, width, 3), dtype=np.uint8)

    @classmethod
    def black(
        cls,
        name: str,
        *,
        transforms: Iterable[ImageTransform] | None = None,
    ) -> "SyntheticImage":
        return cls(name, "black", transforms=transforms)


__all__ = ["Image", "SyntheticImage"]
