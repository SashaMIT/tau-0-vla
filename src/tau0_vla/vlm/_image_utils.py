"""Numpy-only image normalizers shared between the finch training dataset
and the finch deploy policy. Kept here (not in ``utils/``) so deploy-only
install footprints don't drag in torch/yaml/``tau0_vla.data`` imports."""

from __future__ import annotations

import numpy as np


def to_uint8_hwc(img: np.ndarray) -> np.ndarray:
    """Coerce an array to uint8 HWC so PIL.Image.fromarray doesn't silently
    clip float pixel data or swap channel ordering.

    The data pipeline's payload contract is uint8 HWC, but upstream transforms
    occasionally return ``(C, H, W)`` or float32 ``[0, 1]`` variants — this
    helper makes the downstream code robust to both.

    NOT interchangeable with ``tau0_vla.data.pipeline.canonicalize_image_hwc``,
    despite the similar name. That one is the pipeline's *internal* normalizer:
    its inputs come from LeRobot in known formats, so it is strict (raises on an
    unexpected layout) and preserves single-channel images. This one is a
    *defensive adapter* for ``PIL.Image.fromarray``, whose failure mode is
    silent corruption rather than an exception, so it is deliberately lenient:
    it transposes CHW ndarrays, treats float data with ``max > 1`` as already
    scaled to ``[0, 255]``, and broadcasts grayscale to 3 channels. Merging the
    two would force one contract onto both call sites and lose whichever
    property the other depends on.
    """
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if float(arr.max()) <= 1.0 + 1e-3:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


__all__ = ["to_uint8_hwc"]
