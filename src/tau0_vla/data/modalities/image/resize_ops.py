"""Numpy/PIL implementations of image ops that `Image` transforms use at
inference time.

Framework-specific accelerated versions (JAX-jitted, torch.nn.functional) live
inside the policy repos and are used for training throughput; this module
provides the canonical reference semantics that every framework must agree on.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def resize_with_pad_numpy(
    image: np.ndarray,
    height: int,
    width: int,
    *,
    fill_value: Any = None,
) -> np.ndarray:
    """Replicate `tf.image.resize_with_pad` semantics in pure numpy/PIL.

    Resize preserving aspect ratio; pad shorter side with `fill_value` to hit
    the target (height, width). Matches openpi's `resize_with_pad` on dtype
    conventions:

    - ``uint8`` input → values in [0, 255], pad with 0
    - ``float32`` input → values in [-1, 1], pad with -1.0

    Args:
        image: ``(H, W, C)`` (single image) or ``(T, H, W, C)`` (batch/time).
        height: target height.
        width: target width.
        fill_value: override padding value; inferred from dtype if None.
    """

    arr = np.asarray(image)
    if arr.ndim == 3:
        return _resize_one(arr, height, width, fill_value=fill_value)
    if arr.ndim == 4:
        return np.stack(
            [_resize_one(arr[i], height, width, fill_value=fill_value) for i in range(arr.shape[0])],
            axis=0,
        )
    raise ValueError(f"resize_with_pad_numpy: expected ndim in (3, 4), got {arr.ndim}")


def _resize_one(image: np.ndarray, height: int, width: int, *, fill_value: Any) -> np.ndarray:
    from PIL import Image as _PILImage

    if image.dtype == np.uint8:
        pad = 0 if fill_value is None else int(fill_value)
        data = image
    elif image.dtype == np.float32:
        pad = -1.0 if fill_value is None else float(fill_value)
        data = ((image + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
    else:
        raise ValueError(f"resize_with_pad_numpy: unsupported dtype {image.dtype}")

    # PIL's ``fromarray`` rejects ``(H, W, 1)`` uint8 — only ``(H, W)`` (mode "L")
    # or ``(H, W, 3)``. Squeeze single-channel 3D arrays for the PIL hop, then
    # restore the trailing channel axis so the caller still sees HWC layout.
    single_channel = data.ndim == 3 and data.shape[-1] == 1
    pil_input = data[..., 0] if single_channel else data

    cur_h, cur_w = data.shape[:2]
    ratio = max(cur_w / width, cur_h / height)
    resized_h = max(1, int(cur_h / ratio))
    resized_w = max(1, int(cur_w / ratio))

    pil = _PILImage.fromarray(pil_input).resize((resized_w, resized_h), resample=_PILImage.BILINEAR)
    resized = np.asarray(pil)
    if single_channel:
        resized = resized[..., None]

    pad_h0, rem_h = divmod(height - resized_h, 2)
    pad_h1 = pad_h0 + rem_h
    pad_w0, rem_w = divmod(width - resized_w, 2)
    pad_w1 = pad_w0 + rem_w

    if image.dtype == np.uint8:
        out = np.pad(
            resized,
            ((pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
            mode="constant",
            constant_values=pad,
        )
        return out
    # float32 round-trip
    out_float = resized.astype(np.float32) / 255.0 * 2.0 - 1.0
    return np.pad(
        out_float,
        ((pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        mode="constant",
        constant_values=pad,
    )


__all__ = ["resize_with_pad_numpy"]
