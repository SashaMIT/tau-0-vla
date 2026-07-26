"""Readable view over a canonical-form observation dict.

`Payload` is the single place that knows how to pull the raw state, canonical
instruction, and per-camera preprocessed images out of a canonical payload.
Policy-repo inference code consumes this directly — it never has to branch on
``observation.state`` vs ``state``, vs the two formats of ``images``, vs
whether the prompt was pre-canonicalized upstream.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

import numpy as np

from tau0_vla.data.modalities.image.base import Image
from tau0_vla.data.modalities.language import Prompt


@dataclasses.dataclass(frozen=True)
class Payload:
    """Encapsulated reader over a canonical payload dict.

    Args:
        raw: the raw payload dict as produced by `robot_config.repack_observation`
            (openloop) or passed directly by the caller (teleop).
        image_specs: per-camera `Image` specs from `robot_config.resolve_images`.
            Drives ordering and pixel preprocessing.
        prompt_spec: the `Prompt` spec from `robot_config.resolve_prompt`.

    Accepted state keys (tried in order):
        - ``observation.state`` (canonical raw-flat form)
        - ``state`` (only if already an ``np.ndarray``)

    Accepted image containers:
        - ``{cam_name: array}`` — ordered by ``image_specs``
        - ``[array, ...]`` — positional, zipped with ``image_specs``
    """

    raw: Mapping[str, Any]
    image_specs: tuple[Image, ...]
    prompt_spec: Prompt

    # ---- state --------------------------------------------------------------

    def state_raw(self) -> np.ndarray:
        """Flat float32 raw state. Shape ``(state_dim_raw,)``."""
        flat = self.raw.get("observation.state")
        if flat is None:
            maybe = self.raw.get("state")
            if isinstance(maybe, np.ndarray):
                flat = maybe
        if flat is None:
            raise ValueError(
                "payload must provide raw state: either 'observation.state' or "
                "'state' (as np.ndarray). Got keys: "
                f"{sorted(self.raw)}"
            )
        return np.asarray(flat, dtype=np.float32).flatten()

    # ---- prompt -------------------------------------------------------------

    def instruction(self) -> str:
        """Canonical instruction text via `Prompt.format()`.

        If the upstream pipeline already applied the template's prefix (e.g.
        `repack_observation` canonicalized before handoff), passes the raw
        string through to avoid double-prefixing.
        """
        raw = self.raw.get("prompt") or ""
        prefix = _template_prefix(self.prompt_spec.template)
        if prefix and raw.startswith(prefix):
            return raw
        return self.prompt_spec.format(raw)

    # ---- images -------------------------------------------------------------

    def images(self, *, train: bool = False) -> list[np.ndarray]:
        """Per-camera images after each spec's `apply(train=train)` pipeline.

        Returned in the order of ``image_specs``.
        """
        raw_images = self.raw.get("images") or []
        ordered = self._order(raw_images)
        return [spec.apply(np.asarray(img), train=train) for spec, img in zip(self.image_specs, ordered, strict=False)]

    def _order(self, raw_images: Any) -> list[Any]:
        from tau0_vla.data.modalities.image import SyntheticImage

        if isinstance(raw_images, Mapping):
            keyed = []
            for spec in self.image_specs:
                if spec.name in raw_images:
                    keyed.append(raw_images[spec.name])
                elif isinstance(spec, SyntheticImage):
                    keyed.append(spec.materialize())
            return keyed if keyed else list(raw_images.values())
        return list(raw_images)


def _template_prefix(template: str) -> str:
    """Return the literal prefix of a prompt template (everything before the
    first ``{`` placeholder). Used to detect already-canonicalized prompts.
    """
    idx = template.find("{")
    return template[:idx] if idx >= 0 else template


__all__ = ["Payload"]
