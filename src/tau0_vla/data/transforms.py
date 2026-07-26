"""Data-pipeline transform framework.

Two halves:

1. **Framework** — `DataTransformFn` Protocol + `compose` + `CompositeTransform`.
   The minimal contract: a transform is a callable ``data -> data`` where
   ``data`` is a ``Dict[str, Any]`` sample. Transforms chain via ``compose``.

2. **Registry** — ``@register_transform("Name")`` decorator + ``build_transforms``
   factory. Policy repos decorate their custom transform classes; data-configs
   then reference them by name:

   .. code:: python

        # in a policy repo
        from tau0_vla.data import register_transform

        @register_transform("ColorJitterImages")
        @dataclasses.dataclass(frozen=True)
        class ColorJitterImages: ...

        # in a data_config
        transforms = [
            {"type": "ColorJitterImages", "prob": 0.3},
            {"type": "GeneratePrompt", "default_prompt": [...]},
        ]

        # when wiring up a dataset
        from tau0_vla.data import build_transforms, compose
        pipeline = compose(build_transforms(config.transforms))

Nothing here assumes torch / torchvision / PIL — those come in via concrete
transforms registered by whichever policy repo needs them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Dict, Protocol, TypeAlias, runtime_checkable

DataDict: TypeAlias = Dict[str, Any]


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict: ...


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    transforms: tuple[DataTransformFn, ...]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Chain transforms into a single callable."""
    return CompositeTransform(tuple(transforms))


# ---------------------------------------------------------------------------
# Registry: lets data-configs reference transforms by name
# ---------------------------------------------------------------------------


_TRANSFORM_REGISTRY: dict[str, type] = {}


def register_transform(name: str):
    """Decorator: register a transform class under ``name`` so
    ``build_transforms`` can instantiate it from a data-config spec.

    Raises ``ValueError`` on duplicate registration from a different class.
    Re-registration by the same class (reloads, etc) is a no-op.
    """

    def _decorator(cls):
        existing = _TRANSFORM_REGISTRY.get(name)
        if existing is not None and existing is not cls:
            raise ValueError(f"Transform '{name}' already registered by {existing.__module__}.{existing.__name__}")
        _TRANSFORM_REGISTRY[name] = cls
        return cls

    return _decorator


def get_transform(name: str) -> type:
    """Look up a registered transform class by name."""
    try:
        return _TRANSFORM_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(_TRANSFORM_REGISTRY)) or "<empty>"
        raise KeyError(
            f"Unknown transform '{name}'. Registered: {available}. "
            f"Make sure the module defining this transform was imported "
            f"so its @register_transform decorator ran."
        ) from exc


def list_registered_transforms() -> list[str]:
    return sorted(_TRANSFORM_REGISTRY)


def build_transforms(specs: Sequence[dict | str]) -> list[DataTransformFn]:
    """Instantiate transforms from a data-config specs list.

    Each spec is one of:

    - A bare string ``"TypeName"`` — instantiates the registered class with no
      kwargs.
    - A dict ``{"type": "TypeName", **kwargs}`` — pops ``type`` and passes the
      remainder as kwargs to the class constructor.

    Returns a list of instantiated transforms (in input order). Wrap with
    ``compose`` if you want a single callable.
    """
    built: list[DataTransformFn] = []
    for spec in specs:
        if isinstance(spec, str):
            built.append(get_transform(spec)())
            continue
        cfg = dict(spec)
        type_name = cfg.pop("type", None)
        if type_name is None:
            raise ValueError(f"transform spec is missing 'type' key: {spec}")
        built.append(get_transform(type_name)(**cfg))
    return built


__all__ = [
    "CompositeTransform",
    "DataDict",
    "DataTransformFn",
    "build_transforms",
    "compose",
    "get_transform",
    "list_registered_transforms",
    "register_transform",
]
