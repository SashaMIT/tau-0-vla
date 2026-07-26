"""Component + transform (de)serialization for checkpoint-self-sufficient deploy.

At train time, a checkpoint saves the exact state/action ComponentSpec lists
(ArmJoint / Gripper / RelativeToState / ...) that the data pipeline used.
At deploy time, the lists are reconstructed from that JSON alone — without
calling any ``@register_config`` function. Deploy's only source dependency
is that the component / transform classes themselves (``ArmJoint``,
``RelativeToState``, ...) remain importable from this package.

The serialized shape is::

    {"_type": "ArmJoint", "normalize": "mean_std", "transforms": [
        {"_type": "RelativeToState", "mask": null}
    ]}

class-name-addressed via ``getattr(tau0_vla.data.modalities, name)``. Adding a
new ComponentSpec / ComponentTransform subclass is therefore zero extra wiring:
the class only has to live under ``tau0_vla.data.modalities`` (and accept its
own init kwargs via ``__init__``). Non-JSON-safe attributes (e.g. a NumPy
mask) are listed-ified on the way out and rebuilt by the class's ``__init__``
on the way in.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from tau0_vla.data.modalities.base import ComponentSpec, ComponentTransform

_COMPONENTS_MODULE = "tau0_vla.data.modalities"
_TRANSFORMS_MODULE = "tau0_vla.data.modalities.transforms"


def component_to_dict(comp: ComponentSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {"_type": type(comp).__name__}
    if comp.key != type(comp).component_key:
        payload["key"] = comp.key
    payload["normalize"] = comp.normalize
    if comp.field_key is not None:
        payload["field_key"] = comp.field_key
    if comp.transforms:
        payload["transforms"] = [transform_to_dict(t) for t in comp.transforms]
    return payload


def component_from_dict(payload: dict[str, Any]) -> ComponentSpec:
    cls = _resolve(_COMPONENTS_MODULE, payload["_type"], expect=ComponentSpec)
    kwargs = {k: v for k, v in payload.items() if k not in {"_type", "transforms"}}
    if "transforms" in payload:
        kwargs["transforms"] = [transform_from_dict(t) for t in payload["transforms"]]
    return cls(**kwargs)


def transform_to_dict(transform: ComponentTransform) -> dict[str, Any]:
    payload: dict[str, Any] = {"_type": type(transform).__name__}
    for key, value in vars(transform).items():
        if key.startswith("_"):
            continue
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if value is None:
            continue
        payload[key] = value
    return payload


def transform_from_dict(payload: dict[str, Any]) -> ComponentTransform:
    cls = _resolve(_TRANSFORMS_MODULE, payload["_type"], expect=ComponentTransform)
    kwargs = {k: v for k, v in payload.items() if k != "_type"}
    return cls(**kwargs)


def components_to_dict(components: "list[ComponentSpec]") -> list[dict[str, Any]]:
    return [component_to_dict(c) for c in components]


def components_from_dict(payload: "list[dict[str, Any]]") -> list[ComponentSpec]:
    return [component_from_dict(d) for d in payload]


def _resolve(module_path: str, class_name: str, *, expect: type) -> type:
    module = importlib.import_module(module_path)
    try:
        cls = getattr(module, class_name)
    except AttributeError as err:
        raise ValueError(f"Unknown {expect.__name__} class {class_name!r} in {module_path}") from err
    if not (isinstance(cls, type) and issubclass(cls, expect)):
        raise ValueError(f"{module_path}.{class_name} is not a {expect.__name__} subclass")
    return cls


__all__ = [
    "component_from_dict",
    "component_to_dict",
    "components_from_dict",
    "components_to_dict",
    "transform_from_dict",
    "transform_to_dict",
]
