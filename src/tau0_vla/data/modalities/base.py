from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import numpy as np


@dataclasses.dataclass(frozen=True)
class WorkflowContext:
    state: dict[str, np.ndarray]
    action: dict[str, np.ndarray]
    state_already_transformed: bool = False
    # True when the state arrays carry a leading temporal axis from
    # ``RobotConfig.temporal_sequence["state"]``. Transforms that use state as
    # a *reference* (``RelativeToState`` / ``ImpulseToStep``) consult this to
    # decide whether to collapse to the current frame. Stats compute (batched,
    # ``(batch, 1, state_dim)``) leaves this False so the leading axis is
    # treated as the batch axis and broadcasts correctly against the action
    # chunk instead of being collapsed.
    state_has_temporal_axis: bool = False


class FieldDescription:
    def __init__(self, description: str, dimensions: int, indices: tuple[int, ...]):
        self.description = description
        self.dimensions = dimensions
        self.indices = indices
        if self.dimensions != len(self.indices):
            raise ValueError(f"dimensions={self.dimensions} does not match indices length={len(self.indices)}")

    def project(self, values: np.ndarray) -> np.ndarray:
        return np.take(values, self.indices, axis=-1)

    def to_json(self) -> dict[str, object]:
        return {
            "description": self.description,
            "dimensions": self.dimensions,
            "indices": list(self.indices),
        }

    def __repr__(self) -> str:
        return (
            "FieldDescription("
            f"description={self.description!r}, dimensions={self.dimensions!r}, indices={self.indices!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FieldDescription):
            return False
        return (
            self.description == other.description
            and self.dimensions == other.dimensions
            and self.indices == other.indices
        )


FieldDescriptionMap: TypeAlias = dict[str, FieldDescription]
ResolvedFieldDescriptions: TypeAlias = dict[str, dict[str, dict[str, Any]]]


def normalize_field_description_map(payload: dict[str, Any] | None) -> FieldDescriptionMap:
    if not payload:
        return {}
    normalized: FieldDescriptionMap = {}
    for key, value in payload.items():
        # Idempotent: if ``value`` is already a ``FieldDescription`` (e.g.
        # coming from a prior ``_normalize_field_descriptions`` pass), keep
        # it as-is instead of treating it as a raw dict.
        if isinstance(value, FieldDescription):
            normalized[key] = value
            continue
        normalized[key] = FieldDescription(
            description=str(value.get("description", "")),
            dimensions=int(value.get("dimensions", len(value.get("indices", [])))),
            indices=tuple(int(index) for index in value.get("indices", [])),
        )
    return normalized


def serialize_field_description_map(payload: FieldDescriptionMap) -> dict[str, dict[str, Any]]:
    return {key: value.to_json() for key, value in payload.items()}


def resolve_field_descriptions(payload: dict[str, Any] | None = None) -> ResolvedFieldDescriptions:
    """Normalize a raw ``{state, action}`` field_descriptions payload.

    Missing sides are returned as empty dicts rather than raising — robot
    configs may inject hardcoded fallbacks later in
    ``RobotConfig._normalize_field_descriptions`` for datasets whose
    ``meta/info.json`` predates the field_descriptions schema. Any robot that
    still requires them will raise the familiar
    ``"Dataset metadata must define state field_descriptions"`` there."""
    payload = payload or {}
    state = normalize_field_description_map(payload.get("state"))
    action = normalize_field_description_map(payload.get("action"))
    return {
        "state": serialize_field_description_map(state),
        "action": serialize_field_description_map(action),
    }


def extract_field_descriptions_from_dataset_info(dataset_info: dict[str, Any] | None) -> ResolvedFieldDescriptions:
    features = (dataset_info or {}).get("features", {})
    return resolve_field_descriptions(
        {
            "state": features.get("observation.state", {}).get("field_descriptions"),
            "action": features.get("action", {}).get("field_descriptions"),
        }
    )


NormalizeMode: TypeAlias = Literal["mean_std", "quantile", "none"]
TransformMode: TypeAlias = Literal["pre", "post"]


class ComponentTransform:
    uses_state_reference = False
    applies_to_state_reference = True

    def preserves_static_window_range(self, *, component: "ComponentSpec") -> bool:
        """Whether ``max(window) - min(window)`` is unchanged by this transform.

        Static-frame computation can reuse overlapping windows only for
        transforms with this property. Transforms opt in explicitly; the
        conservative default keeps the generic path.
        """
        del component
        return False

    def forward(
        self,
        value: np.ndarray,
        context: WorkflowContext,
        *,
        component: "ComponentSpec",
        state_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        del context, component, state_reference
        return value

    def inverse(
        self,
        value: np.ndarray,
        context: WorkflowContext,
        *,
        component: "ComponentSpec",
        state_reference: np.ndarray | None = None,
    ) -> np.ndarray:
        del context, component, state_reference
        return value


class ComponentSpec:
    component_key = ""

    def __init__(
        self,
        key: str | None = None,
        *,
        normalize: NormalizeMode,
        field_key: str | None = None,
        transforms: Sequence[ComponentTransform] | None = None,
    ):
        self.key = key or self.component_key
        self.normalize = normalize
        self.field_key = field_key
        self.transforms: list[ComponentTransform] = list(transforms or ())

    def bind(self, *, field_key: str) -> ComponentSpec:
        return type(self)(
            key=self.key,
            normalize=self.normalize,
            field_key=field_key,
            transforms=self.transforms,
        )

    def apply_transforms(self, value: np.ndarray, *, context: WorkflowContext, mode: TransformMode) -> np.ndarray:
        transformed = value
        prefixes = self._state_reference_prefixes(context)
        if mode == "pre":
            for index, transform in enumerate(self.transforms):
                transformed = transform.forward(
                    transformed,
                    context,
                    component=self,
                    state_reference=prefixes[index] if transform.uses_state_reference else None,
                )
            return transformed

        for index in reversed(range(len(self.transforms))):
            transform = self.transforms[index]
            transformed = transform.inverse(
                transformed,
                context,
                component=self,
                state_reference=prefixes[index] if transform.uses_state_reference else None,
            )
        return transformed

    def _state_reference_prefixes(self, context: WorkflowContext) -> list[np.ndarray | None]:
        state = context.state.get(self.key)
        if state is None:
            return [None] * (len(self.transforms) + 1)
        if context.state_already_transformed:
            return [state] * (len(self.transforms) + 1)

        prefixes: list[np.ndarray | None] = [state]
        transformed = state
        for transform in self.transforms:
            if transform.applies_to_state_reference:
                transformed = transform.forward(transformed, context, component=self)
            prefixes.append(transformed)
        return prefixes

    def transform(self, context: WorkflowContext, *, role: str, mode: TransformMode) -> np.ndarray:
        state = context.state.get(self.key)
        action = context.action.get(self.key)
        value = state if role == "state" else action
        if value is None:
            raise KeyError(f"Missing {role} value for component '{self.key}'")
        return self.apply_transforms(value, context=context, mode=mode)

    def requires_normalization(self) -> bool:
        return self.normalize != "none"

    def use_quantile_norm(self) -> bool:
        return self.normalize == "quantile"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(key={self.key!r}, normalize={self.normalize!r}, field_key={self.field_key!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ComponentSpec):
            return False
        return self.key == other.key and self.normalize == other.normalize and self.field_key == other.field_key
