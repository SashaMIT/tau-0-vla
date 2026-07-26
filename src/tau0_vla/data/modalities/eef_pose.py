from tau0_vla.data.modalities.base import ComponentSpec
from tau0_vla.data.modalities.transforms import RelativeToState


class EefPose(ComponentSpec):
    component_key = "eef_pose"

    def __init__(self, *, abs2relative: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.abs2relative = abs2relative

        if abs2relative and not any(isinstance(transform, RelativeToState) for transform in self.transforms):
            self.transforms.append(RelativeToState())

    def bind(self, *, field_key: str) -> ComponentSpec:
        return type(self)(
            key=self.key,
            abs2relative=self.abs2relative,
            normalize=self.normalize,
            field_key=field_key,
            transforms=self.transforms[:-1] if self.abs2relative else self.transforms,
        )
