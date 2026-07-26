from tau0_vla.data.modalities.base import ComponentSpec
from tau0_vla.data.modalities.transforms import ImpulseToStep


class Gripper(ComponentSpec):
    component_key = "gripper"

    def __init__(self, *, imp2step: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.imp2step = imp2step

        if imp2step and not any(isinstance(transform, ImpulseToStep) for transform in self.transforms):
            self.transforms.append(ImpulseToStep())

    def bind(self, *, field_key: str) -> ComponentSpec:
        return type(self)(
            key=self.key,
            imp2step=self.imp2step,
            normalize=self.normalize,
            field_key=field_key,
            transforms=self.transforms[:-1] if self.imp2step else self.transforms,
        )
