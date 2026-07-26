"""Skeleton adapter — copy this directory, rename it, fill in the blanks.

    cp -r src/tau0_vla/adapters/_template src/tau0_vla/adapters/my_robot

Every value you must supply is spelled ``<YOUR_...>``, so the two files will not
run until you have replaced them all — that is deliberate. Grep for ``<YOUR_``
to find what is left.

This skeleton covers a **joint-controlled** robot, which is the common case: arm
joints plus grippers, no end-effector pose. Native EEF datasets should declare
their EEF fields directly; the public release does not synthesize EEF from
joints.

``../g1/`` is the worked example and the authority. When this skeleton and the
G1 adapter disagree about how something works, the G1 adapter is right: it is
the one exercised by the bundled real-data example.
"""

from tau0_vla.adapters._template.layout import TemplateObservation, TemplateRobot

__all__ = [
    "TemplateObservation",
    "TemplateRobot",
]
