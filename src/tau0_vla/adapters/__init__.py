"""Per-embodiment adapters.

One subdirectory per robot, each covering the specialisation layers it needs —
data layout and deploy I/O. The general
pipeline in ``tau0_vla.data`` imports *from* here; nothing here is imported
*by* the pipeline's interior.

Subpackages are not eagerly imported. Import the adapter you want explicitly,
for example ``from tau0_vla.adapters.g1 import G1Agibot``.
"""

__all__: list[str] = []
