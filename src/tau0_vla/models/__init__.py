"""Model architectures.

``model_builder.ModelBuilder`` is the entry point: it loads a checkpoint,
replaces attention, extends the FAST vocabulary, and applies the freezing or
LoRA configuration, in that order. Import it directly — this package does not
import its submodules eagerly, since doing so pulls in ``flash_attn``.
"""

__all__: list[str] = []
