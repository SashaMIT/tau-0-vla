"""tau-0-vla — training and inference for a vision-language-action model.

Subpackages are not eagerly imported. The model stack pulls in heavyweight
optional dependencies such as ``flash_attn``; a caller that only needs the data
pipeline should not pay for them. Import what you need explicitly::

    from tau0_vla.data import FinchDataLoader
    from tau0_vla.models.model_builder import ModelBuilder
"""

__all__: list[str] = []
