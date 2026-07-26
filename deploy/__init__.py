"""Serving and evaluation for trained checkpoints.

``policy.Tau0VLAPolicy`` is the inference wrapper; ``server`` exposes it over
HTTP; ``openloop`` scores predicted action chunks against recorded ground truth.
``_bootstrap`` must run before any of them so that ``@register_config`` has
fired — each entry point calls it itself.
"""

__all__: list[str] = []
