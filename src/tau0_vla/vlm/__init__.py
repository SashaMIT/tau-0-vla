"""The bridge from the data pipeline to VLM tokenization.

``dataset`` wraps a pipeline loader as a torch ``Dataset`` and collates
batches for the VLM; ``policy_transforms`` registers the policy-side transforms
that YAML transform specs resolve through. Imported for their side effects in
the training path, so import the module rather than expecting re-exports here.
"""

__all__: list[str] = []
