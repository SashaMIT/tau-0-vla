"""Language modality — the `Prompt` class.

A ``Prompt(template="pick up the {object}")`` carries the template string that
turns a raw task instruction into the canonical ``payload["prompt"]`` text.
Tokenization is framework-specific (handled by the policy repo's own
processor), so `Prompt` is intentionally thin — it only owns the template
format step.
"""

from __future__ import annotations

import dataclasses
import string

_INSTRUCTION_KEY = "instruction"


@dataclasses.dataclass(frozen=True)
class Prompt:
    """Declarative prompt template.

    Typical usage::

        # Identity (pass instruction through unchanged)
        prompt = Prompt(template="{instruction}")
        text = prompt.format("pick up the red block")
        # → "pick up the red block"

        # Prefix template
        prompt = Prompt(template="Task: {instruction}.")
        text = prompt.format("pick up the red block")
        # → "Task: pick up the red block."

        # Extra keyword placeholders
        prompt = Prompt(template="pick up the {object}")
        text = prompt.format("", object="red block")
        # → "pick up the red block"

    If the template has no ``{instruction}`` placeholder and no extra kwargs
    are given, calling ``prompt.format(text)`` concatenates: ``template + text``.
    """

    template: str = "{instruction}"

    def format(self, instruction: str = "", /, **extras: object) -> str:
        """Apply the template to an instruction string.

        Bare positional placeholders (``{}``) are treated as
        ``{instruction}`` — tolerates checkpoints saved before the
        ``_build_data_spec`` writer fix, which hard-coded ``"{}"`` into
        ``prompt_template`` regardless of the config.
        """
        template, fields = _normalize_template(self.template)
        params: dict[str, object] = {_INSTRUCTION_KEY: instruction, **extras}
        if fields and fields <= set(params):
            return template.format(**params)
        if not fields:
            # No placeholders — concatenate (useful for pure prefixes).
            return template + instruction
        missing = fields - set(params)
        raise KeyError(
            f"Prompt template {self.template!r} is missing values for placeholders: "
            f"{sorted(missing)}. Provide them as keyword args to `prompt.format(...)`."
        )


def _normalize_template(template: str) -> tuple[str, set[str]]:
    """Parse ``template`` and rebuild it with any empty-name ``{}``
    placeholder renamed to ``{instruction}``. Returns ``(rebuilt_template,
    named_fields)``.
    """
    parts: list[str] = []
    fields: set[str] = set()
    for literal, field_name, format_spec, conversion in string.Formatter().parse(template):
        parts.append(literal)
        if field_name is None:
            continue
        name = field_name if field_name else _INSTRUCTION_KEY
        fields.add(name)
        conv = f"!{conversion}" if conversion else ""
        spec = f":{format_spec}" if format_spec else ""
        parts.append("{" + name + conv + spec + "}")
    return "".join(parts), fields


__all__ = ["Prompt"]
