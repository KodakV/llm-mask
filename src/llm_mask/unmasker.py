from __future__ import annotations

import re
from pathlib import Path

from .mapping import MaskingResult

_ADDRESS_PREFIXES: frozenset[str] = frozenset({
    "г.", "д.", "кв.", "ул.", "пр.", "наб.", "корп.", "стр.",
    "оф.", "индекс", "серия", "номер",
})


class Unmasker:
    """Reverse a masking operation via plain string replacement (no LLM needed)."""

    def unmask(self, masked_text: str, mapping: dict[str, str]) -> str:
        """Replace all placeholders in *masked_text* with their originals."""
        inverted = {v: k for k, v in mapping.items()}
        for placeholder in sorted(inverted, key=len, reverse=True):
            original = inverted[placeholder]
            masked_text = _context_replace(masked_text, placeholder, original)
        return masked_text

    def unmask_file(
        self, masked_file: str | Path, mapping_file: str | Path
    ) -> str:
        """Load a masked file and its mapping JSON, return the restored text."""
        masked_text = Path(masked_file).read_text(encoding="utf-8")
        mapping = MaskingResult.load_mapping(mapping_file)
        return self.unmask(masked_text, mapping)


def _context_replace(text: str, placeholder: str, original: str) -> str:
    """Replace *placeholder* with *original*, eliminating prefix duplication.

    If the original starts with an address/context prefix word (e.g. "д.")
    and that same word immediately precedes *placeholder* in the text, the
    combined ``"prefix placeholder"`` span is replaced by *original* as a
    whole (avoiding ``"д. д. 145"``).  All remaining bare occurrences of
    *placeholder* are then replaced normally.
    """
    words = original.split()
    if len(words) > 1:
        first = words[0]
        if first.lower() in _ADDRESS_PREFIXES or (first.endswith(".") and len(first) <= 4):
            ph_esc = re.escape(placeholder)
            first_esc = re.escape(first)
            # Replace "first_word <placeholder>" → full original
            text = re.sub(
                first_esc + r"\s+" + ph_esc,
                lambda _: original,
                text,
                flags=re.IGNORECASE,
            )
    # Replace any remaining bare occurrences
    return text.replace(placeholder, original)
