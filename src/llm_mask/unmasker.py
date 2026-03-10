from __future__ import annotations

from pathlib import Path

from .mapping import MaskingResult


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

    The LLM sometimes keeps a prefix of the original value as literal context
    and only replaces the remaining suffix with the placeholder, e.g.:
    - original = "д. 145", masked text = "д. <building_1>"  → "д. д. 145" (bad)
    - original = "запись №12", masked text = "запись №<id_2>" → "запись №запись №12" (bad)
    - original = "№МК-4567890", masked text = "№<doc_4>"      → "№№МК-4567890" (bad)

    This function detects any prefix of ``original`` that appears immediately
    before ``placeholder`` in the text, and replaces the combined span
    ``(prefix + placeholder)`` with ``original`` as a whole — avoiding the
    duplication.  Any remaining bare occurrences are then replaced normally.
    """
    # Strip spurious surrounding quotes the LLM may have added around the placeholder
    # (e.g. masked text has `"<id_1>"` but original is `987654321` without quotes).
    if not (original.startswith('"') or original.startswith("'")):
        for quote in ('"', "'"):
            quoted_ph = quote + placeholder + quote
            if quoted_ph in text:
                text = text.replace(quoted_ph, original)

    # Try prefixes of `original` from longest (all but last char) to shortest (1 char).
    # If `original[:k] + placeholder` appears in text, replace it with original.
    for k in range(len(original) - 1, 0, -1):
        combined = original[:k] + placeholder
        if combined in text:
            text = text.replace(combined, original)
            break

    # Replace any remaining bare occurrences.
    return text.replace(placeholder, original)
