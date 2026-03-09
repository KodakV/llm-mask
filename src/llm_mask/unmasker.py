from pathlib import Path

from .mapping import MaskingResult


class Unmasker:
    """Reverse a masking operation via plain string replacement (no LLM needed)."""

    def unmask(self, masked_text: str, mapping: dict[str, str]) -> str:
        """Replace all placeholders in *masked_text* with their originals."""
        inverted = {v: k for k, v in mapping.items()}
        for placeholder in sorted(inverted, key=len, reverse=True):
            masked_text = masked_text.replace(placeholder, inverted[placeholder])
        return masked_text

    def unmask_file(
        self, masked_file: str | Path, mapping_file: str | Path
    ) -> str:
        """Load a masked file and its mapping JSON, return the restored text."""
        masked_text = Path(masked_file).read_text(encoding="utf-8")
        mapping = MaskingResult.load_mapping(mapping_file)
        return self.unmask(masked_text, mapping)
