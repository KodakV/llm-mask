"""NER-based pre-masker using natasha (optional dependency)."""
from __future__ import annotations

import re

try:
    from natasha import (  # type: ignore[import-untyped]
        Doc,
        NewsEmbedding,
        NewsMorphTagger,
        NewsNERTagger,
        Segmenter,
    )

    _NATASHA_AVAILABLE = True
    _SEGMENTER = Segmenter()
    _EMB = NewsEmbedding()
    _MORPH_TAGGER = NewsMorphTagger(_EMB)
    _NER_TAGGER = NewsNERTagger(_EMB)
except ImportError:
    _NATASHA_AVAILABLE = False
    Doc = None  # type: ignore[assignment,misc]
    _SEGMENTER = _EMB = _MORPH_TAGGER = _NER_TAGGER = None  # type: ignore[assignment]

_PLACEHOLDER_RE = re.compile(r"<[a-z_]+_\d+>")
_COUNTER_SCAN_RE = re.compile(r"<(person|company|location)_(\d+)>")


class NerMasker:
    """Apply NER-based masking for PER/ORG/LOC entities using natasha."""

    available: bool = _NATASHA_AVAILABLE

    _TYPE_MAP: dict[str, str] = {"PER": "person", "ORG": "company", "LOC": "location"}

    def mask(
        self, text: str, preloaded: dict[str, str] | None = None
    ) -> tuple[str, dict[str, str]]:
        """Return *(masked_text, ner_mapping)*.

        If natasha is not installed, returns *(text, {})* unchanged.
        Counter offsets are derived from *preloaded* to avoid placeholder clashes.
        """
        if not self.available:
            return text, {}

        # Start counters at 1; bump up past any existing placeholders in preloaded.
        counters: dict[str, int] = {"person": 1, "company": 1, "location": 1}
        if preloaded:
            for value in preloaded.values():
                for etype, num_str in _COUNTER_SCAN_RE.findall(value):
                    counters[etype] = max(counters[etype], int(num_str) + 1)

        doc = Doc(text)  # type: ignore[misc]
        doc.segment(_SEGMENTER)
        doc.tag_morph(_MORPH_TAGGER)
        doc.tag_ner(_NER_TAGGER)

        registry: dict[str, str] = {}
        chars = list(text)

        # Process spans in reverse so character offsets stay valid after replacement.
        for span in sorted(doc.spans, key=lambda s: s.start, reverse=True):
            if span.type not in self._TYPE_MAP:
                continue

            raw = text[span.start : span.stop]

            if _PLACEHOLDER_RE.fullmatch(raw.strip()):
                continue

            etype = self._TYPE_MAP[span.type]

            if raw in registry:
                placeholder = registry[raw]
            else:
                placeholder = f"<{etype}_{counters[etype]}>"
                registry[raw] = placeholder
                counters[etype] += 1

            chars[span.start : span.stop] = list(placeholder)

        return "".join(chars), registry
