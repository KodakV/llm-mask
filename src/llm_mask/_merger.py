import re


class ChunkMerger:
    """Merge masking results from multiple chunks into a single consistent result.

    Guarantees:
    - The same original entity always gets the same placeholder across chunks.
    - Placeholder counters per type are globally unique (no ``<url_1>`` from
      chunk 2 colliding with ``<url_1>`` from chunk 1 if they refer to
      different originals).
    """

    def __init__(self) -> None:
        self._global_registry: dict[str, str] = {}
        self._counters: dict[str, int] = {}

    def add_chunk(
        self, masked_text: str, mapping: dict[str, str]
    ) -> tuple[str, dict[str, str]]:
        """Normalise *mapping* against the global registry and patch *masked_text*.

        Returns the (patched_text, normalised_mapping) pair for this chunk.
        """
        substitutions: dict[str, str] = {}
        normalised: dict[str, str] = {}

        for original, chunk_placeholder in mapping.items():
            if original in self._global_registry:
                global_ph = self._global_registry[original]
            else:
                global_ph = self._allocate(chunk_placeholder)
                self._global_registry[original] = global_ph

            if chunk_placeholder != global_ph:
                substitutions[chunk_placeholder] = global_ph

            normalised[original] = global_ph

        patched = _apply_substitutions(masked_text, substitutions)
        return patched, normalised

    def global_mapping(self) -> dict[str, str]:
        return dict(self._global_registry)

    def _allocate(self, template_placeholder: str) -> str:
        """Allocate the next available placeholder for the given type."""
        ph_type = _extract_type(template_placeholder)
        count = self._counters.get(ph_type, 0) + 1
        self._counters[ph_type] = count
        return _format_placeholder(ph_type, count)


_PH_ANGLE = re.compile(r"<([a-z_]+)_\d+>")
_PH_BARE = re.compile(r"\b([a-z_]+)_\d+\b")


def _extract_type(placeholder: str) -> str:
    m = _PH_ANGLE.fullmatch(placeholder)
    if m:
        return m.group(1)
    m = _PH_BARE.fullmatch(placeholder)
    if m:
        return m.group(1)
    return placeholder


def _format_placeholder(ph_type: str, n: int) -> str:
    # Types that use bare format (no angle brackets) as defined in prompts
    bare_types = {"service", "project"}
    if ph_type in bare_types:
        return f"{ph_type}_{n}"
    return f"<{ph_type}_{n}>"


def _apply_substitutions(text: str, subs: dict[str, str]) -> str:
    if not subs:
        return text
    for old in sorted(subs, key=len, reverse=True):
        text = text.replace(old, subs[old])
    return text
