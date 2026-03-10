import re


class ChunkMerger:
    """Merge masking results from multiple chunks into a single consistent result.

    Guarantees:
    - The same original entity always gets the same placeholder across chunks.
    - Placeholder counters per type are globally unique (no ``<url_1>`` from
      chunk 2 colliding with ``<url_1>`` from chunk 1 if they refer to
      different originals).
    - Regex-preloaded placeholders are never silently overwritten by LLM
      collisions — the collision is resolved at the count level only.
    """

    def __init__(self, preloaded: dict[str, str] | None = None) -> None:
        self._global_registry: dict[str, str] = {}
        self._counters: dict[str, int] = {}
        # Set of placeholder values owned by the regex pre-masker.
        # When a LLM chunk_placeholder collides with one of these, the
        # substitution must not clobber already-placed regex tokens in text.
        self._preloaded_values: set[str] = set()
        if preloaded:
            self._global_registry.update(preloaded)
            self._preloaded_values = set(preloaded.values())
            for ph in preloaded.values():
                ph_type = _extract_type(ph)
                m = re.search(r"\d+", ph)
                if m:
                    n = int(m.group())
                    if n > self._counters.get(ph_type, 0):
                        self._counters[ph_type] = n

    def add_chunk(
        self,
        masked_text: str,
        mapping: dict[str, str],
        pre_masked_chunk: str = "",
    ) -> tuple[str, dict[str, str]]:
        """Normalise *mapping* against the global registry and patch *masked_text*.

        Parameters
        ----------
        masked_text:
            Text produced by the LLM for this chunk (already has placeholders).
        mapping:
            ``{original: placeholder}`` as returned by ``parse_llm_response``.
        pre_masked_chunk:
            The regex-pre-masked text that was *sent to* the LLM.  Used to
            determine how many occurrences of each placeholder are "protected"
            (placed by the regex layer) versus "new" (placed by the LLM).

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

        patched = _apply_substitutions_safe(
            masked_text, substitutions, self._preloaded_values, pre_masked_chunk
        )
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


def _apply_substitutions_safe(
    text: str,
    subs: dict[str, str],
    preloaded_values: set[str],
    pre_masked: str,
) -> str:
    """Apply substitutions, protecting regex-preloaded placeholder occurrences.

    For placeholders that belong to the regex layer (``preloaded_values``),
    only the *extra* occurrences — those the LLM introduced beyond what was
    already in ``pre_masked`` — are replaced.  This prevents the merger from
    clobbering a correctly-placed regex token when the LLM reuses the same
    placeholder number for a different entity.
    """
    if not subs:
        return text
    for old in sorted(subs, key=len, reverse=True):
        new = subs[old]
        if old in preloaded_values and pre_masked:
            protected = pre_masked.count(old)
            total = text.count(old)
            extra = total - protected
            if extra <= 0:
                # All occurrences are from the regex layer — nothing to replace.
                continue
            # Replace the last `extra` occurrences (LLM-placed tokens tend to
            # appear after the regex-placed ones in document order).
            for _ in range(extra):
                pos = text.rfind(old)
                if pos == -1:
                    break
                text = text[:pos] + new + text[pos + len(old):]
        else:
            text = text.replace(old, new)
    return text
