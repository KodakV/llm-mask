import difflib
import re

from .exceptions import ParsingError

_SEPARATORS = ("Mapping замен", "Mapping")
_PLACEHOLDER_RE = re.compile(r"^(<[a-z_]+_\d+>|[a-z_]+_\d+)$")
_SEPARATOR_LINE_RE = re.compile(r"^[-=*]{3,}$")
_PH_IN_TEXT_RE = re.compile(r"<[a-z_]+_\d+>|(?<!\w)[a-z_]+_\d+(?!\w)")

# Detects lines where BOTH sides are the same placeholder: "<ph_1> -> <ph_1>"
# The model sometimes appends a dummy example block like this at the end.
_SELF_REF_LINE_RE = re.compile(
    r"^(<[a-z_]+_\d+>|[a-z_]+_\d+)\s*->\s*(<[a-z_]+_\d+>|[a-z_]+_\d+)\s*$"
)

# Characters/patterns that indicate a value is a secret or token, not a name.
_SECRET_PATTERN_RE = re.compile(
    r"[:/\\{}\[\]@]"            # URL / path / email characters
    r"|eyJ"                      # JWT header
    r"|[A-Za-z0-9+/]{20,}"      # long base64-like token
    r"|[A-Z0-9]{16,}"           # long uppercase key (AWS etc.)
)


def parse_llm_response(raw: str) -> tuple[str, dict[str, str]]:
    """Split an LLM response into (masked_text, mapping).

    The response format expected from the LLM::

        <masked document text>

        Mapping замен          <- or "Mapping" / "---" / any header the model chooses
        original -> <placeholder>
        ...

    The mapping direction is auto-detected: both ``original -> placeholder``
    and ``placeholder -> original`` are accepted.

    The parser first tries known separator keywords; if none are found it
    falls back to locating the last paragraph where every non-empty line
    contains ``->``.

    Returns
    -------
    masked_text : str
        The anonymised document.
    mapping : dict[str, str]
        ``{"original": "<placeholder>", ...}``

    Raises
    ------
    ParsingError
        If no mapping section can be identified.
    """
    for sep in _SEPARATORS:
        if sep in raw:
            masked_text, mapping_raw = raw.split(sep, 1)
            masked_text = _strip_self_ref_mapping_suffix(masked_text.strip())
            return masked_text, _parse_mapping_lines(mapping_raw)

    paragraphs = raw.split("\n\n")
    for i in range(len(paragraphs) - 1, -1, -1):
        lines = [line.strip() for line in paragraphs[i].splitlines() if line.strip()]
        content = [line for line in lines if not _SEPARATOR_LINE_RE.match(line)]
        if content and all("->" in line for line in content):
            mapping = _parse_mapping_lines(paragraphs[i])
            if not mapping:
                continue
            end = i
            while end > 0 and _SEPARATOR_LINE_RE.match(paragraphs[end - 1].strip()):
                end -= 1
            masked_text = "\n\n".join(paragraphs[:end]).strip()
            masked_text = _strip_self_ref_mapping_suffix(masked_text)
            return masked_text, mapping

    if "->" not in raw:
        return raw.strip(), {}

    # "->" present but no valid mapping paragraph found — the arrow is likely
    # in the document body (code, diagrams, descriptions).  Return the full
    # text as masked content with an empty mapping rather than raising, so the
    # caller can still score whatever entities were removed implicitly.
    if _PH_IN_TEXT_RE.search(raw):
        return raw.strip(), {}

    raise ParsingError(
        f"LLM response does not contain a recognisable mapping section. "
        f"Raw response snippet: {raw[:200]!r}"
    )


def _strip_self_ref_mapping_suffix(text: str) -> str:
    """Remove a trailing block of self-referential mapping lines.

    Some models append a dummy example block at the end of the masked text
    that looks like::

        <url_1> -> <url_1>
        <email_1> -> <email_1>
        ...

    These lines contribute nothing to the mapping and pollute the masked
    document.  This function strips them (and any surrounding blank lines /
    separator lines) from the end of *text*.
    """
    lines = text.splitlines()
    # Walk backwards, skipping blank / separator / self-ref lines.
    end = len(lines)
    while end > 0:
        stripped = lines[end - 1].strip()
        if not stripped or _SEPARATOR_LINE_RE.match(stripped) or _SELF_REF_LINE_RE.match(stripped):
            end -= 1
        else:
            break
    return "\n".join(lines[:end]).strip()


def recover_mapping(original: str, masked: str, mapping: dict[str, str]) -> dict[str, str]:
    """Recover placeholder→original pairs the LLM omitted from the mapping section.

    Compares the original text against a partially-restored version of the
    masked text (known placeholders replaced back) using a word-level diff to
    identify spans that correspond to unmapped placeholders.

    Parameters
    ----------
    original:
        The original (unmasked) text that was sent to the LLM.
    masked:
        The masked text returned by the LLM.
    mapping:
        The partial mapping already parsed from the LLM response.

    Returns
    -------
    dict[str, str]
        A potentially extended mapping with recovered entries added.
    """
    def _bare(ph: str) -> str:
        """Strip angle brackets for bracket-insensitive comparison."""
        return ph[1:-1] if ph.startswith("<") and ph.endswith(">") else ph

    known_phs = set(mapping.values())
    known_bare = {_bare(ph) for ph in known_phs}
    all_phs = set(_PH_IN_TEXT_RE.findall(masked))
    # Missing: placeholders in text whose bare form is not yet in mapping
    missing = {ph for ph in all_phs if _bare(ph) not in known_bare}

    if not missing:
        return mapping

    inverted = {v: k for k, v in mapping.items()}
    partial = masked
    for ph in sorted(inverted, key=len, reverse=True):
        partial = partial.replace(ph, inverted[ph])

    orig_words = original.split()
    part_words = partial.split()

    recovered: dict[str, str] = {}
    matcher = difflib.SequenceMatcher(None, orig_words, part_words, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "replace":
            continue

        # Skip spans that are too long to be a single entity (> 6 words).
        if i2 - i1 > 6:
            continue

        orig_span = _strip_markdown(" ".join(orig_words[i1:i2]))
        # Strip trailing punctuation from word-split artefacts
        # (e.g. "Викторовна," when name is followed by a comma in the original).
        orig_span = orig_span.rstrip(",;")

        # Skip spans that look like secrets / tokens / URLs rather than names.
        if _SECRET_PATTERN_RE.search(orig_span):
            continue

        for ph in missing:
            if ph in part_words[j1:j2]:
                recovered[orig_span] = ph
                break

    return {**mapping, **recovered}


_SELF_REF_IN_LINE_RE = re.compile(
    r"(<[a-z_]+_\d+>|[a-z_]+_\d+)(\s*->\s*)(\1)"
)


def repair_arrow_collisions(
    masked: str, original: str, mapping: dict[str, str]
) -> str:
    """Fix ``<ph> -> <ph>`` self-references caused by ``->`` operators in code.

    Network / firewall rules use ``->`` as a direction symbol::

        ALLOW 192.168.100.10 -> 10.0.2.10:5432

    The model sometimes assigns the same placeholder to both sides.
    This function detects those patterns line-by-line, looks up the correct
    placeholder for the right-hand-side value, and fixes the masked text.
    """
    inverted = {ph: orig for orig, ph in mapping.items()}
    masked_lines = masked.splitlines()
    orig_lines   = original.splitlines()

    result: list[str] = []
    orig_cursor = 0

    for mline in masked_lines:
        m = _SELF_REF_IN_LINE_RE.search(mline)
        if not m:
            result.append(mline)
            continue

        ph      = m.group(1)
        ph_orig = inverted.get(ph, "")

        # Find the closest original line that contains ph_orig and "->"
        found_orig = None
        search_start = max(0, orig_cursor - 3)
        search_end   = min(len(orig_lines), orig_cursor + 10)
        for oi in range(search_start, search_end):
            if ph_orig and ph_orig in orig_lines[oi] and "->" in orig_lines[oi]:
                found_orig = orig_lines[oi]
                orig_cursor = oi + 1
                break

        if not found_orig:
            result.append(mline)
            continue

        # Right-hand side of the -> in the original line
        arrow_pos  = found_orig.find("->")
        right_part = found_orig[arrow_pos + 2:]

        # Find the longest mapped original value that appears in right_part
        # and has a DIFFERENT placeholder.
        correct_ph  = None
        best_len    = 0
        for orig_val, pl in mapping.items():
            if pl != ph and orig_val in right_part and len(orig_val) > best_len:
                correct_ph = pl
                best_len   = len(orig_val)

        if correct_ph:
            fixed = mline[:m.start(3)] + correct_ph + mline[m.end(3):]
            result.append(fixed)
        else:
            result.append(mline)

    return "\n".join(result)


_MD_STRIP_RE = re.compile(r"^[*_`\[\]#>|\\]+|[*_`\[\]#|\\]+$")


def _strip_markdown(text: str) -> str:
    """Strip leading/trailing markdown decoration from an entity key."""
    return _MD_STRIP_RE.sub("", text).strip()


def _parse_mapping_lines(mapping_raw: str) -> dict[str, str]:
    """Parse mapping lines, auto-detecting direction.

    Accepts both ``original -> <placeholder>`` and ``<placeholder> -> original``.
    Lines where neither side is a valid placeholder are silently skipped.
    """
    result: dict[str, str] = {}
    ph_seen: dict[str, str] = {}  # placeholder → first original that claimed it
    line_counts: dict[str, int] = {}  # raw line → occurrence count
    for line in mapping_raw.splitlines():
        line = line.strip()
        if not line or "->" not in line:
            continue
        # Loop guard: if the same raw line appears 3+ times, the model is in a
        # repetition loop — stop parsing to avoid processing thousands of dupes.
        line_counts[line] = line_counts.get(line, 0) + 1
        if line_counts[line] >= 3:
            break
        parts = line.split("->", 1)
        left = parts[0].strip()
        right = parts[1].strip()
        if not left or not right:
            continue
        if _PLACEHOLDER_RE.match(left):
            original, placeholder = right, left
        elif _PLACEHOLDER_RE.match(right):
            original, placeholder = left, right
        else:
            continue
        original = _strip_markdown(original)
        # Strip trailing punctuation that leaked from surrounding context
        # (e.g. a comma after a name in a sentence).
        original = original.rstrip(",;")
        if not original or _PLACEHOLDER_RE.match(original):
            continue
        # Skip keys that contain existing placeholder tokens — these are
        # artefacts where the LLM included already-masked spans in the key
        # (e.g. "ООО «Альфа», ОГРН <doc_3>, ИНН <doc_1>" → <doc_11>).
        if _PH_IN_TEXT_RE.search(original):
            continue
        # Skip keys that contain the mapping arrow — these are firewall /
        # network rules that the model mistakenly tried to map.
        if "->" in original:
            continue
        # DUP-PH guard: skip if this placeholder was already claimed
        # by a different original (repair_dup_ph handles it post-LLM).
        if placeholder in ph_seen and ph_seen[placeholder] != original:
            continue
        ph_seen[placeholder] = original
        result[original] = placeholder
    return result
