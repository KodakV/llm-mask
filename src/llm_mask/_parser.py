import difflib
import re

from .exceptions import ParsingError

_SEPARATORS = ("Mapping замен", "Mapping")
_PLACEHOLDER_RE = re.compile(r"^(<[a-z_]+_\d+>|[a-z_]+_\d+)$")
_SEPARATOR_LINE_RE = re.compile(r"^[-=*]{3,}$")
_PH_IN_TEXT_RE = re.compile(r"<[a-z_]+_\d+>|(?<!\w)[a-z_]+_\d+(?!\w)")


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
            return masked_text.strip(), _parse_mapping_lines(mapping_raw)

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
    known_phs = set(mapping.values())
    all_phs = set(_PH_IN_TEXT_RE.findall(masked))
    missing = all_phs - known_phs

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
        orig_span = _strip_markdown((" ".join(orig_words[i1:i2])))
        for ph in missing:
            if ph in part_words[j1:j2]:
                recovered[orig_span] = ph
                break

    return {**mapping, **recovered}


_MD_STRIP_RE = re.compile(r"^[*_`\[\]#>|\\]+|[*_`\[\]#>|\\]+$")


def _strip_markdown(text: str) -> str:
    """Strip leading/trailing markdown decoration from an entity key."""
    return _MD_STRIP_RE.sub("", text).strip()


def _parse_mapping_lines(mapping_raw: str) -> dict[str, str]:
    """Parse mapping lines, auto-detecting direction.

    Accepts both ``original -> <placeholder>`` and ``<placeholder> -> original``.
    Lines where neither side is a valid placeholder are silently skipped.
    """
    result: dict[str, str] = {}
    for line in mapping_raw.splitlines():
        line = line.strip()
        if not line or "->" not in line:
            continue
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
        if original and not _PLACEHOLDER_RE.match(original):
            result[original] = placeholder
    return result
