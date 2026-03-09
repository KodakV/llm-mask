"""Post-LLM repair: detect and fix placeholder collisions (DUP-PH)."""
from __future__ import annotations

import re

_PH_ANGLE_RE = re.compile(r"<([a-z_]+)_(\d+)>")
_PH_BARE_RE  = re.compile(r"^([a-z_]+)_(\d+)$")


def _parse_ph(ph: str) -> tuple[str, int] | None:
    """Return (type, number) for a placeholder, or None if not a placeholder."""
    m = _PH_ANGLE_RE.fullmatch(ph)
    if m:
        return m.group(1), int(m.group(2))
    m = _PH_BARE_RE.fullmatch(ph)
    if m:
        return m.group(1), int(m.group(2))
    return None


def _make_ph(ph_type: str, n: int, angle: bool) -> str:
    return f"<{ph_type}_{n}>" if angle else f"{ph_type}_{n}"


def repair_dup_ph(
    masked_text: str, mapping: dict[str, str]
) -> tuple[str, dict[str, str]]:
    """Detect placeholders assigned to multiple originals and reallocate.

    Strategy: keep the first original→placeholder assignment unchanged.
    Each subsequent original that shares the same placeholder gets a new,
    unique placeholder.  The *second* occurrence of the duplicate in the
    masked text is patched to the new placeholder, and so on.

    Returns *(repaired_text, repaired_mapping)*.
    """
    # Group originals by placeholder
    ph_to_origs: dict[str, list[str]] = {}
    for orig, ph in mapping.items():
        ph_to_origs.setdefault(ph, []).append(orig)

    # Short-circuit: nothing to fix
    if all(len(v) == 1 for v in ph_to_origs.values()):
        return masked_text, dict(mapping)

    # Build set of all currently-used numbers per type
    used: dict[str, set[int]] = {}
    for ph in mapping.values():
        parsed = _parse_ph(ph)
        if parsed:
            ph_type, ph_num = parsed
            used.setdefault(ph_type, set()).add(ph_num)

    def _next_number(ph_type: str) -> int:
        n = max(used.get(ph_type, {0})) + 1
        while n in used.get(ph_type, set()):
            n += 1
        used.setdefault(ph_type, set()).add(n)
        return n

    new_mapping: dict[str, str] = {}
    text = masked_text

    for ph, origs in ph_to_origs.items():
        # First original keeps the original placeholder
        new_mapping[origs[0]] = ph

        for orig in origs[1:]:
            parsed = _parse_ph(ph)
            if parsed is None:
                # Can't parse — just keep original assignment, skip repair
                new_mapping[orig] = ph
                continue

            ph_type, _ = parsed
            angle = ph.startswith("<")
            n = _next_number(ph_type)
            new_ph = _make_ph(ph_type, n, angle)
            new_mapping[orig] = new_ph

            # Patch the text: replace the *next* occurrence of the old ph
            # with the new one (occurrences assumed to be in document order)
            idx = text.find(ph)
            if idx != -1:
                next_idx = text.find(ph, idx + len(ph))
                if next_idx != -1:
                    text = text[:next_idx] + new_ph + text[next_idx + len(ph):]

    return text, new_mapping
