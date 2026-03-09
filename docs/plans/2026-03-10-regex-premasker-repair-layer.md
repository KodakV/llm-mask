# Regex Pre-masker + Repair Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a deterministic regex pre-masking layer for structured PII types and a repair layer for DUP-PH, bringing recall/precision/reversibility from ~40-60% to ~90%+.

**Architecture:** `RegexMasker` runs on full text before LLM, deterministically replacing emails/phones/IPs/URLs/secrets/bank-accounts/СНИЛС/ИНН/ОГРН/КПП with typed placeholders. LLM receives pre-masked text and handles only fuzzy types (names, companies, hosts, addresses). A `repair_dup_ph` pass after LLM fixes cases where the model assigned the same placeholder to different values. `Unmasker` gets a context-aware restore that eliminates "д. д. 145"-style prefix duplication.

**Tech Stack:** Python 3.11+, `re`, existing `_merger.py` / `_parser.py` / `unmasker.py` / `client.py` infrastructure.

---

## Task 1: `_regex_masker.py` — Deterministic pre-masker

**Files:**
- Create: `src/llm_mask/_regex_masker.py`
- Create: `tests/test_regex_masker.py`

### Step 1: Write failing tests

```python
# tests/test_regex_masker.py
import pytest
from llm_mask._regex_masker import RegexMasker


def test_masks_email():
    m = RegexMasker()
    text, mapping = m.mask("Contact: ivan@example.com")
    assert "ivan@example.com" not in text
    assert "<email_1>" in text
    assert mapping["ivan@example.com"] == "<email_1>"


def test_masks_phone_ru():
    m = RegexMasker()
    text, mapping = m.mask("Тел: +7 916 234-56-78")
    assert "+7 916 234-56-78" not in text
    assert "<phone_1>" in text
    assert mapping["+7 916 234-56-78"] == "<phone_1>"


def test_masks_ip():
    m = RegexMasker()
    text, mapping = m.mask("Host: 192.168.1.10")
    assert "<ip_1>" in text
    assert mapping["192.168.1.10"] == "<ip_1>"


def test_masks_cidr():
    m = RegexMasker()
    text, mapping = m.mask("Net: 10.0.0.0/24")
    assert "<ip_1>" in text
    assert mapping["10.0.0.0/24"] == "<ip_1>"


def test_masks_url():
    m = RegexMasker()
    text, mapping = m.mask("API: https://api.example.com/v2/users")
    assert "https://api.example.com/v2/users" not in text
    assert "<url_1>" in text


def test_url_before_email_no_double_mask():
    """Email inside URL must NOT produce two separate placeholders."""
    m = RegexMasker()
    text, mapping = m.mask("DSN: postgresql://admin:pass@db.host.com:5432/mydb")
    # The full DSN is a URL, only one placeholder expected
    assert text.count("<url_") == 1 or text.count("<secret_") == 1


def test_masks_snils():
    m = RegexMasker()
    text, mapping = m.mask("СНИЛС: 045-678-912 00")
    assert "045-678-912 00" not in text
    assert "<doc_1>" in text
    assert mapping["045-678-912 00"] == "<doc_1>"


def test_masks_inn_with_context():
    m = RegexMasker()
    text, mapping = m.mask("ООО «Рога», ИНН 7814567234, КПП 781401001")
    assert "7814567234" not in text
    assert "781401001" not in text
    assert "<doc_1>" in text


def test_masks_ogrn_with_context():
    m = RegexMasker()
    text, mapping = m.mask("ОГРН 1167847312890")
    assert "1167847312890" not in text
    assert "<doc_1>" in text


def test_masks_bank_account():
    m = RegexMasker()
    text, mapping = m.mask("Р/с: 40702810600001234567")
    assert "40702810600001234567" not in text
    assert "<doc_1>" in text


def test_masks_jwt():
    m = RegexMasker()
    text, mapping = m.mask("Token: eyJhbGciOiJIUzI1NiJ9.payload.sig")
    assert "eyJhbGciOiJIUzI1NiJ9.payload.sig" not in text
    assert "<secret_1>" in text


def test_masks_stripe_key():
    m = RegexMasker()
    text, mapping = m.mask("STRIPE_KEY=sk_live_51H8xBz2eZvKYlo1N8q")
    assert "sk_live_51H8xBz2eZvKYlo1N8q" not in text
    assert "<secret_1>" in text


def test_masks_aws_key():
    m = RegexMasker()
    text, mapping = m.mask("AWS_KEY=AKIAIOSFODNN7EXAMPLE")
    assert "AKIAIOSFODNN7EXAMPLE" not in text
    assert "<secret_1>" in text


def test_same_value_same_placeholder():
    """Same entity always gets the same placeholder."""
    m = RegexMasker()
    text, mapping = m.mask("a@b.com and a@b.com again")
    assert text.count("<email_1>") == 2
    assert len(mapping) == 1


def test_different_values_different_placeholders():
    m = RegexMasker()
    text, mapping = m.mask("a@b.com and c@d.com")
    assert "<email_1>" in text
    assert "<email_2>" in text
    assert len(mapping) == 2


def test_counters_are_independent_per_type():
    m = RegexMasker()
    text, mapping = m.mask("a@b.com and 192.168.1.1")
    assert "<email_1>" in text
    assert "<ip_1>" in text


def test_existing_placeholder_not_double_masked():
    """Text that already contains <email_1> must not be re-masked."""
    m = RegexMasker()
    text, mapping = m.mask("Contact: <email_1>")
    # Should remain unchanged — not a real email address
    assert text == "Contact: <email_1>"
    assert not mapping


def test_mask_is_stateless_between_calls():
    """Each call to mask() starts fresh — counters reset."""
    m = RegexMasker()
    _, mapping1 = m.mask("a@b.com")
    _, mapping2 = m.mask("a@b.com")
    assert mapping1 == mapping2  # same result for same input
```

### Step 2: Run tests — confirm they fail

```bash
cd /Users/vladislavkodak/PycharmProjects/llm_mask
.venv/bin/python -m pytest tests/test_regex_masker.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'llm_mask._regex_masker'`

### Step 3: Implement `_regex_masker.py`

```python
# src/llm_mask/_regex_masker.py
"""Deterministic regex-based pre-masker for structured PII types.

Handles: URL, email, secret keys (JWT/Stripe/AWS/GitLab/SendGrid),
phone (RU), IP/CIDR, СНИЛС, ИНН/ОГРН/КПП (context-aware), bank
accounts, correspondent accounts.

Applied BEFORE the LLM so the model only processes fuzzy entities
(names, companies, host names, addresses).
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Regex patterns — order matters (most specific first)
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # 1. PostgreSQL/generic DSN — must come before URL/email to avoid splitting
    (re.compile(
        r'(?:postgresql|mysql|mongodb|redis)://[^\s,\)\]\}\'\"<>\n]+'
    ), "secret"),

    # 2. URL (http/https)
    (re.compile(
        r'https?://[^\s,\)\]\}\'\"<>\n]+'
    ), "url"),

    # 3. Email
    (re.compile(
        r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    ), "email"),

    # 4. JWT token
    (re.compile(
        r'eyJ[A-Za-z0-9+/=_\-]+\.[A-Za-z0-9+/=_\-]+(?:\.[A-Za-z0-9+/=_\-]+)?'
    ), "secret"),

    # 5. AWS access key
    (re.compile(r'AKIA[A-Z0-9]{16}'), "secret"),

    # 6. AWS secret (after '=' or ':')
    (re.compile(
        r'(?:AWS_SECRET(?:_ACCESS_KEY)?|aws_secret(?:_access_key)?)(?:\s*[=:]\s*)([^\s\n\]\[,;\'\"]{16,})'
    ), "secret"),

    # 7. Stripe key
    (re.compile(r'sk_(?:live|test)_[A-Za-z0-9]{16,}'), "secret"),

    # 8. GitLab token
    (re.compile(r'gl(?:pat|dt)-[A-Za-z0-9_\-]{10,}'), "secret"),

    # 9. SendGrid key
    (re.compile(r'SG\.[A-Za-z0-9_\-]{16,}\.[A-Za-z0-9_\-]{16,}'), "secret"),

    # 10. Russian phone: +7 or 8, then 10 digits with various separators
    (re.compile(
        r'(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}'
    ), "phone"),

    # 11. IP address with optional CIDR
    (re.compile(
        r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
        r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:/\d{1,2})?\b'
    ), "ip"),

    # 12. СНИЛС: XXX-XXX-XXX XX
    (re.compile(r'\b\d{3}-\d{3}-\d{3}\s\d{2}\b'), "doc"),

    # 13. ИНН (10 or 12 digits) after "ИНН" keyword
    (re.compile(r'(?<=ИНН\s)\d{10,12}|(?<=ИНН:\s)\d{10,12}'), "doc"),

    # 14. ОГРН (13 or 15 digits) after "ОГРН" keyword
    (re.compile(r'(?<=ОГРН\s)\d{13,15}|(?<=ОГРН:\s)\d{13,15}'), "doc"),

    # 15. КПП (9 digits) after "КПП" keyword
    (re.compile(r'(?<=КПП\s)\d{9}|(?<=КПП:\s)\d{9}'), "doc"),

    # 16. Russian bank settlement account (20 digits starting with 4)
    (re.compile(r'\b4\d{19}\b'), "doc"),

    # 17. Russian bank correspondent account (20 digits starting with 3)
    (re.compile(r'\b3\d{19}\b'), "doc"),
]

# Matches existing placeholders so we skip re-masking them
_EXISTING_PH_RE = re.compile(r'<[a-z_]+_\d+>')


def _format_ph(ph_type: str, n: int) -> str:
    return f"<{ph_type}_{n}>"


class RegexMasker:
    """Apply deterministic regex masking to text before the LLM call.

    Each call to :meth:`mask` is independent — state is reset.
    """

    def mask(self, text: str) -> tuple[str, dict[str, str]]:
        """Return *(masked_text, mapping)*.

        *mapping* has the same ``{original: placeholder}`` format as the
        LLM-produced mapping so both can be merged seamlessly.
        """
        registry: dict[str, str] = {}   # original → placeholder
        counters: dict[str, int] = {}   # ph_type → current count

        def replace(ph_type: str, m: re.Match) -> str:
            # Some patterns use a capture group (group 1) for the secret value
            try:
                original = m.group(1)
                span_start, span_end = m.start(1), m.end(1)
            except IndexError:
                original = m.group(0)
                span_start, span_end = m.start(), m.end()

            # Skip if it looks like an existing placeholder
            if _EXISTING_PH_RE.fullmatch(original.strip()):
                return m.group(0)

            if original in registry:
                # Re-use existing placeholder; patch text only where the
                # capture group differs from full match
                ph = registry[original]
                if m.group(0) == original:
                    return ph
                # Full match includes a prefix like "AWS_SECRET="; keep prefix
                return m.group(0)[: span_start - m.start()] + ph

            n = counters.get(ph_type, 0) + 1
            counters[ph_type] = n
            ph = _format_ph(ph_type, n)
            registry[original] = ph

            if m.group(0) == original:
                return ph
            # Keep prefix before the captured group
            prefix = m.group(0)[: span_start - m.start()]
            return prefix + ph

        for pattern, ph_type in _PATTERNS:
            text = pattern.sub(lambda m, t=ph_type: replace(t, m), text)

        return text, registry
```

### Step 4: Run tests — confirm they pass

```bash
.venv/bin/python -m pytest tests/test_regex_masker.py -v
```
Expected: all pass (fix any edge-case failures by adjusting regex, not test intent).

### Step 5: Expose in package `__init__.py` (internal — no public export needed)

Add import to `src/llm_mask/__init__.py` only if needed for external access. For now it stays internal.

### Step 6: Commit

```bash
git add src/llm_mask/_regex_masker.py tests/test_regex_masker.py
git commit -m "feat: add RegexMasker for deterministic structured PII pre-masking"
```

---

## Task 2: `_repair.py` — DUP-PH detection and fix

**Files:**
- Create: `src/llm_mask/_repair.py`
- Create: `tests/test_repair.py`

### Step 1: Write failing tests

```python
# tests/test_repair.py
import pytest
from llm_mask._repair import repair_dup_ph


def test_noop_when_no_dup():
    mapping = {"Ivan": "<person_1>", "Petrov": "<person_2>"}
    text = "Hello <person_1> <person_2>"
    result_text, result_map = repair_dup_ph(text, mapping)
    assert result_text == text
    assert result_map == mapping


def test_detects_dup_ph_two_originals():
    """<person_1> assigned to both 'Ivan' and 'Anna' — second gets new ph."""
    mapping = {"Ivan": "<person_1>", "Anna": "<person_1>"}
    text = "<person_1> met <person_1>"
    result_text, result_map = repair_dup_ph(text, mapping)
    # Two distinct placeholders now
    assert result_map["Ivan"] != result_map["Anna"]
    assert len(set(result_map.values())) == 2


def test_second_occurrence_patched_in_text():
    """The second occurrence of the DUP-PH in text is replaced with new ph."""
    mapping = {"Ivan": "<person_1>", "Anna": "<person_1>"}
    # Ivan appears first in document, Anna second
    text = "Name: <person_1>. Another: <person_1>."
    result_text, result_map = repair_dup_ph(text, mapping)
    new_ph = result_map["Anna"]
    assert new_ph != "<person_1>"
    assert new_ph in result_text
    assert result_text.count("<person_1>") == 1
    assert result_text.count(new_ph) == 1


def test_new_placeholder_number_does_not_collide():
    """Newly allocated placeholder must not collide with existing ones."""
    mapping = {
        "Ivan": "<person_1>",
        "Anna": "<person_1>",   # DUP
        "Boris": "<person_2>",  # already used
    }
    text = "<person_1> <person_1> <person_2>"
    _, result_map = repair_dup_ph(text, mapping)
    # Anna must get person_3 (not person_2, already taken)
    assert result_map["Anna"] == "<person_3>"


def test_three_originals_for_one_placeholder():
    mapping = {"A": "<x_1>", "B": "<x_1>", "C": "<x_1>"}
    text = "<x_1> <x_1> <x_1>"
    result_text, result_map = repair_dup_ph(text, mapping)
    # Three distinct placeholders
    assert len(set(result_map.values())) == 3
    # Each appears exactly once in text
    for ph in result_map.values():
        assert result_text.count(ph) == 1


def test_mapping_with_no_angle_brackets_bare_form():
    """Bare placeholders (project_1, service_1) also handled."""
    mapping = {"Phoenix": "project_1", "Atlas": "project_1"}
    text = "project_1 and project_1"
    result_text, result_map = repair_dup_ph(text, mapping)
    assert result_map["Phoenix"] != result_map["Atlas"]
```

### Step 2: Run tests — confirm they fail

```bash
.venv/bin/python -m pytest tests/test_repair.py -v 2>&1 | head -10
```

### Step 3: Implement `_repair.py`

```python
# src/llm_mask/_repair.py
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
```

### Step 4: Run tests

```bash
.venv/bin/python -m pytest tests/test_repair.py -v
```

### Step 5: Commit

```bash
git add src/llm_mask/_repair.py tests/test_repair.py
git commit -m "feat: add repair_dup_ph to fix placeholder collision after LLM"
```

---

## Task 3: `unmasker.py` — Prefix-dedup restoration fix

**Problem:** LLM writes `"д. 145" → <building_1>` in the mapping but replaces only `145` in the text, leaving `д.` in place. Naive `str.replace("<building_1>", "д. 145")` produces `"д. д. 145"`.

**Fix:** Before replacing placeholder with original, check if the original starts with a word that immediately precedes the placeholder in text. If yes, substitute the whole `"prefix placeholder"` span with the full original (no duplication).

**Files:**
- Modify: `src/llm_mask/unmasker.py`
- Modify: `tests/test_unmasker.py`

### Step 1: Write failing tests

Add to `tests/test_unmasker.py`:

```python
# --- Prefix-dedup tests ---

def test_no_prefix_dup_address():
    """'д. <building_1>' with mapping 'д. 145' → '<building_1>' restores cleanly."""
    from llm_mask.unmasker import Unmasker
    u = Unmasker()
    masked = "ул. Ленина, д. <building_1>, кв. <building_2>"
    mapping = {"д. 145": "<building_1>", "кв. 32": "<building_2>"}
    result = u.unmask(masked, mapping)
    assert result == "ул. Ленина, д. 145, кв. 32"
    assert "д. д." not in result
    assert "кв. кв." not in result


def test_no_prefix_dup_zipcode():
    """'индекс <zipcode_1>' with mapping 'индекс 620014' restores cleanly."""
    from llm_mask.unmasker import Unmasker
    u = Unmasker()
    masked = "г. Москва, индекс <zipcode_1>"
    mapping = {"индекс 620014": "<zipcode_1>"}
    result = u.unmask(masked, mapping)
    assert result == "г. Москва, индекс 620014"
    assert "индекс индекс" not in result


def test_bare_placeholder_no_prefix_dup():
    """Works for bare (no angle bracket) placeholders too."""
    from llm_mask.unmasker import Unmasker
    u = Unmasker()
    masked = "Проект: project_1 запущен"
    mapping = {"МойПроект": "project_1"}
    result = u.unmask(masked, mapping)
    assert result == "Проект: МойПроект запущен"


def test_no_prefix_when_standalone():
    """Placeholder not preceded by its original's first word — full original used."""
    from llm_mask.unmasker import Unmasker
    u = Unmasker()
    masked = "Индекс: <zipcode_1>"
    mapping = {"индекс 620014": "<zipcode_1>"}
    # "Индекс:" is not "индекс" (case differs) — full original inserted
    result = u.unmask(masked, mapping)
    assert "620014" in result
```

### Step 2: Run — confirm they fail

```bash
.venv/bin/python -m pytest tests/test_unmasker.py -v -k "prefix_dup"
```

### Step 3: Implement fix in `unmasker.py`

Replace the `unmask` method body:

```python
import re

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
```

Keep the existing `from pathlib import Path` and `from .mapping import MaskingResult` imports.

### Step 4: Run tests

```bash
.venv/bin/python -m pytest tests/test_unmasker.py -v
```

### Step 5: Commit

```bash
git add src/llm_mask/unmasker.py tests/test_unmasker.py
git commit -m "fix: eliminate prefix duplication in Unmasker restoration (д. д. 145)"
```

---

## Task 4: `_merger.py` — Accept pre-loaded regex mapping

**Purpose:** Allow ChunkMerger to be initialized with the regex mapping so it pre-populates its registry and counters. This prevents counter collisions between regex and LLM placeholders.

**Files:**
- Modify: `src/llm_mask/_merger.py`
- Modify: `tests/test_client.py` (or add a small merger test)

### Step 1: Modify `ChunkMerger.__init__`

```python
import re as _re

class ChunkMerger:
    def __init__(self, preloaded: dict[str, str] | None = None) -> None:
        self._global_registry: dict[str, str] = {}
        self._counters: dict[str, int] = {}
        if preloaded:
            self._global_registry.update(preloaded)
            for ph in preloaded.values():
                ph_type = _extract_type(ph)
                m = _re.search(r"\d+", ph)
                if m:
                    n = int(m.group())
                    if n > self._counters.get(ph_type, 0):
                        self._counters[ph_type] = n
```

### Step 2: Verify existing tests still pass

```bash
.venv/bin/python -m pytest tests/ -v -x
```

### Step 3: Commit

```bash
git add src/llm_mask/_merger.py
git commit -m "feat: ChunkMerger accepts pre-loaded regex mapping to avoid counter collisions"
```

---

## Task 5: `client.py` — Wire up the full pipeline

**Files:**
- Modify: `src/llm_mask/client.py`

### Step 1: Update imports and `__init__`

```python
# Add to imports:
from ._regex_masker import RegexMasker
from ._repair import repair_dup_ph
```

In `MaskingClient.__init__`, add:
```python
self._regex_masker = RegexMasker()
```

### Step 2: Replace `mask()` method body

```python
def mask(self, text: str) -> MaskingResult:
    # ── Stage 1: deterministic regex pre-masking ──────────────────────
    pre_masked, regex_mapping = self._regex_masker.mask(text)

    # ── Stage 2: LLM masking of fuzzy entities ────────────────────────
    chunks = split_into_chunks(pre_masked, self._chunk_size)
    merger = ChunkMerger(preloaded=regex_mapping)
    masked_parts: list[str] = []

    for chunk in chunks:
        raw = self._llm.complete(self._system_prompt, chunk)
        masked_chunk, chunk_mapping = parse_llm_response(raw)
        chunk_mapping = recover_mapping(chunk, masked_chunk, chunk_mapping)
        patched, _ = merger.add_chunk(masked_chunk, chunk_mapping)
        masked_parts.append(patched)

    masked_text = "".join(masked_parts)

    # ── Stage 3: merge regex + LLM mappings ──────────────────────────
    # regex_mapping already in merger registry; global_mapping() returns LLM part.
    # Merge: regex_mapping takes precedence for shared keys (shouldn't overlap).
    combined_mapping = {**merger.global_mapping(), **regex_mapping}

    # ── Stage 4: post-processing repairs ─────────────────────────────
    masked_text, combined_mapping = repair_dup_ph(masked_text, combined_mapping)
    masked_text = repair_arrow_collisions(masked_text, text, combined_mapping)

    # ── Stage 5: optional judge ───────────────────────────────────────
    judge_iterations = 0
    remaining_entities: list[str] = []
    if self._judge is not None:
        masked_text, judge_iterations, remaining_entities = self._judge.review(
            masked_text=masked_text,
            merger=merger,
            masking_llm=self._llm,
            masking_prompt=self._system_prompt,
        )

    return MaskingResult(
        masked_text=masked_text,
        mapping=combined_mapping,
        chunks_processed=len(chunks),
        judge_iterations=judge_iterations,
        remaining_entities=remaining_entities,
    )
```

### Step 3: Run all tests

```bash
.venv/bin/python -m pytest tests/ -v
```

### Step 4: Commit

```bash
git add src/llm_mask/client.py
git commit -m "feat: integrate RegexMasker and repair_dup_ph into MaskingClient pipeline"
```

---

## Task 6: `_parser.py` — DUP-PH guard in `_parse_mapping_lines`

**Purpose:** When the LLM assigns the same placeholder to two different originals in a single response, detect it at parse time and skip the second assignment (so the mapping stays unambiguous before `repair_dup_ph` runs).

**Files:**
- Modify: `src/llm_mask/_parser.py`
- Modify: `tests/test_parser.py`

### Step 1: Add test

```python
# In tests/test_parser.py

def test_parse_mapping_deduplicates_dup_ph():
    """When LLM assigns same placeholder to two originals, second is skipped."""
    from llm_mask._parser import _parse_mapping_lines
    lines = "Ivan -> <person_1>\nAnna -> <person_1>\nBoris -> <person_2>"
    result = _parse_mapping_lines(lines)
    # Only one original maps to <person_1>
    ph_counts = {}
    for v in result.values():
        ph_counts[v] = ph_counts.get(v, 0) + 1
    assert ph_counts.get("<person_1>", 0) == 1
    # Boris is unaffected
    assert result["Boris"] == "<person_2>"
```

### Step 2: Run — confirm it fails

```bash
.venv/bin/python -m pytest tests/test_parser.py::test_parse_mapping_deduplicates_dup_ph -v
```

### Step 3: Add DUP-PH guard to `_parse_mapping_lines`

In the existing `_parse_mapping_lines` function, add a `ph_seen` dict and skip duplicates:

```python
def _parse_mapping_lines(mapping_raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    ph_seen: dict[str, str] = {}   # placeholder → first original that claimed it

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
        original = original.rstrip(",;")
        if original and not _PLACEHOLDER_RE.match(original):
            # DUP-PH guard: if this placeholder was already claimed by a
            # different original, skip the line (repair_dup_ph handles it later)
            if placeholder in ph_seen and ph_seen[placeholder] != original:
                continue
            ph_seen[placeholder] = original
            result[original] = placeholder
    return result
```

### Step 4: Run all parser tests

```bash
.venv/bin/python -m pytest tests/test_parser.py -v
```

### Step 5: Commit

```bash
git add src/llm_mask/_parser.py tests/test_parser.py
git commit -m "fix: skip duplicate placeholder assignments in _parse_mapping_lines"
```

---

## Task 7: Update RU prompt — LLM handles only fuzzy types

**Purpose:** The LLM now receives text where emails/phones/IPs/URLs/secrets/bank-accounts are already masked. Update the prompt to reflect this and reduce its cognitive load.

**Files:**
- Modify: `src/llm_mask/prompts/ru/mask_prompt.txt`

### Step 1: Replace prompt content

```text
Ты — система анонимизации. В тексте уже заменены структурированные данные
(email, телефоны, IP, URL, секреты, ИНН, ОГРН, КПП, счета, СНИЛС).
Твоя задача — найти и заменить оставшиеся чувствительные сущности:

ТИПЫ И ПЛЕЙСХОЛДЕРЫ (N — целое число, 1, 2, 3, …):
- Имена людей (каждая часть ФИО отдельно): <person_1>, <person_2>, ...
- Компании / бренды / организации: <company_1>, <company_2>, ...
- Инфраструктура (хосты, кластеры, окружения): <host_1>, <env_1>, ...
- Названия сервисов / микросервисов: <service_1>, <service_2>, ...
- Внутренние проекты / кодовые имена: <project_1>, <project_2>, ...
- Город / населённый пункт (с префиксом г., д., п.): <city_1>, ...
- Улица (с типом ул., пр., наб.): <street_1>, ...
- Дом / квартира (с префиксом д., кв., корп.): <building_1>, ...
- Почтовый индекс: <zipcode_1>, ...
- Даты рождения и личные даты: <date_1>, ...
- Документы (паспорт серия/номер, если не заменены): <doc_1>, ...

ПРАВИЛА:
1. Каждое уникальное значение → свой уникальный номер. Нельзя один плейсхолдер для двух разных значений.
2. Одно и то же значение → всегда один и тот же плейсхолдер.
3. Нумерация сквозная, по порядку первого появления сверху вниз.
4. НЕ трогать уже существующие плейсхолдеры в тексте (например, <email_1>).
5. Имена: каждая часть ФИО → отдельный <person_N>. Каждая часть — отдельная строка маппинга.
6. Ключ маппинга — ТОЛЬКО заменённое значение, без предшествующих слов-префиксов. Если в тексте «д. 145» и заменяется «145», ключ = «145», не «д. 145».
7. Запятая после имени НЕ входит в ключ маппинга.
8. Стрелка -> в коде/firewall-правилах — оператор направления, НЕ маппинг. Оба хоста получают разные плейсхолдеры.
9. Каждый плейсхолдер из текста ОБЯЗАН быть в маппинге.
10. Маппинг НЕ должен содержать строки вида <ph> -> <ph>.
11. Не заменять: структуру markdown, имена полей JSON, публичные библиотеки.

ФОРМАТ ОТВЕТА:

Сначала полный анонимизированный markdown-документ.

Затем строго:

Mapping замен
точный_оригинал -> <placeholder_N>
точный_оригинал -> <placeholder_N>
...

Правила маппинга:
- Левая часть — ТОЧНЫЙ текст из документа, без изменений регистра, без кавычек.
- Только реальные замены — никаких примеров и шаблонов.
- Не добавлять комментарии и объяснения.
```

### Step 2: Verify no unit tests break

```bash
.venv/bin/python -m pytest tests/ -v
```

### Step 3: Commit

```bash
git add src/llm_mask/prompts/ru/mask_prompt.txt
git commit -m "feat: update RU prompt — LLM handles fuzzy types only, regex covers structured"
```

---

## Task 8: Integration smoke test

**Purpose:** Run the examples script against the real LLM and verify all 6 files produce clean mappings (no DUP-PH, no orphans, no prefix duplication in restored text).

### Step 1: Run examples (requires LLM server on :8001)

```bash
.venv/bin/python examples/run_masking.py \
  --model "Qwen3-30B-A3B-Instruct-2507-IQ4_XS-3.63bpw.gguf" \
  --max-tokens 8000
```

### Step 2: Check for known artifacts

```bash
# No "д. д." or "индекс индекс" in restored files
grep -r "д\. д\.\|индекс индекс\|кв\. кв\." examples/documents/*_restored.md \
  && echo "FAIL: prefix duplication found" || echo "OK: no prefix duplication"

# No orphan placeholders in masked files
python3 -c "
import re, sys, pathlib
ph = re.compile(r'<[a-z_]+_\d+>')
any_fail = False
for f in pathlib.Path('examples/documents').glob('*_masked.md'):
    text = f.read_text()
    # Quick scan for placeholder-like text that might be orphaned
    phs = set(ph.findall(text))
    if phs:
        import json
        map_file = f.parent / (f.stem.replace('_masked','_mapping') + '.json')
        if map_file.exists():
            data = json.loads(map_file.read_text())
            mapping_vals = set((data.get('mapping') or data).values())
            orphans = phs - mapping_vals
            if orphans:
                print(f'ORPHAN in {f.name}: {orphans}')
                any_fail = True
if not any_fail:
    print('OK: no orphan placeholders')
"
```

### Step 3: Final test suite run

```bash
.venv/bin/python -m pytest tests/ -v --tb=short
```

Expected: all tests pass.

### Step 4: Tag the release

```bash
git tag v0.2.0
```

---

## Summary of changes

| File | Change |
|------|--------|
| `src/llm_mask/_regex_masker.py` | **New** — deterministic pre-masker |
| `src/llm_mask/_repair.py` | **New** — DUP-PH repair |
| `src/llm_mask/unmasker.py` | Fix prefix duplication in restoration |
| `src/llm_mask/_merger.py` | Accept preloaded mapping in `__init__` |
| `src/llm_mask/client.py` | Wire RegexMasker + repair into pipeline |
| `src/llm_mask/_parser.py` | DUP-PH guard in `_parse_mapping_lines` |
| `src/llm_mask/prompts/ru/mask_prompt.txt` | Focus on fuzzy types only |
| `tests/test_regex_masker.py` | **New** |
| `tests/test_repair.py` | **New** |
| `tests/test_unmasker.py` | Extend with prefix-dedup tests |
| `tests/test_parser.py` | Extend with DUP-PH guard test |
