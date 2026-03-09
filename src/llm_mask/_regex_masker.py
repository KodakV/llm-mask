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

    # 10. СНИЛС: XXX-XXX-XXX XX (before phone to avoid digit confusion)
    (re.compile(r'\b\d{3}-\d{3}-\d{3}\s\d{2}\b'), "doc"),

    # 11. ИНН (10 or 12 digits) after "ИНН" keyword
    (re.compile(r'(?<=ИНН\s)\d{10,12}|(?<=ИНН:\s)\d{10,12}'), "doc"),

    # 12. ОГРН (13 or 15 digits) after "ОГРН" keyword
    (re.compile(r'(?<=ОГРН\s)\d{13,15}|(?<=ОГРН:\s)\d{13,15}'), "doc"),

    # 13. КПП (9 digits) after "КПП" keyword
    (re.compile(r'(?<=КПП\s)\d{9}|(?<=КПП:\s)\d{9}'), "doc"),

    # 14. Russian bank settlement account (20 digits starting with 4)
    #     Must come before phone to prevent phone regex matching substring
    (re.compile(r'\b4\d{19}\b'), "doc"),

    # 15. Russian bank correspondent account (20 digits starting with 3)
    (re.compile(r'\b3\d{19}\b'), "doc"),

    # 16. Russian phone: +7 or 8, then 10 digits with various separators
    #     Require word boundary or non-digit before to avoid matching inside long numbers
    (re.compile(
        r'(?<!\d)(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)'
    ), "phone"),

    # 17. IP address with optional CIDR
    (re.compile(
        r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
        r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)(?:/\d{1,2})?\b'
    ), "ip"),
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

            # If group 1 doesn't exist (no capturing group), fall back to full match
            if original is None:
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
