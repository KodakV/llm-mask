import pytest
from llm_mask._parser import (
    parse_llm_response,
    recover_mapping,
    repair_arrow_collisions,
    _parse_mapping_lines,
)
from llm_mask.exceptions import ParsingError


SAMPLE_RU = """\
# Документ

Контакт: <email_1>, тел. <phone_1>

Mapping замен
ivan@company.com -> <email_1>
+7 999 123-45-67 -> <phone_1>
"""

SAMPLE_EN = """\
# Document

Contact: <email_1>, phone <phone_1>

Mapping
ivan@company.com -> <email_1>
+1 555 123-4567 -> <phone_1>
"""

NO_SEPARATOR = "Sentence with an arrow x -> y but no valid placeholder on either side."

EXTRA_WHITESPACE = """\
Masked text here.

Mapping замен

ivan@company.com -> <email_1>
  Apple  ->  <company_1>
"""


def test_parse_ru_separator():
    masked, mapping = parse_llm_response(SAMPLE_RU)
    assert "Mapping" not in masked
    assert mapping["ivan@company.com"] == "<email_1>"
    assert mapping["+7 999 123-45-67"] == "<phone_1>"


def test_parse_en_separator():
    masked, mapping = parse_llm_response(SAMPLE_EN)
    assert mapping["ivan@company.com"] == "<email_1>"
    assert mapping["+1 555 123-4567"] == "<phone_1>"


def test_masked_text_stripped():
    masked, _ = parse_llm_response(SAMPLE_RU)
    assert not masked.startswith("\n")
    assert not masked.endswith("\n\n")


def test_no_separator_raises():
    with pytest.raises(ParsingError):
        parse_llm_response(NO_SEPARATOR)


def test_no_arrow_returns_text_as_clean():
    """LLM response with no '->' at all is treated as clean masked text."""
    raw = "# Clean paragraph\n\nNo sensitive data here."
    masked, mapping = parse_llm_response(raw)
    assert masked == raw.strip()
    assert mapping == {}


def test_extra_whitespace_in_mapping():
    _, mapping = parse_llm_response(EXTRA_WHITESPACE)
    # Keys / values should be stripped
    assert "Apple" in mapping
    assert mapping["Apple"] == "<company_1>"


def test_parse_mapping_lines_skips_empty():
    lines = "\n\nApple -> <company_1>\n\n"
    result = _parse_mapping_lines(lines)
    assert result == {"Apple": "<company_1>"}


def test_parse_mapping_lines_skips_no_arrow():
    lines = "Apple <company_1>\nGoogle -> <company_2>"
    result = _parse_mapping_lines(lines)
    assert "Apple" not in result
    assert result["Google"] == "<company_2>"


REVERSED_FORMAT = """\
# Встреча команды

Организатор: <person_1> (<email_1>)

---

<person_1> -> Иван Петров
<email_1> -> ivan@company.com
"""


def test_parse_reversed_mapping_direction():
    """LLM sometimes outputs placeholder -> original (reversed); parser must normalise."""
    masked, mapping = parse_llm_response(REVERSED_FORMAT)
    assert mapping["Иван Петров"] == "<person_1>"
    assert mapping["ivan@company.com"] == "<email_1>"
    assert "---" not in masked


def test_parse_dash_separator_fallback():
    """--- as separator is handled via fallback paragraph detection."""
    raw = "Masked text.\n\n---\n\nApple -> <company_1>\nIvan -> <person_1>"
    _, mapping = parse_llm_response(raw)
    assert mapping["Apple"] == "<company_1>"
    assert mapping["Ivan"] == "<person_1>"


def test_recover_mapping_finds_missing_entry():
    """recover_mapping fills in placeholders the LLM forgot to list."""
    original = "Project: Fortune Tavern (https://api.example.com)"
    masked = "Project: project_1 (<url_1>)"
    partial = {"https://api.example.com": "<url_1>"}
    result = recover_mapping(original, masked, partial)
    assert result["Fortune Tavern"] == "project_1"
    assert result["https://api.example.com"] == "<url_1>"


def test_recover_mapping_noop_when_complete():
    """recover_mapping returns the same dict when nothing is missing."""
    original = "Hello Ivan"
    masked = "Hello <person_1>"
    mapping = {"Ivan": "<person_1>"}
    result = recover_mapping(original, masked, mapping)
    assert result == mapping


def test_parse_mapping_skips_placeholder_as_original():
    """Entries where original side is already a placeholder are silently dropped."""
    lines = "<url_1> -> <url_1>\nApple -> <company_1>"
    result = _parse_mapping_lines(lines)
    assert "<url_1>" not in result
    assert result["Apple"] == "<company_1>"


def test_parse_mapping_strips_markdown_from_original():
    """Markdown decoration is stripped from entity keys."""
    lines = "**Космокафе** -> project_1"
    result = _parse_mapping_lines(lines)
    assert "Космокафе" in result
    assert "**Космокафе**" not in result


def test_strip_markdown_preserves_trailing_angle_bracket():
    """Trailing '>' must NOT be stripped — it closes placeholder angle brackets.

    Regression for: <company_1> appearing as key in mapping when the model
    emits a self-referential entry like '<company_1> -> <company_1>'.  The
    _strip_markdown call was stripping the trailing '>' and producing the
    broken key '<company_1'.
    """
    from llm_mask._parser import _strip_markdown
    assert _strip_markdown("<company_1>") == "<company_1>"
    assert _strip_markdown("<secret_pass_1>") == "<secret_pass_1>"


def test_parse_mapping_self_ref_placeholder_dropped():
    """Self-referential mapping lines like '<ph> -> <ph>' are dropped silently.

    When the model outputs '<company_1> -> <company_1>' the entry should be
    skipped, not stored with a broken key like '<company_1'.
    """
    lines = "<company_1> -> <company_1>\nApple -> <company_2>"
    result = _parse_mapping_lines(lines)
    assert "<company_1>" not in result
    assert "<company_1" not in result  # broken key must not appear
    assert result["Apple"] == "<company_2>"


def test_parse_mapping_original_ending_angle_bracket_preserved():
    """Original values that end with '>' (e.g. HTML tags) keep their '>'."""
    from llm_mask._parser import _strip_markdown
    # A template variable in the original document
    assert _strip_markdown("<var>") == "<var>"
    # Leading blockquote marker is stripped but trailing '>' is not
    assert _strip_markdown("> blockquote") == "blockquote"


def test_parse_mapping_strips_trailing_comma_from_key():
    """Trailing comma/semicolon leaked from sentence context is stripped from key."""
    lines = "Белоусова Наталья Викторовна, -> <person_3>\nApple; -> <company_1>"
    result = _parse_mapping_lines(lines)
    assert "Белоусова Наталья Викторовна" in result
    assert "Белоусова Наталья Викторовна," not in result
    assert "Apple" in result
    assert "Apple;" not in result


def test_strip_self_ref_mapping_suffix_removes_fake_block():
    """Self-referential mapping appended by the model is removed from masked text."""
    from llm_mask._parser import _strip_self_ref_mapping_suffix
    text = (
        "# Doc\n\nHello <person_1>\n\n"
        "<url_1> -> <url_1>\n"
        "<email_1> -> <email_1>\n"
        "service_1 -> service_1"
    )
    result = _strip_self_ref_mapping_suffix(text)
    assert "<url_1> -> <url_1>" not in result
    assert "Hello <person_1>" in result


def test_parse_response_with_self_ref_suffix_stripped():
    """parse_llm_response removes the fake self-ref block before returning text."""
    raw = (
        "# Doc\n\nHello <person_1>\n\n"
        "Mapping замен\n"
        "Ivan -> <person_1>\n\n"
        "<url_1> -> <url_1>\n"
        "<email_1> -> <email_1>"
    )
    masked, mapping = parse_llm_response(raw)
    assert "<url_1> -> <url_1>" not in masked
    assert mapping["Ivan"] == "<person_1>"


def test_repair_arrow_collisions_fixes_firewall_rule():
    """repair_arrow_collisions corrects <ph> -> <ph> from firewall-style -> rules."""
    original = "ALLOW 192.168.1.1 -> 10.0.0.5:5432  # DB access"
    masked   = "ALLOW <ip_3> -> <ip_3>:5432  # DB access"
    mapping  = {"192.168.1.1": "<ip_3>", "10.0.0.5": "<ip_5>"}
    result = repair_arrow_collisions(masked, original, mapping)
    assert "<ip_3> -> <ip_5>" in result
    assert "<ip_3> -> <ip_3>" not in result


def test_repair_arrow_collisions_noop_when_clean():
    """repair_arrow_collisions leaves clean text unchanged."""
    original = "ALLOW 192.168.1.1 -> 10.0.0.5:5432"
    masked   = "ALLOW <ip_3> -> <ip_5>:5432"
    mapping  = {"192.168.1.1": "<ip_3>", "10.0.0.5": "<ip_5>"}
    result = repair_arrow_collisions(masked, original, mapping)
    assert result == masked


def test_recover_mapping_skips_secret_like_spans():
    """recover_mapping does not match JWT tokens or long keys as person names."""
    original = "Token: eyJhbGciOiJIUzI1NiJ9.payload Ivan"
    masked = "Token: <secret_1> <person_1>"
    mapping = {"eyJhbGciOiJIUzI1NiJ9.payload": "<secret_1>"}
    result = recover_mapping(original, masked, mapping)
    # Ivan should be recovered, JWT payload should NOT be mapped to <person_1>
    assert result.get("Ivan") == "<person_1>"
    # The JWT span must not appear mapped to <person_1>
    for k, v in result.items():
        if v == "<person_1>":
            assert "eyJ" not in k


def test_parse_mapping_deduplicates_dup_ph():
    """When LLM assigns same placeholder to two originals, second is skipped."""
    lines = "Ivan -> <person_1>\nAnna -> <person_1>\nBoris -> <person_2>"
    result = _parse_mapping_lines(lines)
    ph_counts: dict[str, int] = {}
    for v in result.values():
        ph_counts[v] = ph_counts.get(v, 0) + 1
    assert ph_counts.get("<person_1>", 0) == 1
    assert result["Boris"] == "<person_2>"


def test_parse_mapping_skips_key_with_placeholder():
    """Keys containing existing <ph_N> tokens must be dropped (artefact entries)."""
    lines = "ООО «Альфа», ОГРН <doc_3>, ИНН <doc_1> -> <doc_11>\nИван -> <person_1>"
    result = _parse_mapping_lines(lines)
    # The nested entry should be skipped
    assert "<doc_11>" not in result.values()
    assert result.get("Иван") == "<person_1>"


def test_parse_mapping_skips_arrow_in_original():
    """Keys containing '->' (firewall rules) must be skipped."""
    lines = "10.0.0.1 -> 10.0.0.2:5432 -> <rule_1>\nИван -> <person_1>"
    result = _parse_mapping_lines(lines)
    assert "<rule_1>" not in result.values()
    assert result.get("Иван") == "<person_1>"
