import pytest
from llm_mask._parser import parse_llm_response, recover_mapping, _parse_mapping_lines
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
