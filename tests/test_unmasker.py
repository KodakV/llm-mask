import json

from llm_mask.unmasker import Unmasker


def test_basic_unmask():
    unmasker = Unmasker()
    mapping = {"Ivan": "<person_1>", "Apple": "<company_1>"}
    masked = "Hello, <person_1>! Welcome to <company_1>."
    result = unmasker.unmask(masked, mapping)
    assert result == "Hello, Ivan! Welcome to Apple."


def test_unmask_longer_before_shorter():
    """<url_10> must not be partially replaced by <url_1> match."""
    unmasker = Unmasker()
    mapping = {
        "https://short.example.com": "<url_1>",
        "https://long.example.com/path": "<url_10>",
    }
    masked = "Visit <url_10> and also <url_1>."
    result = unmasker.unmask(masked, mapping)
    assert "https://long.example.com/path" in result
    assert "https://short.example.com" in result


def test_unmask_no_placeholders():
    unmasker = Unmasker()
    text = "No placeholders here."
    result = unmasker.unmask(text, {"Ivan": "<person_1>"})
    assert result == text


def test_unmask_file(tmp_path):
    unmasker = Unmasker()
    masked_text = "Hello, <person_1>!"
    mapping_data = {
        "source_file": "test.md",
        "masked_at": "2026-01-01T00:00:00Z",
        "mapping": {"Ivan": "<person_1>"},
    }
    masked_file = tmp_path / "test_masked.md"
    mapping_file = tmp_path / "test_mapping.json"
    masked_file.write_text(masked_text, encoding="utf-8")
    mapping_file.write_text(json.dumps(mapping_data), encoding="utf-8")

    result = unmasker.unmask_file(masked_file, mapping_file)
    assert result == "Hello, Ivan!"


def test_empty_mapping():
    unmasker = Unmasker()
    text = "Unchanged text."
    assert unmasker.unmask(text, {}) == text


# --- Prefix-dedup tests ---

def test_no_prefix_dup_address():
    """'д. <building_1>' with mapping 'д. 145' → '<building_1>' restores cleanly."""
    u = Unmasker()
    masked = "ул. Ленина, д. <building_1>, кв. <building_2>"
    mapping = {"д. 145": "<building_1>", "кв. 32": "<building_2>"}
    result = u.unmask(masked, mapping)
    assert result == "ул. Ленина, д. 145, кв. 32"
    assert "д. д." not in result
    assert "кв. кв." not in result


def test_no_prefix_dup_zipcode():
    """'индекс <zipcode_1>' with mapping 'индекс 620014' restores cleanly."""
    u = Unmasker()
    masked = "г. Москва, индекс <zipcode_1>"
    mapping = {"индекс 620014": "<zipcode_1>"}
    result = u.unmask(masked, mapping)
    assert result == "г. Москва, индекс 620014"
    assert "индекс индекс" not in result


def test_bare_placeholder_no_prefix_dup():
    """Works for bare (no angle bracket) placeholders too."""
    u = Unmasker()
    masked = "Проект: project_1 запущен"
    mapping = {"МойПроект": "project_1"}
    result = u.unmask(masked, mapping)
    assert result == "Проект: МойПроект запущен"


def test_no_prefix_when_standalone():
    """Placeholder not preceded by its original's first word — full original used."""
    u = Unmasker()
    masked = "Индекс: <zipcode_1>"
    mapping = {"индекс 620014": "<zipcode_1>"}
    # "Индекс:" is not "индекс" (case differs) — full original inserted
    result = u.unmask(masked, mapping)
    assert "620014" in result
