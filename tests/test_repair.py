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
    assert result_map["Anna"] == "<person_3>"


def test_three_originals_for_one_placeholder():
    mapping = {"A": "<x_1>", "B": "<x_1>", "C": "<x_1>"}
    text = "<x_1> <x_1> <x_1>"
    result_text, result_map = repair_dup_ph(text, mapping)
    assert len(set(result_map.values())) == 3
    for ph in result_map.values():
        assert result_text.count(ph) == 1


def test_mapping_with_no_angle_brackets_bare_form():
    """Bare placeholders (project_1, service_1) also handled."""
    mapping = {"Phoenix": "project_1", "Atlas": "project_1"}
    text = "project_1 and project_1"
    result_text, result_map = repair_dup_ph(text, mapping)
    assert result_map["Phoenix"] != result_map["Atlas"]
