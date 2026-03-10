"""Tests for MaskingJudge and helpers — no real LLM required."""
from unittest.mock import MagicMock, patch


from llm_mask._judge import (
    MaskingJudge,
    _dirty_paragraph_indices,
    _filter_placeholders,
    _fix_double_brackets,
    _parse_entity_list,
)
from llm_mask._merger import ChunkMerger


def test_parse_clean_json():
    assert _parse_entity_list('["Ivan", "Apple"]') == ["Ivan", "Apple"]


def test_parse_empty_array():
    assert _parse_entity_list("[]") == []


def test_parse_with_markdown_fence():
    assert _parse_entity_list('```json\n["Ivan"]\n```') == ["Ivan"]


def test_parse_json_embedded_in_prose():
    raw = 'Found these entities: ["Ivan", "Apple"]. Please review.'
    assert _parse_entity_list(raw) == ["Ivan", "Apple"]


def test_parse_invalid_returns_empty():
    assert _parse_entity_list("everything looks fine, nothing found") == []


def test_dirty_first_paragraph():
    text = "Hello Ivan.\n\nNothing here."
    assert _dirty_paragraph_indices(text, ["Ivan"]) == {0}


def test_dirty_second_paragraph():
    text = "Clean.\n\nHello Ivan."
    assert _dirty_paragraph_indices(text, ["Ivan"]) == {1}


def test_dirty_multiple_paragraphs():
    text = "Ivan here.\n\nClean.\n\nApple there."
    assert _dirty_paragraph_indices(text, ["Ivan", "Apple"]) == {0, 2}


def test_dirty_entity_not_in_text():
    text = "Nothing here.\n\nStill nothing."
    assert _dirty_paragraph_indices(text, ["Ivan"]) == set()


def test_dirty_no_entities():
    assert _dirty_paragraph_indices("Any text.", []) == set()


def test_filter_placeholders_removes_angle():
    assert _filter_placeholders(["Ivan", "<person_1>", "Apple"]) == ["Ivan", "Apple"]


def test_filter_placeholders_removes_bare():
    assert _filter_placeholders(["Ivan", "project_1", "service_2"]) == ["Ivan"]


def test_filter_placeholders_removes_containing():
    assert _filter_placeholders(["Bearer <secret_1>", "Ivan"]) == ["Ivan"]


def test_filter_placeholders_removes_double_bracket():
    assert _filter_placeholders(["<<phone_1>>"]) == []


def test_fix_double_brackets():
    assert _fix_double_brackets("Hello <<person_1>> at <<email_1>>") == "Hello <person_1> at <email_1>"


def test_fix_double_brackets_noop():
    assert _fix_double_brackets("Hello <person_1>") == "Hello <person_1>"


REMASKED_RESPONSE = "<person_1> is here.\n\nMapping\nIvan -> <person_1>"


def _make_judge(scan_results_sequence, max_iterations=3):
    """Build a MaskingJudge with mocked _scan and a mocked masking LLM."""
    judge = MaskingJudge(
        base_url="http://fake/v1",
        model="fake-model",
        max_iterations=max_iterations,
    )
    judge._scan = MagicMock(side_effect=list(scan_results_sequence) + [[]] * 10)

    masking_llm = MagicMock()
    masking_llm.complete.return_value = REMASKED_RESPONSE
    return judge, masking_llm


def test_review_already_clean():
    judge, masking_llm = _make_judge([[]])

    text, iterations, remaining = judge.review(
        "Hello <person_1>.", ChunkMerger(), masking_llm, "system"
    )

    assert iterations == 0
    assert remaining == []
    masking_llm.complete.assert_not_called()


def test_review_one_cycle():
    judge, masking_llm = _make_judge([["Ivan"], []])

    text, iterations, remaining = judge.review(
        "Hello Ivan.\n\nOther paragraph.", ChunkMerger(), masking_llm, "system"
    )

    assert iterations == 1
    assert remaining == []
    assert masking_llm.complete.call_count == 1


def test_review_only_dirty_paragraph_re_sent():
    judge, masking_llm = _make_judge([["Ivan"], []])

    judge.review(
        "Clean start.\n\nHello Ivan.\n\nClean end.", ChunkMerger(), masking_llm, "system"
    )

    sent = masking_llm.complete.call_args[0][1]
    assert "Ivan" in sent
    assert "Clean start" not in sent


def test_review_respects_max_iterations():
    judge, masking_llm = _make_judge([["Ivan"]] * 10, max_iterations=2)

    _, iterations, remaining = judge.review(
        "Hello Ivan.\n\nOther.", ChunkMerger(), masking_llm, "system"
    )

    assert iterations == 2
    assert "Ivan" in remaining


def test_client_judge_model_none_skips_judge():
    """No judge is created when judge_model is not set."""
    from llm_mask import MaskingClient
    client = MaskingClient(base_url="http://fake/v1")
    assert client._judge is None


def test_client_judge_wires_correctly():
    """judge_model triggers MaskingJudge creation with correct params."""
    from llm_mask import MaskingClient
    from llm_mask._judge import MaskingJudge

    with patch("llm_mask.client.MaskingJudge") as MockJudge:
        MockJudge.return_value = MagicMock(spec=MaskingJudge)
        MaskingClient(
            base_url="http://mask/v1",
            model="mask-model",
            judge_model="judge-model",
            judge_base_url="http://judge/v1",
            judge_iterations=2,
        )

    MockJudge.assert_called_once_with(
        base_url="http://judge/v1",
        model="judge-model",
        api_key="EMPTY",
        language="ru",
        max_iterations=2,
        enable_thinking=False,
    )


def test_client_judge_defaults_to_same_server():
    """When judge_base_url is omitted, it falls back to base_url."""
    from llm_mask import MaskingClient
    from llm_mask._judge import MaskingJudge

    with patch("llm_mask.client.MaskingJudge") as MockJudge:
        MockJudge.return_value = MagicMock(spec=MaskingJudge)
        MaskingClient(
            base_url="http://mask/v1",
            judge_model="judge-model",
        )

    call_kwargs = MockJudge.call_args[1]
    assert call_kwargs["base_url"] == "http://mask/v1"


def test_client_mask_with_judge_populates_result():
    """judge_iterations and remaining_entities are set on MaskingResult."""
    from llm_mask import MaskingClient
    from llm_mask._judge import MaskingJudge

    fake_mask_resp = "Hello <person_1>.\n\nMapping\nIvan -> <person_1>"

    mock_judge = MagicMock(spec=MaskingJudge)
    mock_judge.review.return_value = ("Hello <person_1>.", 1, [])

    with patch("llm_mask.client.MaskingJudge", return_value=mock_judge):
        client = MaskingClient(
            base_url="http://fake/v1",
            language="en",
            judge_model="judge-model",
        )

    with patch.object(client._llm, "complete", return_value=fake_mask_resp):
        result = client.mask("Hello Ivan.")

    assert result.judge_iterations == 1
    assert result.remaining_entities == []
