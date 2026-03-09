"""Smoke tests for MaskingClient that don't require a live LLM.

The LLM layer is monkey-patched so tests run entirely offline.
"""
from unittest.mock import patch


from llm_mask import MaskingClient, MaskingResult


FAKE_LLM_RESPONSE = """\
Hello, <person_1>! You work at <company_1>.

Mapping
Ivan -> <person_1>
Apple -> <company_1>
"""


def test_mask_returns_result():
    client = MaskingClient(base_url="http://localhost:9999/v1", language="en")
    with patch.object(client._llm, "complete", return_value=FAKE_LLM_RESPONSE):
        result = client.mask("Hello, Ivan! You work at Apple.")

    assert isinstance(result, MaskingResult)
    assert "<person_1>" in result.masked_text
    assert result.mapping["Ivan"] == "<person_1>"
    assert result.mapping["Apple"] == "<company_1>"


def test_mask_supports_tuple_unpacking():
    client = MaskingClient(base_url="http://localhost:9999/v1", language="en")
    with patch.object(client._llm, "complete", return_value=FAKE_LLM_RESPONSE):
        masked_text, mapping = client.mask("Hello, Ivan! You work at Apple.")

    assert "<person_1>" in masked_text
    assert mapping["Ivan"] == "<person_1>"


def test_mask_chunks_processed():
    client = MaskingClient(base_url="http://localhost:9999/v1", language="en", chunk_size=10)
    long_text = "Ivan " * 30
    with patch.object(client._llm, "complete", return_value=FAKE_LLM_RESPONSE):
        result = client.mask(long_text)

    assert result.chunks_processed > 1


def test_unmask_roundtrip():
    client = MaskingClient(base_url="http://localhost:9999/v1", language="en")
    with patch.object(client._llm, "complete", return_value=FAKE_LLM_RESPONSE):
        result = client.mask("Hello, Ivan! You work at Apple.")

    restored = client.unmask(result.masked_text, result.mapping)
    assert "Ivan" in restored
    assert "Apple" in restored


def test_mask_file(tmp_path):
    sample = tmp_path / "doc.md"
    sample.write_text("Hello, Ivan! You work at Apple.", encoding="utf-8")

    client = MaskingClient(base_url="http://localhost:9999/v1", language="en")
    with patch.object(client._llm, "complete", return_value=FAKE_LLM_RESPONSE):
        result = client.mask_file(sample, save_masked=True, save_mapping=True, mapping_dir=tmp_path)

    assert result.source_file == str(sample)
    assert (tmp_path / "doc_masked.md").exists()
    assert (tmp_path / "doc_mapping.json").exists()


def test_mask_directory(tmp_path):
    for i in range(3):
        (tmp_path / f"file{i}.md").write_text("Ivan works at Apple.", encoding="utf-8")

    client = MaskingClient(base_url="http://localhost:9999/v1", language="en")
    store_path = tmp_path / "store.json"
    with patch.object(client._llm, "complete", return_value=FAKE_LLM_RESPONSE):
        results = client.mask_directory(tmp_path, pattern="*.md", mapping_store_path=store_path)

    assert len(results) == 3
    assert store_path.exists()
