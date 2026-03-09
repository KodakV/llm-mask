from llm_mask._chunker import split_into_chunks


SHORT_TEXT = "Hello world"
LONG_TEXT = "A" * 100


def test_short_text_single_chunk():
    chunks = split_into_chunks(SHORT_TEXT, chunk_size=6000)
    assert chunks == [SHORT_TEXT]


def test_exact_size_single_chunk():
    text = "X" * 100
    chunks = split_into_chunks(text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_hard_split_large_text():
    text = "B" * 250
    chunks = split_into_chunks(text, chunk_size=100)
    assert len(chunks) == 3
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks) == text


def test_paragraph_boundary_preserved():
    para1 = "First paragraph.\n\n"
    para2 = "Second paragraph.\n\n"
    para3 = "Third paragraph."
    text = para1 + para2 + para3
    size = len(para1) + len(para2) + 1
    chunks = split_into_chunks(text, chunk_size=size)
    assert "".join(chunks) == text


def test_code_block_not_split():
    code_block = "```python\nprint('hello')\n```"
    text = "Before\n\n" + code_block + "\n\nAfter"
    chunks = split_into_chunks(text, chunk_size=len(text) - 1)
    combined = "".join(chunks)
    assert "```python\nprint('hello')\n```" in combined


def test_total_content_preserved():
    import random
    import string
    random.seed(42)
    text = "".join(random.choices(string.ascii_letters + " \n", k=5000))
    chunks = split_into_chunks(text, chunk_size=300)
    assert "".join(chunks) == text
