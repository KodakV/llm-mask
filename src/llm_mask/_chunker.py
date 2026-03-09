def split_into_chunks(text: str, chunk_size: int) -> list[str]:
    """Split *text* into chunks of at most *chunk_size* characters.

    The splitter tries to keep logical boundaries intact:

    1. Split on blank lines (paragraph boundaries).
    2. If a paragraph is still too large, split on single newlines.
    3. If a line is still too large, hard-split by character count.

    Markdown fenced code blocks (triple backtick) are kept whole whenever
    they fit within *chunk_size*; if a code block is larger than *chunk_size*
    it is hard-split at the character boundary.
    """
    if len(text) <= chunk_size:
        return [text]

    paragraphs = _split_respecting_code_blocks(text)
    return _pack_segments(paragraphs, chunk_size)


def _split_respecting_code_blocks(text: str) -> list[str]:
    """Split text on blank lines, but keep fenced code blocks intact."""
    segments: list[str] = []
    current: list[str] = []
    in_code_block = False

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block

        if not in_code_block and stripped == "":
            current.append(line)
            segments.append("".join(current))
            current = []
        else:
            current.append(line)

    if current:
        segments.append("".join(current))

    return segments


def _pack_segments(segments: list[str], chunk_size: int) -> list[str]:
    """Greedily pack *segments* into chunks of at most *chunk_size* chars."""
    chunks: list[str] = []
    buffer = ""

    for seg in segments:
        if len(seg) > chunk_size:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_hard_split(seg, chunk_size))
            continue

        if len(buffer) + len(seg) <= chunk_size:
            buffer += seg
        else:
            if buffer:
                chunks.append(buffer)
            buffer = seg

    if buffer:
        chunks.append(buffer)

    return chunks


def _hard_split(text: str, chunk_size: int) -> list[str]:
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
