from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


@dataclass
class MaskingResult:
    """Result of a masking operation.

    Supports both attribute access and tuple unpacking::

        result = client.mask(text)
        result.masked_text
        result.mapping

        masked, mapping = client.mask(text)

    When a judge model is configured, extra fields are populated::

        result.judge_iterations    # how many judge→re-mask cycles ran
        result.remaining_entities  # entities judge still found (ideally empty)
    """

    masked_text: str
    mapping: dict[str, str]
    source_file: str | None = None
    chunks_processed: int = 1
    judge_iterations: int = 0
    remaining_entities: list[str] = field(default_factory=list)

    def __iter__(self) -> Iterator:
        """Allow ``masked_text, mapping = result`` unpacking."""
        yield self.masked_text
        yield self.mapping

    def save_mapping(self, path: str | Path) -> None:
        """Write the mapping to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_file": self.source_file,
            "masked_at": datetime.now(timezone.utc).isoformat(),
            "mapping": self.mapping,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_mapping(cls, path: str | Path) -> dict[str, str]:
        """Read a mapping JSON file and return the ``mapping`` dict."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "mapping" in data:
            return data["mapping"]
        return data


class MappingStore:
    """Persist mappings for a whole directory of files in one JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[str, dict[str, str]] = {}
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._data = raw

    def add(self, result: MaskingResult) -> None:
        """Add a result's mapping keyed by its source file."""
        key = result.source_file or "<unknown>"
        self._data[key] = result.mapping

    def save(self) -> None:
        """Persist all mappings to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def get(self, source_file: str) -> dict[str, str]:
        """Return the mapping for *source_file*, or an empty dict if not found."""
        return self._data.get(source_file, {})
