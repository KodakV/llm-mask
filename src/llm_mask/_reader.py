from pathlib import Path
import json

from .exceptions import FileReadError, UnsupportedFileTypeError

SUPPORTED_SUFFIXES = {".txt", ".md", ".json", ".html"}


def read_file(file_path: str | Path) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileReadError(f"{file_path} not found")

    suffix = path.suffix.lower()

    if suffix in {".txt", ".md", ".html"}:
        return path.read_text(encoding="utf-8")

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps(data, indent=2, ensure_ascii=False)

    raise UnsupportedFileTypeError(f"Unsupported file type: {suffix}")
