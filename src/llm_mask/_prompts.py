"""Load bundled system prompts via importlib.resources."""
from importlib import resources

_PACKAGE = "llm_mask.prompts"

_SUPPORTED = {"ru", "en"}


def load_prompt(language: str) -> str:
    if language not in _SUPPORTED:
        raise ValueError(f"Unsupported language {language!r}. Choose from {_SUPPORTED}.")
    ref = resources.files(_PACKAGE) / language / "mask_prompt.txt"
    return ref.read_text(encoding="utf-8")
