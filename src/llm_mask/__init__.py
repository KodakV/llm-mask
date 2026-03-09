"""llm-mask: mask sensitive data in documents using a local OpenAI-compatible LLM."""

from .client import MaskingClient
from .exceptions import FileReadError, LLMError, LLMMaskError, ParsingError, UnsupportedFileTypeError
from .mapping import MappingStore, MaskingResult
from .unmasker import Unmasker

__all__ = [
    "MaskingClient",
    "MaskingResult",
    "MappingStore",
    "Unmasker",
    "LLMMaskError",
    "LLMError",
    "ParsingError",
    "UnsupportedFileTypeError",
    "FileReadError",
]
