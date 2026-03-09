class LLMMaskError(Exception):
    """Base exception for llm-mask."""


class ParsingError(LLMMaskError):
    """Raised when the LLM response cannot be parsed."""


class LLMError(LLMMaskError):
    """Raised when the LLM request fails."""


class UnsupportedFileTypeError(LLMMaskError):
    """Raised when the file type is not supported."""


class FileReadError(LLMMaskError):
    """Raised when a file cannot be read (not found or unreadable)."""
