import logging

from openai import OpenAI

from .exceptions import LLMError

log = logging.getLogger(__name__)


class _LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 16384,
        enable_thinking: bool = False,
        repeat_penalty: float = 1.1,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking
        self._repeat_penalty = repeat_penalty

    def complete(self, system_prompt: str, user_content: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": self._enable_thinking},
                    "repeat_penalty": self._repeat_penalty,
                },
            )
        except Exception as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        choice = response.choices[0]
        if choice.finish_reason == "length":
            # Response was truncated — the masked body is usually complete by
            # this point; only the mapping section loops.  Return the partial
            # content and let the parser extract what it can.
            log.warning(
                "LLM response truncated at max_tokens=%d. "
                "Consider reducing chunk_size if mapping quality is poor.",
                self._max_tokens,
            )
        return choice.message.content or ""
