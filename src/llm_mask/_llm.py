from openai import OpenAI

from .exceptions import LLMError


class _LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_tokens: int = 16384,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

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
            )
        except Exception as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMError(
                "LLM response was truncated (finish_reason='length'). "
                f"Increase max_tokens (current: {self._max_tokens}) or reduce chunk_size."
            )
        return choice.message.content or ""
