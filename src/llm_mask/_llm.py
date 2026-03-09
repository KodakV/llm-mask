from openai import OpenAI

from .exceptions import LLMError


class _LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature

    def complete(self, system_prompt: str, user_content: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
        except Exception as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        return response.choices[0].message.content or ""
