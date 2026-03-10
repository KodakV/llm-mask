"""LLM judge — second-pass quality review of masked text.

The judge model receives the already-masked text and checks whether any
sensitive entities were missed.  If it finds remaining entities it returns
their exact strings; the caller then re-sends the affected paragraphs to
the masking LLM for correction.

The judge can be the same model as the masking LLM (same base_url + model)
or a separate, potentially larger model hosted elsewhere.
"""
from __future__ import annotations

import json
import logging
import re

from openai import OpenAI

from ._llm import _LLMClient
from ._merger import ChunkMerger
from ._parser import ParsingError, parse_llm_response
from .exceptions import LLMError

log = logging.getLogger(__name__)

_JUDGE_PROMPT_EN = """\
You are a quality-control judge for a text anonymisation system.

Your task: review the text below and find any sensitive entities that were \
NOT anonymised.

Look for: person names, usernames, company names, brand names, organisation \
names, email addresses, phone numbers, IP addresses, tokens, secrets, file \
paths, internal project names.

Rules:
- Tokens matching these patterns are ALREADY masked — ignore them completely:
  <person_1>, <person_2>, ... — person names
  <company_1>, <company_2>, ... — company/brand names
  <email_1>, <url_1>, <phone_1>, <ip_1>, ... — contact/network data
  project_1, project_2, service_1, service_2, ... — bare tokens (no angle brackets) \
for project/service names
- Return ONLY a JSON array of strings with the exact text of any unmasked \
entities you find.
- If everything is properly anonymised, return an empty array: []
- No explanations, no comments — only the JSON array.

Examples:
  Input:  "Hello Ivan, please contact Apple at support@apple.com."
  Output: ["Ivan", "Apple", "support@apple.com"]

  Input:  "Hello <person_1>, please contact <company_1> at <email_1>."
  Output: []

  Input:  "Project project_1 uses service_1 at <url_1>."
  Output: []
"""

_JUDGE_PROMPT_RU = """\
Ты судья-контролёр качества системы анонимизации текстов.

Твоя задача: проверить текст ниже и найти чувствительные сущности, которые \
НЕ были анонимизированы.

Ищи: имена людей, никнеймы, названия компаний, брендов, организаций, \
email-адреса, телефоны, IP-адреса, токены, секреты, файловые пути, \
названия внутренних проектов.

Правила:
- Токены следующих паттернов УЖЕ замаскированы — игнорируй их полностью:
  <person_1>, <person_2>, ... — имена людей
  <company_1>, <company_2>, ... — компании/бренды
  <email_1>, <url_1>, <phone_1>, <ip_1>, ... — контактные/сетевые данные
  project_1, project_2, service_1, service_2, ... — bare-токены (без угловых скобок) \
для проектов/сервисов
- Верни ТОЛЬКО JSON-массив строк с точным текстом найденных незамаскированных \
сущностей.
- Если всё корректно анонимизировано — верни пустой массив: []
- Никаких объяснений и комментариев — только JSON-массив.

Примеры:
  Вход:  "Привет Иван, напиши на ivan@apple.com или позвони в Apple."
  Выход: ["Иван", "ivan@apple.com", "Apple"]

  Вход:  "Привет <person_1>, напиши на <email_1> или позвони в <company_1>."
  Выход: []

  Вход:  "Проект project_1 использует service_1 по адресу <url_1>."
  Выход: []
"""

_JUDGE_PROMPTS: dict[str, str] = {"en": _JUDGE_PROMPT_EN, "ru": _JUDGE_PROMPT_RU}


class MaskingJudge:
    """Review masked text with a second LLM and re-mask missed entities.

    Parameters
    ----------
    base_url:
        Base URL of the judge LLM server (OpenAI-compatible).
        Can be the same as the masking server or a different endpoint.
    model:
        Model name on the judge server.
    api_key:
        API key for the judge server.
    language:
        Prompt language: ``"ru"`` or ``"en"``.
    temperature:
        Keep at 0.0 for deterministic output.
    max_iterations:
        Maximum judge → re-mask cycles.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        language: str = "ru",
        temperature: float = 0.0,
        max_iterations: int = 3,
        enable_thinking: bool = False,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._prompt = _JUDGE_PROMPTS.get(language, _JUDGE_PROMPT_EN)
        self._max = max_iterations
        self._enable_thinking = enable_thinking

    def review(
        self,
        masked_text: str,
        merger: ChunkMerger,
        masking_llm: _LLMClient,
        masking_prompt: str,
    ) -> tuple[str, int, list[str]]:
        """Run the judge review loop.

        Parameters
        ----------
        masked_text:
            Output of the main masking pass.
        merger:
            The same :class:`ChunkMerger` used during masking — ensures
            placeholder numbering stays consistent across re-mask cycles.
        masking_llm:
            The masking :class:`_LLMClient` used for corrections.
        masking_prompt:
            The masking system prompt.

        Returns
        -------
        masked_text : str
            Corrected text after all judge cycles.
        iterations : int
            Number of judge → re-mask cycles executed.
        remaining : list[str]
            Entities the judge still found after ``max_iterations``
            (empty = fully clean).
        """
        for iteration in range(1, self._max + 1):
            found = self._scan(masked_text)

            if not found:
                log.debug("Judge: clean after %d iteration(s)", iteration - 1)
                return masked_text, iteration - 1, []

            log.debug("Judge iteration %d: found %s", iteration, found)

            dirty = _dirty_paragraph_indices(masked_text, found)
            paragraphs = masked_text.split("\n\n")

            for idx in sorted(dirty):
                log.debug("  re-masking paragraph %d", idx)
                try:
                    raw = masking_llm.complete(masking_prompt, paragraphs[idx])
                    try:
                        fixed, para_mapping = parse_llm_response(raw)
                    except ParsingError:
                        fixed, para_mapping = raw.strip(), {}
                    fixed = _fix_double_brackets(fixed)
                    patched, _ = merger.add_chunk(fixed, para_mapping)
                    paragraphs[idx] = patched
                except Exception as exc:
                    log.warning("  could not re-mask paragraph %d: %s", idx, exc)

            masked_text = "\n\n".join(paragraphs)

        remaining = self._scan(masked_text)
        if remaining:
            log.warning(
                "Judge: %d entity(-ies) remain after %d iteration(s): %s",
                len(remaining), self._max, remaining,
            )
        return masked_text, self._max, remaining

    def _scan(self, text: str) -> list[str]:
        """Ask the judge model what is still unmasked. Returns entity strings."""
        if not text.strip():
            return []
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": self._prompt},
                    {"role": "user", "content": text},
                ],
                extra_body={"chat_template_kwargs": {"enable_thinking": self._enable_thinking}},
            )
        except Exception as exc:
            raise LLMError(f"Judge LLM request failed: {exc}") from exc

        raw = response.choices[0].message.content or "[]"
        return _filter_placeholders(_parse_entity_list(raw))


_PH_EXACT_RE = re.compile(r"^<{1,2}[a-z_]+_\d+>{1,2}$|^[a-z_]+_\d+$")
_DOUBLE_BRACKET_RE = re.compile(r"<<([a-z_]+_\d+)>>")
_PH_CONTAINS_RE = re.compile(r"<[a-z_]+_\d+>")


def _fix_double_brackets(text: str) -> str:
    """Collapse ``<<type_N>>`` into ``<type_N>`` — artifact of re-masking."""
    return _DOUBLE_BRACKET_RE.sub(r"<\1>", text)


def _filter_placeholders(entities: list[str]) -> list[str]:
    """Remove entries that are already-masked placeholders or contain one.

    Filters out:
    - Exact placeholders: ``<person_1>``, ``project_1``, ``<<phone_1>>``
    - Strings that contain a placeholder: ``Bearer <secret_3>``
    """
    result = []
    for e in entities:
        if _PH_EXACT_RE.match(e):
            continue
        if _PH_CONTAINS_RE.search(e):
            continue
        result.append(e)
    return result


def _dirty_paragraph_indices(text: str, entities: list[str]) -> set[int]:
    """Return paragraph indices (split on ``\\n\\n``) that contain an entity."""
    paragraphs = text.split("\n\n")
    dirty: set[int] = set()
    for idx, para in enumerate(paragraphs):
        for entity in entities:
            if entity and entity in para:
                dirty.add(idx)
                break
    return dirty


def _parse_entity_list(raw: str) -> list[str]:
    """Extract a JSON string list from the judge response.

    Tolerates markdown fences and minor formatting noise.
    """
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    raw = raw.strip()

    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return [str(e) for e in result if e]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(e) for e in result if e]
        except json.JSONDecodeError:
            pass

    log.warning("Judge returned unparseable response: %r", raw[:200])
    return []
