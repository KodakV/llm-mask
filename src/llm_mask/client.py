from __future__ import annotations

from pathlib import Path

from ._chunker import split_into_chunks
from ._judge import MaskingJudge
from ._llm import _LLMClient
from ._merger import ChunkMerger
from ._parser import parse_llm_response, recover_mapping, repair_arrow_collisions
from ._prompts import load_prompt
from ._reader import read_file
from ._ner_masker import NerMasker
from ._regex_masker import RegexMasker
from ._repair import repair_dup_ph
from .mapping import MappingStore, MaskingResult
from .unmasker import Unmasker


class MaskingClient:
    """Mask and unmask sensitive data in text documents.

    Parameters
    ----------
    base_url:
        Base URL of the masking LLM server (OpenAI-compatible).
    model:
        Model name on the masking server.
    api_key:
        API key (most local servers accept any non-empty string).
    language:
        Built-in prompt language: ``"ru"`` or ``"en"``.
    chunk_size:
        Maximum characters per masking LLM request.
    temperature:
        Sampling temperature (0.0 = deterministic).
    use_ner:
        Enable NER pre-masking stage (requires ``pip install llm-mask[ner]``).
        Automatically disabled if natasha is not installed.
    judge_model:
        When set, enables a second LLM pass that reviews the masked output
        and triggers re-masking of any paragraphs where sensitive entities
        were missed.  The judge can be the same model as the masking LLM
        or a separate, larger model for more sensitive workloads.
    judge_base_url:
        Base URL of the judge LLM server.  Defaults to ``base_url`` when
        not provided, meaning the same server handles both passes.
    judge_api_key:
        API key for the judge server.  Defaults to ``api_key``.
    judge_iterations:
        Maximum judge → re-mask cycles per ``mask()`` call.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        model: str = "local-model",
        api_key: str = "EMPTY",
        language: str = "ru",
        chunk_size: int = 3000,
        temperature: float = 0.0,
        max_tokens: int = 16384,
        use_ner: bool = True,
        judge_model: str | None = None,
        judge_base_url: str | None = None,
        judge_api_key: str | None = None,
        judge_iterations: int = 3,
        enable_thinking: bool = False,
        repeat_penalty: float = 1.1,
    ) -> None:
        self._llm = _LLMClient(
            base_url=base_url, model=model, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            repeat_penalty=repeat_penalty,
        )
        self._system_prompt = load_prompt(language)
        self._chunk_size = chunk_size
        self._regex_masker = RegexMasker()
        self._ner_masker: NerMasker | None = NerMasker() if (use_ner and NerMasker.available) else None
        self._unmasker = Unmasker()

        self._judge: MaskingJudge | None = None
        if judge_model is not None:
            self._judge = MaskingJudge(
                base_url=judge_base_url or base_url,
                model=judge_model,
                api_key=judge_api_key or api_key,
                language=language,
                max_iterations=judge_iterations,
                enable_thinking=enable_thinking,
            )

    def mask(self, text: str) -> MaskingResult:
        """Mask sensitive entities in *text*.

        Pipeline:
        1. Deterministic regex pre-masking (emails, phones, IPs, URLs, secrets…)
        2. LLM masking of fuzzy entities (names, companies, addresses…) per chunk
        3. Merge regex + LLM mappings
        4. Post-processing: repair DUP-PH and arrow collisions
        5. Optional judge review cycles

        Supports both attribute access and tuple unpacking::

            result = client.mask(text)
            result.masked_text
            result.mapping

            masked_text, mapping = client.mask(text)
        """
        # ── Stage 1a: deterministic regex pre-masking ─────────────────────
        pre_masked, regex_mapping = self._regex_masker.mask(text)

        # ── Stage 1b: NER pre-masking (optional) ──────────────────────────
        ner_mapping: dict[str, str] = {}
        if self._ner_masker is not None:
            pre_masked, ner_mapping = self._ner_masker.mask(pre_masked, preloaded=regex_mapping)

        # ── Stage 2: LLM masking of fuzzy entities ────────────────────────
        chunks = split_into_chunks(pre_masked, self._chunk_size)
        merger = ChunkMerger(preloaded={**regex_mapping, **ner_mapping})
        masked_parts: list[str] = []

        for chunk in chunks:
            raw = self._llm.complete(self._system_prompt, chunk)
            masked_chunk, chunk_mapping = parse_llm_response(raw)
            chunk_mapping = recover_mapping(chunk, masked_chunk, chunk_mapping)
            patched, _ = merger.add_chunk(masked_chunk, chunk_mapping,
                                          pre_masked_chunk=chunk)
            masked_parts.append(patched)

        masked_text = "".join(masked_parts)

        # ── Stage 3: merge regex + NER + LLM mappings ────────────────────
        combined_mapping = {**merger.global_mapping(), **ner_mapping, **regex_mapping}

        # ── Stage 4: post-processing repairs ─────────────────────────────
        masked_text, combined_mapping = repair_dup_ph(masked_text, combined_mapping)
        masked_text = repair_arrow_collisions(masked_text, text, combined_mapping)

        # ── Stage 5: optional judge ───────────────────────────────────────
        judge_iterations = 0
        remaining_entities: list[str] = []

        if self._judge is not None:
            masked_text, judge_iterations, remaining_entities = self._judge.review(
                masked_text=masked_text,
                merger=merger,
                masking_llm=self._llm,
                masking_prompt=self._system_prompt,
            )

        return MaskingResult(
            masked_text=masked_text,
            mapping=combined_mapping,
            chunks_processed=len(chunks),
            judge_iterations=judge_iterations,
            remaining_entities=remaining_entities,
        )

    def unmask(self, masked_text: str, mapping: dict[str, str]) -> str:
        """Restore masked text using its mapping (no LLM call)."""
        return self._unmasker.unmask(masked_text, mapping)

    def mask_file(
        self,
        file_path: str | Path,
        *,
        save_masked: bool = False,
        save_mapping: bool = False,
        mapping_dir: str | Path | None = None,
    ) -> MaskingResult:
        """Mask a file (.md / .txt / .json / .html).

        Parameters
        ----------
        save_masked:
            Write the masked document with a ``_masked`` suffix.
        save_mapping:
            Write the mapping JSON.
        mapping_dir:
            Where to save the mapping file (defaults to the file's directory).
        """
        path = Path(file_path)
        text = read_file(path)
        result = self.mask(text)
        result.source_file = str(path)

        if save_masked:
            out_path = path.with_stem(path.stem + "_masked")
            out_path.write_text(result.masked_text, encoding="utf-8")

        if save_mapping:
            mdir = Path(mapping_dir) if mapping_dir else path.parent
            mapping_path = mdir / (path.stem + "_mapping.json")
            result.save_mapping(mapping_path)

        return result

    def unmask_file(
        self, masked_file: str | Path, mapping_file: str | Path
    ) -> str:
        """Load a masked file and its mapping JSON, return restored text."""
        return self._unmasker.unmask_file(masked_file, mapping_file)

    def mask_directory(
        self,
        directory: str | Path,
        *,
        pattern: str = "*.md",
        overwrite_originals: bool = False,
        mapping_store_path: str | Path | None = None,
    ) -> list[MaskingResult]:
        """Mask all files matching *pattern* in *directory*.

        Parameters
        ----------
        overwrite_originals:
            If ``True``, overwrite each file in place.
            If ``False``, write ``*_masked`` copies next to the originals.
        mapping_store_path:
            Single JSON file that accumulates mappings for all processed files.
        """
        directory = Path(directory)
        files = list(directory.glob(pattern))
        store = MappingStore(mapping_store_path) if mapping_store_path else None

        results: list[MaskingResult] = []
        try:
            for file_path in files:
                text = read_file(file_path)
                result = self.mask(text)
                result.source_file = str(file_path)

                if overwrite_originals:
                    file_path.write_text(result.masked_text, encoding="utf-8")
                else:
                    out = file_path.with_stem(file_path.stem + "_masked")
                    out.write_text(result.masked_text, encoding="utf-8")

                if store is not None:
                    store.add(result)

                results.append(result)
        finally:
            if store is not None:
                store.save()

        return results
