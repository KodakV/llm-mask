# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-09

### Added
- `MaskingClient` — main public API for masking and unmasking text
- Support for `.txt`, `.md`, `.json`, `.html` file types
- Paragraph-aware, code-block-safe text chunker
- Cross-chunk placeholder deduplication via `ChunkMerger`
- Optional judge LLM pass to catch missed entities
- Built-in prompts for Russian (`ru`) and English (`en`)
- `mask_file()` and `mask_directory()` helpers
- `MappingStore` for accumulating mappings across multiple files
- Full pytest suite (55 tests, no live LLM required)
