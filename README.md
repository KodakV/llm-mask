# llm-mask

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://github.com/KodakV/llm-mask/actions/workflows/tests.yml/badge.svg)](https://github.com/KodakV/llm-mask/actions/workflows/tests.yml)

A Python library that masks sensitive data in documents (PII, tokens, URLs, company names, etc.) using a **local OpenAI-compatible LLM**, and restores the original text via a saved mapping — no data leaves your infrastructure.

## Installation

```bash
pip install llm-mask
```

## Requirements

A running local LLM server with an OpenAI-compatible API, e.g. [vLLM](https://github.com/vllm-project/vllm), [LM Studio](https://lmstudio.ai/), or [Ollama](https://ollama.com/).

## Quick start

```python
from llm_mask import MaskingClient

client = MaskingClient(
    base_url="http://localhost:8001/v1",   # your LLM server
    model="local-model",
    language="ru",   # "ru" or "en"
)

text = "Привет, меня зовут Иван, работаю в Apple. Email: ivan@apple.com"

# ── mask ──────────────────────────────────────────────────────────────
masked_text, mapping = client.mask(text)
# masked_text → "Привет, меня зовут <person_1>, работаю в <company_1>. Email: <email_1>"
# mapping     → {"Иван": "<person_1>", "Apple": "<company_1>", "ivan@apple.com": "<email_1>"}

# ── unmask (no LLM call) ───────────────────────────────────────────────
original = client.unmask(masked_text, mapping)
# original → original text restored exactly
```

Attribute-style access also works:

```python
result = client.mask(text)
result.masked_text
result.mapping
```

## File & directory helpers

```python
# Mask a file (nothing written to disk by default)
result = client.mask_file("document.md")

# Write masked file + mapping JSON to disk
result = client.mask_file(
    "document.md",
    save_masked=True,          # → document_masked.md
    save_mapping=True,         # → document_mapping.json
    mapping_dir="./mappings",
)

# Restore from files (no LLM)
original = client.unmask_file("document_masked.md", "document_mapping.json")

# Mask a whole directory
results = client.mask_directory(
    "./docs",
    pattern="*.md",
    overwrite_originals=False,          # writes *_masked.md next to originals
    mapping_store_path="./mappings.json",
)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_url` | `http://localhost:8001/v1` | LLM server base URL |
| `model` | `local-model` | Model identifier |
| `api_key` | `EMPTY` | API key (ignored by most local servers) |
| `language` | `ru` | Built-in prompt language: `"ru"` or `"en"` |
| `chunk_size` | `6000` | Max characters per LLM call |
| `temperature` | `0.0` | Sampling temperature |
| `judge_model` | `None` | Optional second LLM pass to catch missed entities |

## Entity types

| Entity | Placeholder |
|--------|-------------|
| URLs / domains | `<url_1>` |
| Service names | `service_1` |
| Company / brand names | `<company_1>` |
| Person names / usernames | `<person_1>` |
| Email addresses | `<email_1>` |
| Phone numbers | `<phone_1>` |
| IP addresses | `<ip_1>` |
| Tokens / secrets / keys | `<secret_1>` |
| Numeric IDs | `<id_1>` |
| File paths | `<path_1>` |
| Project / code names | `project_1` |
| Infrastructure names | `<env_1>`, `<host_1>` |

## Mapping file format

```json
{
  "source_file": "document.md",
  "masked_at": "2026-03-05T14:22:00Z",
  "mapping": {
    "Apple": "<company_1>",
    "https://api.company.com": "<url_1>"
  }
}
```

## Development

```bash
git clone https://github.com/KodakV/llm-mask
cd llm-mask
pip install -e ".[dev]"
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## License

MIT
