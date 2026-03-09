# Contributing

Contributions are welcome! Here's how to get started.

## Development setup

```bash
git clone https://github.com/vladislavkodak/llm-mask
cd llm-mask
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

Tests use mocked LLM responses — no live server required.

## Code style

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Submitting a pull request

1. Fork the repo and create a branch: `git checkout -b feature/my-feature`
2. Make your changes and add tests
3. Run `pytest` and `ruff check` — both must pass
4. Open a pull request with a clear description of what and why

## Reporting issues

Please open an issue on [GitHub Issues](https://github.com/vladislavkodak/llm-mask/issues) with:
- Python version
- LLM server and model used
- Minimal reproducible example
