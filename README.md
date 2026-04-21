# simple-translate
English ↔ French neural machine translation.

The model uses an encoder-decoder transformer architecture built from scratch, inspired by _[Attention Is All You Need](https://arxiv.org/abs/1706.03762)_. The project features a Dash web app for the UX and a Modal-hosted inference backend.

# Project Organization
```
├── README.md                <- Overview
├── app.py                   <- Dash web app frontend
├── backend.py               <- Modal inference server (FastAPI)
├── architecture.py          <- Transformer model architecture
├── flavors.py               <- Model configurations/variants
├── schemas.py               <- Shared Pydantic models for API
├── trainer.py               <- Model training logic
├── trainer_job.py           <- Modal entrypoint for training
├── create_tokenizer.py      <- Script for training tokenizers
├── Dockerfile               <- Container config for Dash app
├── pyproject.toml           <- Dependency configuration (uv)
├── uv.lock                  <- Locked dependencies
```

# Installation
This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

1. Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies
```bash
uv sync
```

# Usage
A Dash web app provides the frontend for interacting with the model.

The app can be run locally with:
```bash
uv run python app.py
```

The app connects to an inference backend. Ensure the `SIMPLE_TRANSLATE_SERVER_URL` environment variable is set to your deployed Modal service URL.
