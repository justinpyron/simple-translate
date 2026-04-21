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

# Deployment
The project uses a dual-cloud architecture:
- **Frontend**: A Dockerized Dash app deployed to **Google Cloud Run**.
- **Inference Backend**: A **Modal** app providing a FastAPI endpoint.
- **CI/CD**: A GitHub Action (`.github/workflows/build-and-push-image.yml`) automates the deployment of both components.

# Development
### 1. Training on Modal
Retrain the model using the Modal entrypoint. This requires a [Modal account](https://modal.com/):
```bash
uv run modal run trainer_job.py \
    --flavor small \
    --direction en2fr \
    --tokenizer-dir-source tokenizers/en-vocab_1000 \
    --tokenizer-dir-destination tokenizers/fr-vocab_1000 \
    --dataset-filename-train data/train.csv \
    --dataset-filename-val data/val.csv \
    --num-train-examples 1000000 \
    --batch-size-train 64 \
    --lr-start 1e-4 \
    --eval-every 5000
```

### 2. Tokenizers
Generate new BPE tokenizers for English or French:
```bash
uv run python create_tokenizer.py --lang en --vocab-size 1000
```
