# simple-translate
English to French neural machine translation.

The model uses an encoder-decoder transformer architecture inspired by _[Attention Is All You Need](https://arxiv.org/abs/1706.03762)_.

# Project Organization
```
├── README.md                <- Overview
├── app.py                   <- Streamlit web app frontend
├── simple_translate.py      <- Architecture of underlying transformer model
├── model_configs.py         <- Hyperparameters of the model
├── model_for_app.pt         <- Weights of the trained model used in the app
├── create_tokenizer.py      <- Trains a tokenizer
├── trainer.py               <- Utils for training the model
├── launch_trainer.py        <- Launches a trainer.py training session
├── flask_app.py             <- Flask app for serving predictions
├── Dockerfile               <- Docker image for Flask app
├── pyproject.toml           <- Poetry config specifying Python environment dependencies
├── poetry.lock              <- Locked dependencies to ensure consistent installs
├── .pre-commit-config.yaml  <- Linting configs
```

# Installation
This project uses [Poetry](https://python-poetry.org/docs/) to manage its Python environment.

1. Install Poetry
```
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies
```
poetry install
```

# Usage
A Streamlit web app is the frontend for interacting with the model.

The app can be accessed at https://simple-translate.streamlit.app.

Alternatively, the app can be run locally with
```
poetry run streamlit run app.py
```
