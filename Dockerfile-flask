FROM python:3.10-slim
WORKDIR /app
COPY pyproject.toml \
    poetry.lock \
    simple_translate.py \
    model_configs.py \
    model_for_app_cpu.pt \
    flask_app.py \
    /app/
COPY tokenizer_1000 /app/tokenizer_1000
RUN pip install poetry
RUN poetry install
EXPOSE 8080
ENTRYPOINT ["poetry", "run", "python", "flask_app.py"]
