"""Train a Hugging Face `tokenizers` BPE tokenizer for English or French text."""

from __future__ import annotations

import time
from pathlib import Path

import click
import pandas as pd

from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)

MIN_FREQUENCY = 100
STEP_CHARS = 10_000

TOKENIZERS_DIR = Path("tokenizers")

DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "en-fr-subset1M.csv"


def _load_corpus(csv_path: Path, lang: str) -> str:
    df = pd.read_csv(csv_path)
    col = "en" if lang == "en" else "fr"
    return "\n".join(df[col].dropna().str.strip().astype(str).tolist())


def _corpus_chunks(corpus: str, step: int):
    for i in range(0, len(corpus), step):
        yield corpus[i : i + step]


def _make_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.decoder = decoders.ByteLevel()
    return tok


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--lang",
    type=click.Choice(["en", "fr"], case_sensitive=True),
    required=True,
    help="Train on the English or French column of the CSV.",
)
@click.option(
    "--vocab-size",
    "-v",
    type=int,
    required=True,
    help="Target vocabulary size for BPE.",
)
@click.option(
    "--data",
    "csv_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=DEFAULT_DATA,
    show_default=True,
    help="CSV with columns 'en' and 'fr'.",
)
@click.option(
    "--tokenizers-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=TOKENIZERS_DIR,
    show_default=True,
    help="Directory in which per-tokenizer folders are created.",
)
def main(lang: str, vocab_size: int, csv_path: Path, tokenizers_dir: Path) -> None:
    start = time.perf_counter()
    out_dir = (tokenizers_dir / f"{lang}-{vocab_size}").resolve()
    out_json = out_dir / "tokenizer.json"

    corpus = _load_corpus(csv_path, lang)
    tok = _make_tokenizer()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=MIN_FREQUENCY)
    tok.train_from_iterator(_corpus_chunks(corpus, STEP_CHARS), trainer=trainer)

    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_json))

    click.echo(
        f"Saved tokenizer to {out_json} ({(time.perf_counter() - start) / 60:.2f} min)"
    )


if __name__ == "__main__":
    main()
