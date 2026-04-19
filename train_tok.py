"""Train a Hugging Face `tokenizers` BPE tokenizer for English or French text."""

import argparse
import time
from pathlib import Path

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

TOKENIZERS_DIR = Path("tokenizers")
DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "en-fr-subset1M.csv"
MIN_FREQUENCY = 100
CHUNK_SIZE_ROWS = 10_000


def _iter_corpus(csv_path: Path, lang: str):
    """Yield text chunks from a CSV file to avoid loading everything at once."""
    col = "en" if lang == "en" else "fr"
    for df_chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE_ROWS):
        yield "\n".join(df_chunk[col].dropna().str.strip().astype(str))


def _make_tokenizer() -> Tokenizer:
    """Initialize a BPE tokenizer with ByteLevel pre-tokenization and NFD normalization."""
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase()])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    tok.decoder = decoders.ByteLevel()
    return tok


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on the English or French column of a CSV."
    )
    parser.add_argument(
        "--lang",
        "-l",
        required=True,
        choices=["en", "fr"],
        help="Language to train on ('en' or 'fr').",
    )
    parser.add_argument(
        "--vocab-size",
        "-v",
        type=int,
        required=True,
        help="Target vocabulary size.",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=Path,
        default=DEFAULT_DATA,
        help=f"CSV with columns 'en' and 'fr' (default: {DEFAULT_DATA})",
    )
    parser.add_argument(
        "--tokenizers-dir",
        "-t",
        type=Path,
        default=TOKENIZERS_DIR,
        help=f"Root directory for tokenizer output folders (default: {TOKENIZERS_DIR})",
    )
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"CSV not found: {args.data}")

    start = time.perf_counter()
    out_dir = (args.tokenizers_dir / f"{args.lang}-{args.vocab_size}").resolve()
    out_json = out_dir / "tokenizer.json"

    tok = _make_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size, min_frequency=MIN_FREQUENCY
    )
    tok.train_from_iterator(_iter_corpus(args.data, args.lang), trainer=trainer)

    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_json))

    print(
        f"Saved tokenizer to {out_json} ({(time.perf_counter() - start) / 60:.1f} min)"
    )


if __name__ == "__main__":
    main()
