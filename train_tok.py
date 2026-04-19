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
STEP_SIZE_CHARS = 10_000


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

    corpus = _load_corpus(args.data, args.lang)
    tok = _make_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size, min_frequency=MIN_FREQUENCY
    )
    tok.train_from_iterator(_corpus_chunks(corpus, STEP_SIZE_CHARS), trainer=trainer)

    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_json))

    print(
        f"Saved tokenizer to {out_json} ({(time.perf_counter() - start) / 60:.1f} min)"
    )


if __name__ == "__main__":
    main()
