"""Train a Hugging Face `tokenizers` BPE tokenizer for English or French text."""

import argparse
import time
from pathlib import Path

import pandas as pd
from transformers import PreTrainedTokenizerFast

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
DEFAULT_DATA = Path("data/en-fr-subset10M-shuffled.csv")
MIN_FREQUENCY = 100
CHUNK_SIZE_ROWS = 100_000

# Special tokens for Neural Machine Translation (NMT).
# The order here determines their IDs (PAD=0, BOS=1, EOS=2, UNK=3).
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
TOKEN_PAD, TOKEN_BOS, TOKEN_EOS, TOKEN_UNK = SPECIAL_TOKENS


def _iter_corpus(csv_path: Path, lang: str):
    """Yield text chunks from a CSV file to avoid loading everything at once."""
    col = "en" if lang == "en" else "fr"
    for df_chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE_ROWS):
        yield "\n".join(df_chunk[col].dropna().str.strip().astype(str))


def _make_tokenizer() -> Tokenizer:
    """Initialize a BPE tokenizer with ByteLevel pre-tokenization and NFD normalization."""
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Strip()])
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # TemplateProcessing handles automatic BOS/EOS wrapping for NMT.
    # The IDs provided here MUST match the indices in the SPECIAL_TOKENS list,
    # because BpeTrainer assigns IDs based on the order of the special_tokens list.
    tok.post_processor = processors.TemplateProcessing(
        single=f"{TOKEN_BOS} $A {TOKEN_EOS}",
        special_tokens=[
            (TOKEN_BOS, SPECIAL_TOKENS.index(TOKEN_BOS)),
            (TOKEN_EOS, SPECIAL_TOKENS.index(TOKEN_EOS)),
        ],
    )

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

    # --- Parse and Validate Arguments ---
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"CSV not found: {args.data}")

    # --- Setup Output Paths ---
    start = time.perf_counter()
    out_dir = (args.tokenizers_dir / f"{args.lang}-vocab_{args.vocab_size}").resolve()

    # --- Initialize and Train Tokenizer ---
    tok = _make_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )
    tok.train_from_iterator(_iter_corpus(args.data, args.lang), trainer=trainer)

    # --- Save Artifacts ---
    out_dir.mkdir(parents=True, exist_ok=True)

    # Wrap in PreTrainedTokenizerFast to attach special token meanings (bos_token, etc.)
    # and save helper files (tokenizer_config.json, special_tokens_map.json) for flavors.py.
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=TOKEN_BOS,
        eos_token=TOKEN_EOS,
        pad_token=TOKEN_PAD,
        unk_token=TOKEN_UNK,
    )
    wrapped.save_pretrained(out_dir)

    # --- Final Report ---
    print(
        f"Saved tokenizer to {out_dir} ({(time.perf_counter() - start) / 60:.1f} min)"
    )


if __name__ == "__main__":
    main()
