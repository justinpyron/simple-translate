"""Shuffle and split a large CSV into smaller chunks without loading it into memory.

Uses a two-pass external shuffle ("random bucket" shuffle):
  1. Stream the input file line-by-line, appending each row to one of N output
     files chosen uniformly at random. Memory use stays roughly constant.
  2. Load each (small) chunk into memory, shuffle in place, and rewrite it.

The result is a fully randomly-shuffled dataset spread across N CSV files,
each with the original header. Total disk I/O is ~2x the input size.

Assumptions:
  - The input CSV has a single header line.
  - No fields contain embedded newlines (true for typical ML/parallel-text CSVs).
    If your data has multi-line quoted fields, swap the raw line I/O for
    csv.reader / csv.writer.

Pick `num_chunks` so each chunk fits comfortably in RAM (e.g. file_size / num_chunks
is a few hundred MB). Note the OS file-handle limit: on macOS the default is 256,
so raise `ulimit -n` if `num_chunks` is large.

Usage:
  python data/shuffle_split_csv.py INPUT OUTPUT_DIR [-n NUM_CHUNKS] [-s SEED]

Example:
  python data/shuffle_split_csv.py data/en-fr.csv data/en-fr-shuffled -n 50
"""

import argparse
import random
from pathlib import Path


def shuffle_split_csv(
    input_path: str,
    output_dir: str,
    num_chunks: int,
    seed: int = 0,
) -> None:
    rng = random.Random(seed)
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_paths = [output_dir / f"chunk_{i:02d}.csv" for i in range(num_chunks)]

    # Pass 1: scatter rows into random buckets (streaming, ~constant memory).
    with open(input_path, "r", newline="", encoding="utf-8") as fin:
        header = fin.readline()
        writers = [open(p, "w", newline="", encoding="utf-8") for p in chunk_paths]
        try:
            for w in writers:
                w.write(header)
            for line in fin:
                writers[rng.randrange(num_chunks)].write(line)
        finally:
            for w in writers:
                w.close()

    # Pass 2: shuffle each chunk in memory and overwrite.
    for p in chunk_paths:
        with open(p, "r", newline="", encoding="utf-8") as f:
            header = f.readline()
            rows = f.readlines()
        rng.shuffle(rows)
        with open(p, "w", newline="", encoding="utf-8") as f:
            f.write(header)
            f.writelines(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shuffle a large CSV and split it into N smaller chunks.",
    )
    parser.add_argument("input", help="Path to the input CSV.")
    parser.add_argument(
        "output_dir", help="Directory to write chunk_XX.csv files into."
    )
    parser.add_argument(
        "-n",
        "--num-chunks",
        type=int,
        default=5,
        help="Number of output chunks (default: 5).",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0).",
    )
    args = parser.parse_args()

    shuffle_split_csv(
        input_path=args.input,
        output_dir=args.output_dir,
        num_chunks=args.num_chunks,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
