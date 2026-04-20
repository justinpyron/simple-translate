"""
Shuffles and splits large CSV files into training and validation sets.
Uses a memory-efficient two-pass indexing approach to handle files
that exceed available RAM.
"""
import argparse
import os
import random
import sys


def dataset_splitter():
    parser = argparse.ArgumentParser(
        description="Memory-efficiently shuffle and split a large CSV file."
    )
    parser.add_argument("-i", "--input_csv", help="Path to the source CSV file")
    parser.add_argument(
        "-top", "--train_output_path", help="Path to save the training set"
    )
    parser.add_argument(
        "-vop", "--val_output_path", help="Path to save the validation set"
    )
    parser.add_argument(
        "-ts",
        "--train_size",
        type=int,
        required=True,
        help="Number of rows for the training set",
    )
    parser.add_argument(
        "-vs",
        "--val_size",
        type=int,
        required=True,
        help="Number of rows for the validation set",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Error: Input file '{args.input_csv}' not found.")
        sys.exit(1)

    print(f"Indexing '{args.input_csv}'...")

    offsets = []
    header = None

    # Pass 1: Indexing
    with open(args.input_csv, "rb") as f:
        # Read the header first
        header = f.readline()
        if not header:
            print("Error: Input file is empty.")
            sys.exit(1)

        # Get start of first data row
        pos = f.tell()
        line = f.readline()
        while line:
            offsets.append(pos)
            pos = f.tell()
            line = f.readline()

    total_rows = len(offsets)
    print(f"Found {total_rows} data rows.")

    if args.train_size + args.val_size > total_rows:
        print(
            f"Error: Requested {args.train_size} (train) + {args.val_size} (val) = {args.train_size + args.val_size} rows, but only {total_rows} rows available."
        )
        sys.exit(1)

    # Shuffling
    print(f"Shuffling with seed {args.seed}...")
    random.seed(args.seed)
    random.shuffle(offsets)

    # Pass 2: Extraction
    print(
        f"Writing training set ({args.train_size} rows) to '{args.train_output_path}'..."
    )
    with open(args.input_csv, "rb") as f_in, open(
        args.train_output_path, "wb"
    ) as f_out:
        f_out.write(header)
        for i in range(args.train_size):
            f_in.seek(offsets[i])
            f_out.write(f_in.readline())

    print(
        f"Writing validation set ({args.val_size} rows) to '{args.val_output_path}'..."
    )
    with open(args.input_csv, "rb") as f_in, open(args.val_output_path, "wb") as f_out:
        f_out.write(header)
        for i in range(args.train_size, args.train_size + args.val_size):
            f_in.seek(offsets[i])
            f_out.write(f_in.readline())

    print("Done!")


if __name__ == "__main__":
    dataset_splitter()
