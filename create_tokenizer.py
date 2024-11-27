import time

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
from transformers import PreTrainedTokenizerFast

TOKEN_BOS = "<BOS>"
TOKEN_EOS = "<EOS>"
TOKEN_PAD = "<PAD>"


def get_corpus_generator(
    corpus_filename: str,
    chunksize: int,
):
    for chunk in pd.read_csv(corpus_filename, chunksize=chunksize):
        chunk = chunk.dropna(how="any", axis=0)
        en = chunk["en"].values.tolist()
        fr = chunk["fr"].values.tolist()
        corpus = " ".join(en + fr)
        yield corpus


def create_tokenizer(
    corpus_filename: str,
    vocab_size: int,
) -> None:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.NFD(),
            normalizers.Lowercase(),
            normalizers.Strip(),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{TOKEN_BOS} $A {TOKEN_EOS}",
        special_tokens=[(TOKEN_BOS, 0), (TOKEN_EOS, 1)],
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[TOKEN_BOS, TOKEN_EOS],
    )
    corpus_iterator = get_corpus_generator(corpus_filename, 50000)
    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)
    batch_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=TOKEN_BOS,
        eos_token=TOKEN_EOS,
        pad_token=TOKEN_PAD,
    )
    batch_tokenizer.save_pretrained(f"tokenizer_{vocab_size}", legacy_format=False)


@click.command()
@click.option(
    "filename",
    "-f",
    required=True,
    type=str,
    help="The filename of the dataset on which to train a tokenizer",
)
@click.option(
    "vocab_size",
    "-s",
    required=True,
    type=int,
    help="The vocabulary size of the tokenizer to train",
)
def main(filename, vocab_size):
    start = time.time()
    create_tokenizer(corpus_filename=filename, vocab_size=vocab_size)
    print(f"Tokenizer trained in {(time.time() - start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
