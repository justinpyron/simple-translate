import logging
import os
import time
from collections import deque
from datetime import datetime
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast

from simple_translate import SimpleTranslate

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        device: str,
        model: SimpleTranslate,
        tokenizer: PreTrainedTokenizerFast,
        dataset_filename_train: str,
        dataset_filename_val: str,
        batch_size: int,
        lr: float,
        save_dir: str,
    ) -> None:
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.dataset_filename_train = dataset_filename_train
        self.dataset_filename_val = dataset_filename_val
        self.batch_size = batch_size
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.save_dir = os.path.join(os.getcwd(), save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.examples_trained_on = 0
        self.best_loss = torch.inf
        self.loss_log_train: list[tuple[int, float]] = []
        self.loss_log_eval: list[tuple[int, float]] = []
        self.birthday = datetime.now().strftime("%Y%m%d_%H%M")
        # TODO: Make save_dir have a default given by a global variable
        # TODO: Possible remove birthday and other unnecessary attributes if WandB logging gives you equivalents for free
        # TODO: Don't make device an arg; instead, set it to "cuda" if CUDA is available, otherwise "cpu"

    # TODO: Add boolean switch for choosing en_2_fr or fr_2_en
    def _stream_train_batches(self) -> Iterator[pd.DataFrame]:
        """Yield training batches indefinitely, restarting the CSV when exhausted."""
        while True:
            reader = pd.read_csv(
                self.dataset_filename_train,
                header=0,
                chunksize=self.batch_size,
            )
            for text_batch in reader:
                yield text_batch.dropna(axis=0, how="any")

    def tokenize_batch(
        self,
        text_source: list[str],
        text_destination: list[str],
    ) -> tuple[torch.tensor, torch.tensor]:
        tokens_source = self.tokenizer(
            text_source,
            padding=True,
            truncation=True,
            max_length=self.model.max_sequence_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )["input_ids"]
        tokens_destination = self.tokenizer(
            text_destination,
            padding=True,
            truncation=True,
            max_length=self.model.max_sequence_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )["input_ids"]
        return tokens_source, tokens_destination
        # TODO: Train two tokenizers: one for source and one for destination?

    def train_one_batch(
        self,
        tokens_source: list[str],
        tokens_destination: list[str],
    ) -> float:
        self.model.train()
        tokens_source = tokens_source.to(self.device)
        tokens_destination = tokens_destination.to(self.device)
        loss = self.model(
            tokens_source=tokens_source,
            tokens_destination=tokens_destination[:, :-1],
            targets=tokens_destination[:, 1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.examples_trained_on += tokens_source.shape[0]
        return loss.item()

    def evaluate_one_batch(
        self,
        tokens_source: list[str],
        tokens_destination: list[str],
    ) -> float:
        tokens_source = tokens_source.to(self.device)
        tokens_destination = tokens_destination.to(self.device)
        with torch.no_grad():
            loss = self.model(
                tokens_source=tokens_source,
                tokens_destination=tokens_destination[:, :-1],
                targets=tokens_destination[:, 1:],
            )
            return loss.item()

    def evaluate(self) -> float:
        """Run a full pass over the validation set and return mean loss."""
        self.model.eval()
        reader = pd.read_csv(
            self.dataset_filename_val,
            header=0,
            chunksize=self.batch_size,
        )
        losses = []
        for text_batch in reader:
            text_batch = text_batch.dropna(axis=0, how="any")
            tokens_source, tokens_destination = self.tokenize_batch(
                text_batch["en"].tolist(),
                text_batch["fr"].tolist(),
            )
            loss = self.evaluate_one_batch(tokens_source, tokens_destination)
            losses.append(loss)
        self.model.train()
        return float(np.mean(losses))

    def launch_session(
        self,
        num_examples: int,
        log_every: int,
        eval_every: int,
        save_best: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Train for roughly `num_examples` examples.

        - `log_every`: record running-mean training loss every this many examples.
        - `eval_every`: run a full evaluation pass every this many examples.
        """
        self.model.train()
        recent_losses: deque[float] = deque()
        next_log_at = self.examples_trained_on + log_every
        next_eval_at = self.examples_trained_on + eval_every
        start = time.time()

        for text_batch in self._stream_train_batches():
            tokens_source, tokens_destination = self.tokenize_batch(
                text_batch["en"].tolist(),
                text_batch["fr"].tolist(),
            )
            batch_loss = self.train_one_batch(tokens_source, tokens_destination)
            recent_losses.append(batch_loss)

            if self.examples_trained_on >= next_log_at:
                avg = sum(recent_losses) / len(recent_losses)
                self.loss_log_train.append((self.examples_trained_on, avg))
                recent_losses.clear()
                if verbose:
                    logger.info(
                        f"[train] ex={self.examples_trained_on:>10} " f"loss={avg:6.3f}"
                    )
                next_log_at += log_every

            if self.examples_trained_on >= next_eval_at:
                eval_loss = self.evaluate()
                self.loss_log_eval.append((self.examples_trained_on, eval_loss))
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    if save_best:
                        self.save()
                if verbose:
                    elapsed_min = (time.time() - start) / 60
                    logger.info(
                        f"[eval ] ex={self.examples_trained_on:>10} "
                        f"loss={eval_loss:6.3f} best={self.best_loss:6.3f} "
                        f"elapsed={elapsed_min:5.1f}min"
                    )
                next_eval_at += eval_every

            if self.examples_trained_on >= num_examples:
                break

    # TODO: log results to WandB? Ideally, we could print GPU memory usage, etc. as well

    def save(self) -> None:
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_dir, f"model_{self.birthday}.pt"),
        )
