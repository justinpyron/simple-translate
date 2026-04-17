import logging
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import wandb
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast

from simple_translate import SimpleTranslate

WANDB_ENTITY = "PLACEHOLDER_ENTITY"  # TODO: replace with real entity
WANDB_PROJECT = "PLACEHOLDER_PROJECT"  # TODO: replace with real project
DEFAULT_SAVE_DIR = Path("PLACEHOLDER_SAVE_DIR")  # TODO: replace with real default

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: SimpleTranslate,
        tokenizer: PreTrainedTokenizerFast,
        dataset_filename_train: str,
        dataset_filename_val: str,
        batch_size: int,
        lr: float,
        save_dir: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "WANDB_API_KEY is not set. "
                "Run `wandb login` or export WANDB_API_KEY before instantiating Trainer."
            )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.dataset_filename_train = dataset_filename_train
        self.dataset_filename_val = dataset_filename_val
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.save_dir = Path(save_dir) if save_dir is not None else DEFAULT_SAVE_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.examples_trained_on = 0
        self.best_loss = torch.inf

    # TODO: Add boolean switch for choosing en_2_fr or fr_2_en
    def _stream_train_batches(self) -> Iterator[pd.DataFrame]:
        """Yield training batches indefinitely, restarting the CSV when exhausted."""
        # `pd.read_csv(..., chunksize=...)` returns a one-shot iterator that is
        # spent once the file is fully read. Wrapping it in `while True` reopens
        # the file for another pass, so this generator never terminates on its
        # own — the caller is responsible for stopping (e.g. via `break`).
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
    ) -> None:
        """
        Train until `self.examples_trained_on >= num_examples`.

        - `log_every`: log running-mean training loss to W&B every this many examples.
        - `eval_every`: run a full evaluation pass and log eval loss to W&B every
          this many examples.
        """
        run_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M')}"
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=run_name,
            config={
                "batch_size": self.batch_size,
                "lr": self.lr,
                "num_examples": num_examples,
                "log_every": log_every,
                "eval_every": eval_every,
            },
        )
        logger.info(f"Starting session {run_name} (target: {num_examples} examples)")

        self.model.train()
        recent_losses: deque[float] = deque()
        next_log_at = self.examples_trained_on + log_every
        next_eval_at = self.examples_trained_on + eval_every
        start = time.time()

        try:
            for text_batch in self._stream_train_batches():
                tokens_source, tokens_destination = self.tokenize_batch(
                    text_batch["en"].tolist(),
                    text_batch["fr"].tolist(),
                )
                batch_loss = self.train_one_batch(tokens_source, tokens_destination)
                recent_losses.append(batch_loss)

                if self.examples_trained_on >= next_log_at:
                    avg = sum(recent_losses) / len(recent_losses)
                    recent_losses.clear()
                    wandb.log({"train/loss": avg}, step=self.examples_trained_on)
                    next_log_at += log_every

                if self.examples_trained_on >= next_eval_at:
                    eval_loss = self.evaluate()
                    wandb.log({"eval/loss": eval_loss}, step=self.examples_trained_on)
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        wandb.log(
                            {"eval/best_loss": self.best_loss},
                            step=self.examples_trained_on,
                        )
                        if save_best:
                            self.save(run_name)
                            logger.info(
                                f"Saved best model at {self.examples_trained_on} "
                                f"examples (eval loss {eval_loss:.3f})"
                            )
                    next_eval_at += eval_every

                if self.examples_trained_on >= num_examples:
                    break
        finally:
            elapsed_min = (time.time() - start) / 60
            logger.info(
                f"Finished session {run_name} "
                f"(elapsed {elapsed_min:.1f} min, best eval loss {self.best_loss:.3f})"
            )
            wandb.finish()

    def save(self, run_name: str) -> None:
        torch.save(
            self.model.state_dict(),
            self.save_dir / f"{run_name}.pt",
        )
