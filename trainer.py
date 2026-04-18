import logging
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, Field
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast

import wandb
from simple_translate import SimpleTranslate

WANDB_ENTITY = "pyron"
WANDB_PROJECT = "simple-translate"

COL_EN = "en"
COL_FR = "fr"

logger = logging.getLogger(__name__)


class TrainingConfig(BaseModel):
    """Every knob needed to run a training session, except the model and tokenizer.

    Pair this with a `(tokenizer, model)` produced by `flavors.load_flavor(...)`
    and hand both to `Trainer`.
    """

    # Data
    dataset_filename_train: Path
    dataset_filename_val: Path
    direction: Literal["en2fr", "fr2en"] = "en2fr"

    # Optimization
    batch_size: int = Field(64, gt=0)
    lr: float = Field(1e-3, gt=0)

    # Runtime
    save_dir: Path = Path("results")
    device: str | None = None  # auto-detected if None

    # Session schedule
    num_examples: int = Field(gt=0)
    log_every: int = Field(gt=0)
    eval_every: int = Field(gt=0)
    max_eval_examples: int | None = Field(None, gt=0)


class Trainer:
    def __init__(
        self,
        model: SimpleTranslate,
        tokenizer: PreTrainedTokenizerFast,
        config: TrainingConfig,
    ) -> None:
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "WANDB_API_KEY is not set. "
                "Run `wandb login` or export WANDB_API_KEY before instantiating Trainer."
            )
        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.source_column = COL_EN if config.direction == "en2fr" else COL_FR
        self.target_column = COL_FR if config.direction == "en2fr" else COL_EN
        self.optimizer = AdamW(self.model.parameters(), lr=config.lr)
        config.save_dir.mkdir(parents=True, exist_ok=True)
        self.examples_trained_on = 0
        self.best_loss = torch.inf

    def _stream_train_batches(self) -> Iterator[pd.DataFrame]:
        """Yield training batches indefinitely, restarting the CSV when exhausted."""
        # `pd.read_csv(..., chunksize=...)` returns a one-shot iterator that is
        # spent once the file is fully read. Wrapping it in `while True` reopens
        # the file for another pass, so this generator never terminates on its
        # own — the caller is responsible for stopping (e.g. via `break`).
        while True:
            reader = pd.read_csv(
                self.config.dataset_filename_train,
                header=0,
                chunksize=self.config.batch_size,
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

    def evaluate(self, max_examples: int | None = None) -> float:
        """
        Run over the validation set and return mean loss.

        If `max_examples` is provided, stop after seeing at least that many
        examples (useful for a fast, LLN-based estimate on large val sets).
        If `None`, evaluate on the full validation set.
        """
        self.model.eval()
        reader = pd.read_csv(
            self.config.dataset_filename_val,
            header=0,
            chunksize=self.config.batch_size,
        )
        losses = []
        examples_seen = 0
        for text_batch in reader:
            text_batch = text_batch.dropna(axis=0, how="any")
            tokens_source, tokens_destination = self.tokenize_batch(
                text_batch[self.source_column].tolist(),
                text_batch[self.target_column].tolist(),
            )
            loss = self.evaluate_one_batch(tokens_source, tokens_destination)
            losses.append(loss)
            examples_seen += tokens_source.shape[0]
            if max_examples is not None and examples_seen >= max_examples:
                break
        self.model.train()
        return float(np.mean(losses))

    def launch_session(self) -> None:
        """Train until `self.examples_trained_on >= self.config.num_examples`.

        Schedule and behavior are entirely controlled by `self.config`:
        - `log_every`: log running-mean training loss to W&B every this many examples.
        - `eval_every`: run an evaluation pass and log eval loss every this many
          examples. Whenever eval loss improves, checkpoint the model.
        - `max_eval_examples`: if set, cap each eval pass at this many examples
          (relies on LLN to approximate full-set loss). Otherwise use full val set.
        """
        cfg = self.config
        run_name = f"model_{cfg.direction}_{datetime.now().strftime('%Y%m%d_%H%M')}"
        wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=run_name,
            config=cfg.model_dump(mode="json"),
        )
        logger.info(
            f"Starting session {run_name} (target: {cfg.num_examples} examples)"
        )

        self.model.train()
        recent_losses: deque[float] = deque()
        next_log_at = self.examples_trained_on + cfg.log_every
        next_eval_at = self.examples_trained_on + cfg.eval_every
        start = time.time()

        try:
            for text_batch in self._stream_train_batches():
                tokens_source, tokens_destination = self.tokenize_batch(
                    text_batch[self.source_column].tolist(),
                    text_batch[self.target_column].tolist(),
                )
                batch_loss = self.train_one_batch(tokens_source, tokens_destination)
                recent_losses.append(batch_loss)

                if self.examples_trained_on >= next_log_at:
                    avg = sum(recent_losses) / len(recent_losses)
                    recent_losses.clear()
                    wandb.log({"train/loss": avg}, step=self.examples_trained_on)
                    next_log_at += cfg.log_every

                if self.examples_trained_on >= next_eval_at:
                    eval_loss = self.evaluate(max_examples=cfg.max_eval_examples)
                    wandb.log({"eval/loss": eval_loss}, step=self.examples_trained_on)
                    if eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        wandb.log(
                            {"eval/best_loss": self.best_loss},
                            step=self.examples_trained_on,
                        )
                        self.save(run_name)
                        logger.info(
                            f"Saved best model at {self.examples_trained_on} "
                            f"examples (eval loss {eval_loss:.3f})"
                        )
                    next_eval_at += cfg.eval_every

                if self.examples_trained_on >= cfg.num_examples:
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
            self.config.save_dir / f"{run_name}.pt",
        )
