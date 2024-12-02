import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import PreTrainedTokenizerFast

from simple_translate import SimpleTranslate


class Trainer:
    def __init__(
        self,
        model: SimpleTranslate,
        tokenizer: PreTrainedTokenizerFast,
        dataset_filename_train: str,
        dataset_filename_val: str,
        batch_size: int,
        lr: float,
        lr_min: float,
        T_0: int,
        T_mult: int,
        save_dir: str,
    ) -> None:
        self.model = model
        self.set_device("cpu")
        self.tokenizer = tokenizer
        self.dataset_filename_train = dataset_filename_train
        self.dataset_filename_val = dataset_filename_val
        self.batch_size = batch_size
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=T_0, T_mult=T_mult, eta_min=lr_min
        )
        self.save_dir = os.path.join(os.getcwd(), save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.examples_trained_on = 0
        self.best_loss = torch.inf
        self.loss_curve = list()
        self.birthday = datetime.now().strftime("%Y-%m-%dT%H_%M")

    def set_device(self, device: str) -> None:
        self.device = device
        self.model = self.model.to(device)

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

    def train_one_batch(
        self,
        tokens_source: list[str],
        tokens_destination: list[str],
    ) -> None:
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
        self.scheduler.step()
        self.examples_trained_on += self.batch_size

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
            return loss.cpu()

    def train_one_epoch(
        self,
        verbose: bool = False,
    ) -> None:
        self.model.train()
        dataset = pd.read_csv(
            self.dataset_filename_train,
            header=0,
            chunksize=self.batch_size,
        )
        for i, text_batch in enumerate(dataset):
            text_batch = text_batch.dropna(axis=0, how="any")
            tokens_source, tokens_destination = self.tokenize_batch(
                text_batch["en"].tolist(),
                text_batch["fr"].tolist(),
            )
            self.train_one_batch(tokens_source, tokens_destination)
            if verbose and i % 10 == 0:  # TODO: increase to 100
                print(f"Batch {i:6}  |  Num examples = {self.batch_size * i:9}")
            if i > 50:
                break  # TODO: delete after testing

    def evaluate_one_epoch(self) -> None:
        self.model.eval()
        dataset = pd.read_csv(
            self.dataset_filename_val,
            header=0,
            chunksize=self.batch_size,
        )
        loss_list = list()
        for i, text_batch in enumerate(dataset):
            text_batch = text_batch.dropna(axis=0, how="any")
            tokens_source, tokens_destination = self.tokenize_batch(
                text_batch["en"].tolist(),
                text_batch["fr"].tolist(),
            )
            loss = self.evaluate_one_batch(tokens_source, tokens_destination)
            loss_list.append(loss)
            if i > 50:
                break  # TODO: delete after testing
        self.loss_curve.append((self.examples_trained_on, np.array(loss_list).mean()))

    def launch_session(
        self,
        train_epochs: int,
        save_model: bool = True,
        verbose: bool = True,
    ):
        stopwatch = list()
        for i in range(train_epochs):

            start = time.time()
            print("Train epoch...")
            self.train_one_epoch(verbose)
            print(f"\tTime elapsed = {(time.time() - start)/60:.2f} min")
            print("Evaluation epoch...")
            self.evaluate_one_epoch()
            print(f"\tTime elapsed = {(time.time() - start)/60:.2f} min")
            stopwatch.append(time.time() - start)

            loss = self.loss_curve[-1][1]
            if loss < self.best_loss:
                self.best_loss = loss
                if save_model:
                    self.save()
            if verbose:
                print(
                    " | ".join(
                        [
                            f"Batches {self.examples_trained_on:5}",
                            f"Loss = {loss:6.3f}",
                            f"Stopwatch = {sum(stopwatch)/60:5.1f} min ({sum(stopwatch) / 60 / len(stopwatch):4.2f} min/epoch)",
                            f"lr = {self.scheduler._last_lr[0]:.2E}",
                            f"best_loss = {self.best_loss:6.3f}",
                        ]
                    )
                )

    def save(self):
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_dir, f"model_{self.birthday}.pt"),
        )
