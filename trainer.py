import os
import time
from datetime import datetime

import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from simple_translate import SimpleTranslate

# TODO: handle switching to different devices
# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class Trainer:
    def __init__(
        self,
        model: SimpleTranslate,
        tokenizer: PreTrainedTokenizerFast,
        dataset_filename: str,
        batch_size: int,
        lr: float,
        lr_min: float,
        T_0: int,
        T_mult: int,
        save_dir: str,
    ) -> None:
        # self.model = model.to(DEVICE)
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_filename = dataset_filename
        self.batch_size = batch_size
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=T_0, T_mult=T_mult, eta_min=lr_min
        )
        self.save_dir = save_dir
        self.examples_trained_on = 0
        self.loss_curve = list()
        self.birthday = datetime.now().strftime("%Y-%m-%dT%H_%M")

    def prepare_batch(
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

    def train_one_epoch(
        self,
        verbose: bool = False,
    ) -> None:
        self.model.train()
        dataset = pd.read_csv(
            self.dataset_filename, header=0, chunksize=self.batch_size
        )
        for i, chunk in enumerate(dataset):
            chunk = chunk.dropna(axis=0, how="any")
            tokens_source, tokens_destination = self.prepare_batch(
                chunk["en"].tolist(),
                chunk["fr"].tolist(),
            )
            self.train_one_batch(tokens_source, tokens_destination)
            if verbose and i % 100 == 0:
                print(f"Batch {i:6}  |  Num examples = {self.batch_size * i:9}")

    # def evaluate_one_batch(
    #     self,
    #     text_source: list[str],
    #     text_destination: list[str],
    # ) -> float:
    #     tokens_source, tokens_destination = self.prepare_batch(
    #         text_source, text_destination
    #     )
    #     with torch.no_grad():
    #         loss = self.model(
    #             tokens_source=tokens_source,
    #             tokens_destination=tokens_destination[:, :-1],
    #             targets=tokens_destination[:, 1:],
    #         )
    #         return loss

    # def evaluate_one_epoch(self) -> None:
    #     self.model.eval()
    #     loss_list = list()
    #     for i, batch in enumerate(self.dataloader_val):
    #         loss = self.evaluate_one_batch(*batch)
    #         loss_list.append(loss)
    #         if i > 10:  # TODO: delete after testing
    #             break
    #     self.loss_curve.append(
    #         (self.examples_trained_on, torch.tensor(loss_list).mean().item())
    #     )

    # def launch_train_session(
    #     self,
    #     train_epochs: int,
    #     save_model: bool = True,
    #     verbose: bool = True,
    # ):
    #     best_loss = torch.inf
    #     stopwatch = list()
    #     for i in range(train_epochs):
    #         start = time.time()
    #         print("Train epoch...")
    #         self.train_one_epoch()
    #         print("Evaluation epoch...")
    #         self.evaluate_one_epoch()
    #         stopwatch.append(time.time() - start)
    #         loss = self.loss_curve[-1][1]
    #         if loss < best_loss and save_model:
    #             self.save()
    #             best_loss = loss
    #         if verbose:
    #             print(
    #                 " | ".join(
    #                     [
    #                         f"Batches {self.examples_trained_on:5}",
    #                         f"Loss = {loss:6.3f}",
    #                         f"Stopwatch = {sum(stopwatch)/60:5.1f} min ({sum(stopwatch) / len(stopwatch):4.2f} s/batch)",
    #                         f"lr = {self.scheduler._last_lr[0]:.2E}",
    #                     ]
    #                 )
    #             )

    def save(self):
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_dir, f"model_{self.birthday}.pt"),
        )