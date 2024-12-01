import os
import time
from datetime import datetime

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
        dataloader_train: DataLoader,
        dataloader_val: DataLoader,
        lr: float,
        lr_min: float,
        T_0: int,
        T_mult: int,
        save_dir: str,
    ) -> None:
        # self.model = model.to(DEVICE)
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=T_0, T_mult=T_mult, eta_min=lr_min
        )
        self.save_dir = save_dir
        self.batches_trained_on = 0
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
            return_token_type_ids=False,
            return_tensors="pt",
        )["input_ids"]
        tokens_destination = self.tokenizer(
            text_destination,
            padding=True,
            truncation=True,
            max_length=self.model.max_sequence_length,
            return_token_type_ids=False,
            return_tensors="pt",
        )["input_ids"]
        return tokens_source, tokens_destination

    def train_one_batch(
        self,
        text_source: list[str],
        text_destination: list[str],
    ) -> None:
        tokens_source, tokens_destination = self.prepare_batch(
            text_source, text_destination
        )
        loss = self.model(
            tokens_source=tokens_source,
            tokens_destination=tokens_destination[:, :-1],
            targets=tokens_destination[:, 1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        print(loss)  # TODO: delete

    def train_one_epoch(self) -> None:
        pass

    # def evaluate(
    #     self,
    #     num_batches_evaluate : int,
    #     from_val : bool = True,
    # ) -> torch.tensor:
    #     """Compute average loss on a handful of batches"""
    #     self.model.eval()
    #     loss_samples = list()
    #     for i in range(num_batches_evaluate):
    #         x, y = self.dataloader.get_batch(from_val=from_val)
    #         with torch.no_grad():
    #             loss = self.model(x, y)
    #             loss_samples.append(loss)
    #     self.model.train()
    #     return torch.tensor(loss_samples).mean().item()

    # def launch_train_session(
    #     self,
    #     num_batches : int,
    #     evaluate_every : int,
    #     save_model: bool = True,
    #     verbose : bool = True,
    # ):
    #     self.model.train()
    #     best_loss = torch.inf
    #     stopwatch = list()
    #     for i in range(num_batches):
    #         start = time.time()
    #         self.train_on_one_batch()
    #         stopwatch.append(time.time() - start)
    #         if i % evaluate_every == 0:
    #             loss = self.evaluate(self.num_batches_evaluate)
    #             self.loss_curve.append((self.batches_trained_on, loss))
    #             if loss < best_loss and save_model:
    #                 self.save()
    #                 best_loss = loss
    #             if verbose:
    #                 print(" | ".join([
    #                     f"Batch {self.batches_trained_on:5}",
    #                     f"Loss = {loss:6.3f}",
    #                     f"Stopwatch = {sum(stopwatch)/60:5.1f} min ({sum(stopwatch) / len(stopwatch):4.2f} s/batch)",
    #                     f"lr = {self.scheduler._last_lr[0]:.2E}",
    #                 ]))
    #         self.batches_trained_on += 1

    def save(self):
        torch.save(
            self.model.state_dict(),
            os.path.join(self.save_dir, f"model_{self.birthday}.pt"),
        )
