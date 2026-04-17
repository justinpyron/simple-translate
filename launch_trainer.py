import logging
from datetime import datetime

import torch

from model_configs import model_configs, model_configs_mini, tokenizer, tokenizer_mini
from simple_translate import SimpleTranslate
from trainer import Trainer

logging.basicConfig(
    filename=f"train_{datetime.now().strftime('%Y-%m-%dT%H_%M')}.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


model = SimpleTranslate(**model_configs)
model.load_state_dict(torch.load("results/model_2024-12-04T01_19.pt"))
simple_translate_trainer = Trainer(
    device="cuda",
    model=model,
    tokenizer=tokenizer,
    dataset_filename_train="en-fr-subset10M-shuffled-train.csv",
    dataset_filename_val="en-fr-subset10M-shuffled-val.csv",
    batch_size=64,
    lr=1e-3,
    save_dir="results",
)


if __name__ == "__main__":
    simple_translate_trainer.launch_session(
        num_examples=1_000_000,
        log_every=10_000,
        eval_every=100_000,
    )

# TODO: Write a shell script that launches a training session
