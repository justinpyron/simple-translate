from simple_translate import SimpleTranslate
from model_configs import tokenizer, tokenizer_mini, model_configs, model_configs_mini
from trainer import Trainer
from datetime import datetime
import logging
import torch

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
    simple_translate_trainer.launch_session(train_epochs=1)
