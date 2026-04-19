"""Launch a SimpleTranslate training session on Modal."""

import logging
import os
from pathlib import Path

import modal
import torch

from flavors import FLAVORS, load_flavor
from trainer import Trainer, TrainingConfig

# =============================================================================
# Modal Configuration
# =============================================================================

APP_NAME = "simple-translate-train"  # TODO: Update
VOLUME_NAME = "simple-translate-vol"  # TODO: Update
VOLUME_MOUNT_PATH = "/vol"  # TODO: Update
DEFAULT_GPU = "A10G"  # TODO: Update

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers",
        "pandas",
        "numpy",
        "wandb",
        "pydantic",
        "tokenizers",
    )
    .add_local_python_source(
        "trainer", "flavors", "simple_translate", "model_configs", "interfaces"
    )
    .add_local_dir("tokenizer_1000", "/root/tokenizer_1000")
    .add_local_dir("tokenizer_2000", "/root/tokenizer_2000")
    .add_local_dir("tokenizer_0500", "/root/tokenizer_0500")
)

volume = modal.Volume.from_name(VOLUME_NAME)


# =============================================================================
# Training Function
# =============================================================================


@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: volume},
    gpu=DEFAULT_GPU,
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(config_dict: dict, flavor: str, resume_from: str = None):
    os.chdir("/root")
    config = TrainingConfig(**config_dict)

    # Assume data and weights live in the volume
    config.save_dir = Path(VOLUME_MOUNT_PATH) / config.save_dir
    config.dataset_filename_train = (
        Path(VOLUME_MOUNT_PATH) / config.dataset_filename_train
    )
    config.dataset_filename_val = Path(VOLUME_MOUNT_PATH) / config.dataset_filename_val

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    tokenizer, model = load_flavor(flavor)

    if resume_from:
        logging.info(f"Loading checkpoint from {resume_from}")
        model.load_state_dict(torch.load(resume_from, map_location="cpu"))

    trainer = Trainer(model=model, tokenizer=tokenizer, config=config)
    trainer.launch_session()
    volume.commit()


# =============================================================================
# Local Entrypoint
# =============================================================================


@app.local_entrypoint()
def main(
    flavor: str,
    dataset_filename_train: str,
    dataset_filename_val: str,
    num_examples: int,
    log_every: int,
    eval_every: int,
    resume_from: str = None,
    direction: str = "en2fr",
    batch_size: int = 64,
    lr: float = 1e-3,
    save_dir: str = "weights",
    gpu: str = DEFAULT_GPU,
    max_eval_examples: int = None,
):
    """Launch SimpleTranslate training on Modal."""
    config_dict = {
        "dataset_filename_train": dataset_filename_train,
        "dataset_filename_val": dataset_filename_val,
        "num_examples": num_examples,
        "log_every": log_every,
        "eval_every": eval_every,
        "direction": direction,
        "batch_size": batch_size,
        "lr": lr,
        "save_dir": save_dir,
        "max_eval_examples": max_eval_examples,
    }

    print("=" * 80)
    print(f"Launching SimpleTranslate training on Modal ({gpu})...")
    print(f"  Flavor: {flavor}")
    print(f"  Train data: {dataset_filename_train}")
    print(f"  Val data: {dataset_filename_val}")
    print(f"  Num examples: {num_examples}")
    print(f"  Save dir: {save_dir} (in volume {VOLUME_NAME})")
    print("=" * 80)

    train.with_options(gpu=gpu).remote(
        config_dict=config_dict,
        flavor=flavor,
        resume_from=resume_from,
    )
