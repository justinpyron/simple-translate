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

APP_NAME = "simple-translate-train"
VOLUME_NAME = "simple-translate-vol"
VOLUME_MOUNT_PATH = "/vol"
DEFAULT_GPU = "A10G"

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
    .add_local_dir(
        ".",
        remote_path="/root",
        ignore_list=[".git", "__pycache__", "wandb", "weights", "results", "notebooks"],
    )
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


# =============================================================================
# Training Function
# =============================================================================


@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: volume},
    gpu=DEFAULT_GPU,
    timeout=24 * 60 * 60,  # 24 hours
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(config_dict: dict, flavor: str, resume_from: str = None):
    # Ensure we are in the root directory where the code was added
    os.chdir("/root")

    # Reconstruct the config. All paths are relative to /root or the volume.
    config = TrainingConfig(**config_dict)

    # If save_dir is relative, redirect it to the volume
    if not config.save_dir.is_absolute():
        config.save_dir = Path(VOLUME_MOUNT_PATH) / config.save_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    tokenizer, model = load_flavor(flavor)

    if resume_from is not None:
        resume_path = Path(resume_from)
        # Try finding the checkpoint in various locations
        possible_paths = [
            resume_path,
            Path("/root") / resume_path,
            Path(VOLUME_MOUNT_PATH) / resume_path,
        ]
        found = False
        for p in possible_paths:
            if p.exists():
                logging.info(f"Loading checkpoint from {p}")
                model.load_state_dict(torch.load(p, map_location="cpu"))
                found = True
                break
        if not found:
            logging.warning(f"Could not find checkpoint at {resume_from}")

    trainer = Trainer(model=model, tokenizer=tokenizer, config=config)
    trainer.launch_session()

    # Commit the volume to ensure weights are saved
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
