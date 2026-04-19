"""Launch a SimpleTranslate training session on Modal."""

import logging
import os
from pathlib import Path

import modal

from flavors import FLAVORS
from trainer import Trainer, TrainingConfig

# =============================================================================
# Modal Configuration
# =============================================================================

APP_NAME = "simple-translate"
VOL_NAME = "simple-translate"
VOL_MOUNT_PATH = "/vol"
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
    .add_local_python_source("simple_translate", "trainer", "flavors")
)

volume = modal.Volume.from_name(VOL_NAME)


# =============================================================================
# Training Function
# =============================================================================


@app.function(
    image=image,
    volumes={VOL_MOUNT_PATH: volume},
    gpu=DEFAULT_GPU,
    timeout=24 * 60 * 60,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train(
    config_dict: dict,
    flavor: str,
    tokenizer_dir: Path,
    resume_from: str | None = None,
):
    from transformers import PreTrainedTokenizerFast

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load config
    cfg = TrainingConfig(**config_dict)
    cfg.save_dir = Path(VOL_MOUNT_PATH) / cfg.save_dir
    cfg.dataset_filename_train = Path(VOL_MOUNT_PATH) / cfg.dataset_filename_train
    cfg.dataset_filename_val = Path(VOL_MOUNT_PATH) / cfg.dataset_filename_val

    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        Path(VOL_MOUNT_PATH) / tokenizer_dir
    )

    # Load checkpoint
    checkpoint = Path(VOL_MOUNT_PATH) / resume_from if resume_from else None
    if checkpoint:
        logging.info(f"Loading checkpoint from {checkpoint}")
    model = FLAVORS[flavor].load(tokenizer, checkpoint=str(checkpoint))

    # Launch training session
    trainer = Trainer(model=model, tokenizer=tokenizer, config=cfg)
    trainer.launch_session()
    volume.commit()


# =============================================================================
# Local Entrypoint
# =============================================================================


@app.local_entrypoint()
def main(
    flavor: str,
    tokenizer_dir: str,
    dataset_filename_train: str,
    dataset_filename_val: str,
    direction: str,
    num_examples: int,
    log_every: int,
    eval_every: int,
    resume_from: str = None,
    batch_size: int = 64,
    lr: float = 1e-3,
    save_dir: str = "weights",
    gpu: str = DEFAULT_GPU,
    max_eval_examples: int = None,
):
    """Launch SimpleTranslate training on Modal."""
    if flavor not in FLAVORS:
        raise ValueError(
            f"Unknown flavor {flavor!r}. Available flavors: {sorted(FLAVORS)}"
        )

    config_dict = {
        "dataset_filename_train": dataset_filename_train,
        "dataset_filename_val": dataset_filename_val,
        "direction": direction,
        "num_examples": num_examples,
        "log_every": log_every,
        "eval_every": eval_every,
        "batch_size": batch_size,
        "lr": lr,
        "save_dir": save_dir,
        "max_eval_examples": max_eval_examples,
    }

    print("=" * 80)
    print(f"Launching SimpleTranslate training on Modal ({gpu})...")
    print(f"  Flavor: {flavor}")
    print(f"  Tokenizer dir: {tokenizer_dir} (in volume {VOL_NAME})")
    for k, v in config_dict.items():
        print(f"  {k}: {v}")
    print("=" * 80)

    train.with_options(gpu=gpu).remote(
        config_dict=config_dict,
        flavor=flavor,
        tokenizer_dir=tokenizer_dir,
        resume_from=resume_from,
    )
