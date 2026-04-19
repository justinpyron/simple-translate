"""Launch a SimpleTranslate training session on Modal."""

import logging
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
DEFAULT_GPU = "A10"
DEFAULT_SAVE_DIR = "weights"

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
    config: TrainingConfig,
    flavor: str,
    tokenizer_dir_source: Path,
    tokenizer_dir_destination: Path,
    resume_from: str | None = None,
):
    from transformers import PreTrainedTokenizerFast

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Load config
    config.save_dir = Path(VOL_MOUNT_PATH) / config.save_dir
    config.dataset_filename_train = Path(VOL_MOUNT_PATH) / config.dataset_filename_train
    config.dataset_filename_val = Path(VOL_MOUNT_PATH) / config.dataset_filename_val

    # Load tokenizers
    tokenizer_source = PreTrainedTokenizerFast.from_pretrained(
        Path(VOL_MOUNT_PATH) / tokenizer_dir_source
    )
    tokenizer_destination = PreTrainedTokenizerFast.from_pretrained(
        Path(VOL_MOUNT_PATH) / tokenizer_dir_destination
    )

    # Load checkpoint
    checkpoint = Path(VOL_MOUNT_PATH) / resume_from if resume_from else None
    if checkpoint:
        logging.info(f"Loading checkpoint from {checkpoint}")
    model = FLAVORS[flavor].load(
        tokenizer_source, tokenizer_destination, checkpoint=checkpoint
    )
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameter count: {num_params:,}")

    # Launch training session
    trainer = Trainer(
        model=model,
        tokenizer_source=tokenizer_source,
        tokenizer_destination=tokenizer_destination,
        config=config,
    )
    trainer.launch_session()
    volume.commit()


# =============================================================================
# Local Entrypoint
# =============================================================================


@app.local_entrypoint()
def main(
    flavor: str,
    tokenizer_dir_source: str,
    tokenizer_dir_destination: str,
    dataset_filename_train: str,
    dataset_filename_val: str,
    direction: str,
    num_examples: int,
    log_every: int,
    eval_every: int,
    batch_size: int,
    lr_start: float,
    lr_end: float,
    warmup_steps: int,
    save_dir: str = DEFAULT_SAVE_DIR,
    resume_from: str = None,
    max_eval_examples: int = None,
):
    """Launch SimpleTranslate training on Modal."""
    if flavor not in FLAVORS:
        raise ValueError(
            f"Unknown flavor {flavor!r}. Available flavors: {sorted(FLAVORS)}"
        )

    cfg = TrainingConfig(
        dataset_filename_train=dataset_filename_train,
        dataset_filename_val=dataset_filename_val,
        direction=direction,
        num_examples=num_examples,
        log_every=log_every,
        eval_every=eval_every,
        batch_size=batch_size,
        lr_start=lr_start,
        lr_end=lr_end,
        warmup_steps=warmup_steps,
        save_dir=save_dir,
        max_eval_examples=max_eval_examples,
    )

    print(f"\n🚀 Training {flavor} on Modal ({DEFAULT_GPU})")
    print(f"   Tokenizer (source):      {tokenizer_dir_source}")
    print(f"   Tokenizer (destination): {tokenizer_dir_destination}")
    if resume_from:
        print(f"   Resume:    {resume_from}")
    print("\n   Config:")
    for k, v in cfg.model_dump().items():
        print(f"     {k:<25} {v}")
    print()

    train.remote(
        config=cfg,
        flavor=flavor,
        tokenizer_dir_source=tokenizer_dir_source,
        tokenizer_dir_destination=tokenizer_dir_destination,
        resume_from=resume_from,
    )
