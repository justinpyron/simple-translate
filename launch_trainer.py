"""Launch a SimpleTranslate training session.

Example:
    python launch_trainer.py --flavor tiny \\
        --training-config training_configs/smoke.json \\
        --num-examples 10000 --lr 5e-4
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from flavors import FLAVORS, load_flavor
from trainer import Trainer, TrainingConfig

OVERRIDABLE_FIELDS: set[str] = set(TrainingConfig.model_fields)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch a SimpleTranslate training session.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--flavor",
        required=True,
        choices=sorted(FLAVORS),
        help="Named (tokenizer, model architecture) bundle defined in flavors.py.",
    )
    p.add_argument(
        "--training-config",
        type=Path,
        help="JSON file of TrainingConfig fields. CLI flags override values in it.",
    )
    p.add_argument(
        "--resume-from",
        type=Path,
        help="Optional checkpoint (.pt) to warm-start the model.",
    )

    # Per-field overrides mirror TrainingConfig. All optional; values are applied
    # on top of the JSON base only when explicitly provided on the CLI.
    p.add_argument("--dataset-filename-train", type=Path)
    p.add_argument("--dataset-filename-val", type=Path)
    p.add_argument("--direction", choices=["en2fr", "fr2en"])
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--save-dir", type=Path)
    p.add_argument("--device")
    p.add_argument("--num-examples", type=int)
    p.add_argument("--log-every", type=int)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--max-eval-examples", type=int)

    return p.parse_args()


def _build_training_config(args: argparse.Namespace) -> TrainingConfig:
    """Merge JSON file (if any) with explicit CLI overrides, then validate."""
    base: dict[str, Any] = {}
    if args.training_config is not None:
        base = json.loads(args.training_config.read_text())
    overrides = {
        k: v for k, v in vars(args).items() if k in OVERRIDABLE_FIELDS and v is not None
    }
    return TrainingConfig(**{**base, **overrides})


def main() -> None:
    args = _parse_args()
    config = _build_training_config(args)

    logging.basicConfig(
        filename=f"train_{args.flavor}_{datetime.now():%Y-%m-%dT%H_%M}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    tokenizer, model = load_flavor(args.flavor)
    if args.resume_from is not None:
        model.load_state_dict(torch.load(args.resume_from, map_location="cpu"))

    trainer = Trainer(model=model, tokenizer=tokenizer, config=config)
    trainer.launch_session()


if __name__ == "__main__":
    main()
