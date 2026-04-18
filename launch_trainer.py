"""Launch a SimpleTranslate training session."""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch

from flavors import FLAVORS, load_flavor
from trainer import Trainer, TrainingConfig


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Launch a SimpleTranslate training session.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--flavor", required=True, choices=sorted(FLAVORS))
    p.add_argument(
        "--resume-from",
        type=Path,
        help="Optional checkpoint (.pt) to warm-start the model.",
    )

    # Required TrainingConfig fields
    p.add_argument("--dataset-filename-train", type=Path, required=True)
    p.add_argument("--dataset-filename-val", type=Path, required=True)
    p.add_argument("--num-examples", type=int, required=True)
    p.add_argument("--log-every", type=int, required=True)
    p.add_argument("--eval-every", type=int, required=True)

    # Optional TrainingConfig fields (pydantic defaults apply when omitted)
    p.add_argument("--direction", choices=["en2fr", "fr2en"])
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--save-dir", type=Path)
    p.add_argument("--device")
    p.add_argument("--max-eval-examples", type=int)

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    config = TrainingConfig(
        **{
            k: v
            for k, v in vars(args).items()
            if k in TrainingConfig.model_fields and v is not None
        }
    )

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
