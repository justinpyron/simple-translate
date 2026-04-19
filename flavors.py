"""
Flavors define named SimpleTranslate architectures, in the spirit of
HuggingFace's `AutoModel` pattern.

A `Flavor` is a pure architecture description: it knows the shape of the model
but not the tokenizer. Constructing a usable `SimpleTranslate` requires pairing
a flavor with a tokenizer via `Flavor.load(...)`.

Adding a new flavor is a matter of adding an entry to `FLAVORS` below.
"""

from pathlib import Path

from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerFast

from simple_translate import SimpleTranslate


class Flavor(BaseModel):
    """Pure architecture description for a SimpleTranslate variant."""

    name: str
    max_sequence_length: int = Field(gt=0)
    dim_embedding: int = Field(gt=0)
    dim_head: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    dim_mlp: int = Field(gt=0)
    dropout: float = Field(ge=0, le=1)
    num_blocks: int = Field(gt=0)

    def load(
        self,
        tokenizer: PreTrainedTokenizerFast,
        checkpoint: str | Path | None = None,
    ) -> SimpleTranslate:
        """Build a model for this flavor, optionally loading pretrained weights.

        When `checkpoint` is `None`, returns a freshly-initialized model.
        Otherwise, loads the state dict at `checkpoint` into a model sized for
        `tokenizer`.
        """
        kwargs = dict(
            vocab_size=tokenizer.vocab_size,
            token_id_bos=tokenizer.bos_token_id,
            token_id_eos=tokenizer.eos_token_id,
            token_id_pad=tokenizer.pad_token_id,
            **self.model_dump(),
        )
        if checkpoint is None:
            return SimpleTranslate(**kwargs)
        return SimpleTranslate.from_pretrained(checkpoint, **kwargs)


FLAVORS: dict[str, Flavor] = {
    "tiny": Flavor(
        name="tiny",
        max_sequence_length=256,
        dim_embedding=32,
        dim_head=8,
        num_heads=4,
        dim_mlp=64,
        dropout=0.1,
        num_blocks=4,
    ),
    "mini": Flavor(
        name="mini",
        max_sequence_length=512,
        dim_embedding=128,
        dim_head=16,
        num_heads=4,
        dim_mlp=256,
        dropout=0.1,
        num_blocks=8,
    ),
}
