"""
Flavors bundle a tokenizer and a SimpleTranslate architecture under a single
name, in the spirit of HuggingFace's `AutoModel` / `AutoTokenizer` pattern.

Adding a new flavor is a matter of adding an entry to `FLAVORS` below.
"""

from pathlib import Path

from pydantic import BaseModel, Field
from transformers import PreTrainedTokenizerFast

from simple_translate import SimpleTranslate


class Flavor(BaseModel):
    """A (tokenizer, model architecture) bundle identified by `name`."""

    model_config = {"extra": "forbid"}

    name: str
    tokenizer_dir: Path
    max_sequence_length: int = Field(gt=0)
    dim_embedding: int = Field(gt=0)
    dim_head: int = Field(gt=0)
    num_heads: int = Field(gt=0)
    dim_mlp: int = Field(gt=0)
    dropout: float = Field(ge=0, le=1)
    num_blocks: int = Field(gt=0)

    def load(self) -> tuple[PreTrainedTokenizerFast, SimpleTranslate]:
        """Instantiate the tokenizer and a fresh (randomly-initialized) model."""
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(self.tokenizer_dir))
        model = SimpleTranslate(
            vocab_size=tokenizer.vocab_size,
            token_id_bos=tokenizer.bos_token_id,
            token_id_eos=tokenizer.eos_token_id,
            token_id_pad=tokenizer.pad_token_id,
            **self.model_dump(exclude={"name", "tokenizer_dir"}),
        )
        return tokenizer, model

    # TODO: Add three separate loading methods? --> load_tokenizer, load_model, load_flavors?
    #       Add an arg with default None, which initializes randomly from 0.
    #       If not None, it's a path to weights --> load the model from these weights.
    # TODO: Is it possible to initialize a model without initializing weights?
    #       Would be nice for pre-trained models to directly load them,
    #       instead of init with random then load weights.


FLAVORS: dict[str, Flavor] = {
    "tiny": Flavor(
        name="tiny",
        tokenizer_dir=Path("tokenizer_1000"),
        max_sequence_length=256,
        dim_embedding=128,
        dim_head=16,
        num_heads=8,
        dim_mlp=256,
        dropout=0.1,
        num_blocks=4,
    ),
}


def load_flavor(name: str) -> tuple[PreTrainedTokenizerFast, SimpleTranslate]:
    """Load the tokenizer and model associated with a flavor name."""
    if name not in FLAVORS:
        raise ValueError(
            f"Unknown flavor {name!r}. Available flavors: {sorted(FLAVORS)}"
        )
    return FLAVORS[name].load()
