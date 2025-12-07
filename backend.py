"""Modal-based inference server for SimpleTranslate model."""

import modal
import torch
from fastapi import FastAPI
from transformers import PreTrainedTokenizerFast

from interfaces import TranslateRequest, TranslateResponse
from simple_translate import SimpleTranslate

# Create Modal app
app = modal.App("simple-translate")

# Define the Modal image with all required dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.5.1",
    "transformers==4.46.3",
    "numpy==2.1.3",
)

# Reference to the Modal volume containing model weights and tokenizer
volume = modal.Volume.from_name("simple-translate")


@app.cls(
    image=image,
    volumes={"/data": volume},
    cpu=2.0,
)
class SimpleTranslateServer:
    """Modal class for serving SimpleTranslate model inference."""

    @modal.enter()
    def load_model(self):
        """Load model and tokenizer on container startup."""
        # Load tokenizer from volume
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "/data/tokenizer_1000/"
        )

        # Prepare model configs
        model_configs = {
            "vocab_size": self.tokenizer.vocab_size,
            "max_sequence_length": 256,
            "dim_embedding": 128,
            "dim_head": 16,
            "num_heads": 8,
            "dim_mlp": 256,
            "dropout": 0.1,
            "num_blocks": 4,
            "token_id_bos": self.tokenizer.bos_token_id,
            "token_id_eos": self.tokenizer.eos_token_id,
            "token_id_pad": self.tokenizer.pad_token_id,
        }

        # Load model from volume
        self.model = SimpleTranslate(**model_configs)
        self.model.load_state_dict(
            torch.load(
                "/data/model_for_app_cpu.pt", weights_only=True, map_location="cpu"
            )
        )
        self.model.eval()

    @modal.method()
    def translate(
        self,
        text_source: str,
        temperature: float | None = None,
        beams: int | None = None,
    ) -> str:
        """
        Generate translation for a single input example.

        Args:
            text_source: The English text to translate
            temperature: Temperature for sampling (if using temperature-based generation)
            beams: Number of beams for beam search (if using beam search)

        Returns:
            The translated French text
        """
        # Tokenize input
        tokens_source = self.tokenizer(
            text_source,
            truncation=True,
            max_length=self.model.max_sequence_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )["input_ids"]

        # Generate translation based on strategy
        if temperature is not None:
            temperature = max(1e-3, temperature)  # Ensure temperature is positive
            tokens_destination = self.model.generate_with_temp(
                tokens_source, temperature=temperature
            )
        elif beams is not None:
            tokens_destination = self.model.generate_with_beams(
                tokens_source, beam_width=beams
            )
        else:
            # Default to temperature-based with low temperature
            tokens_destination = self.model.generate_with_temp(
                tokens_source, temperature=0.1
            )

        # Decode and return translation
        translation = self.tokenizer.decode(
            tokens_destination[0], skip_special_tokens=True
        )
        return translation


# Create FastAPI app
web_app = FastAPI(title="SimpleTranslate API")


@web_app.post("/translate", response_model=TranslateResponse)
async def translate_endpoint(request: TranslateRequest) -> TranslateResponse:
    """
    Translate English text to French.

    Args:
        request: Translation request with source text and generation parameters

    Returns:
        Translation response with the translated text
    """
    server = SimpleTranslateServer()
    translation = server.translate.remote(
        text_source=request.text_source,
        temperature=request.temperature,
        beams=request.beams,
    )
    return TranslateResponse(translation=translation)


@web_app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Expose the FastAPI app via Modal
@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app
