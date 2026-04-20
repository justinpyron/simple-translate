"""Modal-based inference server for SimpleTranslate model."""

from pathlib import Path

import modal

from interfaces import TranslationDirection

# =============================================================================
# Deployment configuration
# =============================================================================

VOLUME_NAME = "simple-translate"
VOL_MOUNT_PATH = Path("/data")
FLAVOR = "small"
TOKENIZER_SOURCE_PATH = Path("tokenizers/en-vocab_1000")
TOKENIZER_DESTINATION_PATH = Path("tokenizers/fr-vocab_1000")
WEIGHTS_EN = Path("weights_en.pt")
WEIGHTS_FR = Path("weights_fr.pt")

app = modal.App("simple-translate")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "numpy==2.1.3",
        "pydantic==2.10.4",
        "fastapi==0.124.0",
    )
    .add_local_python_source("interfaces")
    .add_local_python_source("flavors")
    .add_local_python_source("simple_translate")
)
volume = modal.Volume.from_name(VOLUME_NAME)


@app.cls(
    image=image,
    volumes={str(VOL_MOUNT_PATH): volume},
    cpu=1,
    scaledown_window=600,
)
class Server:
    """Modal class for serving SimpleTranslate model inference."""

    @modal.enter()
    def load_model(self):
        """Load tokenizers and both direction-specific models on container startup."""
        from transformers import PreTrainedTokenizerFast

        from flavors import FLAVORS

        tokenizer_en = PreTrainedTokenizerFast.from_pretrained(
            VOL_MOUNT_PATH / TOKENIZER_SOURCE_PATH
        )
        tokenizer_fr = PreTrainedTokenizerFast.from_pretrained(
            VOL_MOUNT_PATH / TOKENIZER_DESTINATION_PATH
        )

        if FLAVOR not in FLAVORS:
            raise ValueError(f"Unknown flavor {FLAVOR!r}. Available: {sorted(FLAVORS)}")

        flavor = FLAVORS[FLAVOR]
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
        self.models = {
            "en2fr": flavor.load(
                tokenizer_en,
                tokenizer_fr,
                checkpoint=VOL_MOUNT_PATH / WEIGHTS_EN,
            ),
            "fr2en": flavor.load(
                tokenizer_fr,
                tokenizer_en,
                checkpoint=VOL_MOUNT_PATH / WEIGHTS_FR,
            ),
        }
        for m in self.models.values():
            m.eval()

    def translate(
        self,
        text_source: str,
        direction: TranslationDirection,
        temperature: float | None = None,
        beams: int | None = None,
    ) -> str:
        """
        Generate a translation for one input string.

        Args:
            text_source: Text in the source language for the chosen direction
            direction: ``en2fr`` (English → French) or ``fr2en`` (French → English)
            temperature: Sampling temperature when using temperature-based generation
            beams: Beam width when using beam search

        Returns:
            Translated text in the target language
        """
        model = self.models[direction]
        tokenizer_source, tokenizer_destination = (
            (self.tokenizer_en, self.tokenizer_fr)
            if direction == "en2fr"
            else (self.tokenizer_fr, self.tokenizer_en)
        )

        tokens_source = tokenizer_source(
            text_source,
            truncation=True,
            max_length=model.max_sequence_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )["input_ids"]

        if temperature is not None:
            temperature = max(1e-3, temperature)
            tokens_destination = model.generate_with_temp(
                tokens_source, temperature=temperature
            )
        elif beams is not None:
            tokens_destination = model.generate_with_beams(
                tokens_source, beam_width=beams
            )
        else:
            tokens_destination = model.generate_with_temp(
                tokens_source, temperature=0.1
            )

        translation = tokenizer_destination.decode(
            tokens_destination[0], skip_special_tokens=True
        )

        return translation

    @modal.asgi_app()
    def fastapi_server(self):
        """Create and configure the FastAPI application."""
        from fastapi import FastAPI

        from interfaces import TranslateRequest, TranslateResponse

        server = FastAPI(title="SimpleTranslate API")

        @server.post("/translate", response_model=TranslateResponse)
        def translate_endpoint(request: TranslateRequest) -> TranslateResponse:
            translation = self.translate(
                text_source=request.text_source,
                direction=request.direction,
                temperature=request.temperature,
                beams=request.beams,
            )
            return TranslateResponse(translation=translation)

        @server.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy"}

        return server
