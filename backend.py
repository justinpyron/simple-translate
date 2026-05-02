"""Modal-based inference server for SimpleTranslate model."""

from pathlib import Path

import modal

from schemas import TranslationDirection

# =============================================================================
# Deployment configuration
# =============================================================================

VOLUME_NAME = "simple-translate"
VOL_MOUNT_PATH = Path("/data")
FLAVOR = "small"
TOKENIZER_EN = Path("tokenizers/en-vocab_1000")
TOKENIZER_FR = Path("tokenizers/fr-vocab_1000")
WEIGHTS_EN = Path("weights/small-en2fr-20260501T1834.pt")
WEIGHTS_FR = Path("weights/small-fr2en-20260501T1838.pt")

app = modal.App("simple-translate")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0",
        "transformers==5.5.4",
        "pydantic==2.13.2",
        "fastapi==0.136.0",
    )
    .add_local_python_source("schemas")
    .add_local_python_source("flavors")
    .add_local_python_source("architecture")
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

        self.tok_en = PreTrainedTokenizerFast.from_pretrained(
            VOL_MOUNT_PATH / TOKENIZER_EN
        )
        self.tok_fr = PreTrainedTokenizerFast.from_pretrained(
            VOL_MOUNT_PATH / TOKENIZER_FR
        )
        flavor = FLAVORS[FLAVOR]
        self.models = {
            "en2fr": flavor.load(
                tokenizer_source=self.tok_en,
                tokenizer_destination=self.tok_fr,
                checkpoint=VOL_MOUNT_PATH / WEIGHTS_EN,
            ),
            "fr2en": flavor.load(
                tokenizer_source=self.tok_fr,
                tokenizer_destination=self.tok_en,
                checkpoint=VOL_MOUNT_PATH / WEIGHTS_FR,
            ),
        }
        for m in self.models.values():
            m.eval()

    def translate(
        self,
        text: str,
        direction: TranslationDirection,
        temperature: float | None = None,
    ) -> str:
        """
        Generate a translation for one input string.

        Args:
            text: Text in the source language for the chosen direction
            direction: ``en2fr`` (English → French) or ``fr2en`` (French → English)
            temperature: Sampling temperature. ``None`` uses 0.1; values are
                floored at 1e-3.

        Returns:
            Translated text in the target language
        """
        model = self.models[direction]
        tok_source, tok_destination = (
            (self.tok_en, self.tok_fr)
            if direction == "en2fr"
            else (self.tok_fr, self.tok_en)
        )

        tokens_source = tok_source(
            text,
            truncation=True,
            max_length=model.max_sequence_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )["input_ids"]

        tokens_destination = model.generate_with_temp(
            tokens_source,
            temperature=(
                0.1
                if temperature is None
                else max(1e-3, temperature)  # Ensure temp > 0
            ),
        )

        translation = tok_destination.decode(
            tokens_destination[0], skip_special_tokens=True
        )

        return translation

    @modal.asgi_app()
    def fastapi_server(self):
        """Create and configure the FastAPI application."""
        from fastapi import FastAPI

        from schemas import TranslateRequest, TranslateResponse

        server = FastAPI(title="SimpleTranslate API")

        @server.post("/translate", response_model=TranslateResponse)
        def translate_endpoint(request: TranslateRequest) -> TranslateResponse:
            translation = self.translate(
                text=request.text,
                direction=request.direction,
                temperature=request.temperature,
            )
            return TranslateResponse(translation=translation)

        @server.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy"}

        return server
