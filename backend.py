"""Modal-based inference server for SimpleTranslate model."""

from pathlib import Path

import modal

# =============================================================================
# Deployment configuration (paths below are relative to VOL_MOUNT_PATH)
# =============================================================================

VOLUME_NAME = "simple-translate"
VOL_MOUNT_PATH = Path("/data")
FLAVOR = "small"
TOKENIZER_SOURCE_PATH = Path("tokenizers/en-vocab_1000")
TOKENIZER_DESTINATION_PATH = Path("tokenizers/fr-vocab_1000")
CHECKPOINT_PATH = Path("model_for_app_cpu.pt")

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
        """Load model and tokenizers on container startup."""
        from transformers import PreTrainedTokenizerFast

        from flavors import FLAVORS

        self.tokenizer_source = PreTrainedTokenizerFast.from_pretrained(
            VOL_MOUNT_PATH / TOKENIZER_SOURCE_PATH
        )
        self.tokenizer_destination = PreTrainedTokenizerFast.from_pretrained(
            VOL_MOUNT_PATH / TOKENIZER_DESTINATION_PATH
        )

        self.model = FLAVORS[FLAVOR].load(
            self.tokenizer_source,
            self.tokenizer_destination,
            checkpoint=VOL_MOUNT_PATH / CHECKPOINT_PATH,
        )
        self.model.eval()

    def translate(
        self,
        text_source: str,
        temperature: float | None = None,
        beams: int | None = None,
    ) -> str:
        """
        Generate translation for a single input example.

        Args:
            text_source: Source-language text to translate
            temperature: Temperature for sampling (if using temperature-based generation)
            beams: Number of beams for beam search (if using beam search)

        Returns:
            Translated text in the destination language
        """
        tokens_source = self.tokenizer_source(
            text_source,
            truncation=True,
            max_length=self.model.max_sequence_length,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors="pt",
        )["input_ids"]

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
            tokens_destination = self.model.generate_with_temp(
                tokens_source, temperature=0.1
            )

        translation = self.tokenizer_destination.decode(
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
            """
            Translate source text using the loaded model.

            Args:
                request: Translation request with source text and generation parameters

            Returns:
                Translation response with the translated text
            """
            translation = self.translate(
                text_source=request.text_source,
                temperature=request.temperature,
                beams=request.beams,
            )
            return TranslateResponse(translation=translation)

        @server.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy"}

        return server
