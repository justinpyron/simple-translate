"""Modal-based inference server for SimpleTranslate model."""

import modal

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
    .add_local_file("interfaces.py", remote_path="/root/interfaces.py")
    .add_local_file("model_configs.py", remote_path="/root/model_configs.py")
    .add_local_file("simple_translate.py", remote_path="/root/simple_translate.py")
)
volume = modal.Volume.from_name("simple-translate")


@app.cls(
    image=image,
    volumes={"/data": volume},
    cpu=1,
)
class SimpleTranslateServer:
    """Modal class for serving SimpleTranslate model inference."""

    @modal.enter()
    def load_model(self):
        """Load model and tokenizer on container startup."""
        import torch
        from transformers import PreTrainedTokenizerFast

        from model_configs import model_configs
        from simple_translate import SimpleTranslate

        # Load tokenizer from volume
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            "/data/tokenizer_1000/"
        )
        # Load model from volume
        self.model = SimpleTranslate(**model_configs)
        self.model.load_state_dict(
            torch.load(
                "/data/model_for_app_cpu.pt", weights_only=True, map_location="cpu"
            )
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

    @modal.asgi_app()
    def fastapi_server(self):
        """Create and configure the FastAPI application."""
        from fastapi import FastAPI

        from interfaces import TranslateRequest, TranslateResponse

        server = FastAPI(title="SimpleTranslate API")

        @server.post("/translate", response_model=TranslateResponse)
        def translate_endpoint(request: TranslateRequest) -> TranslateResponse:
            """
            Translate English text to French.

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
