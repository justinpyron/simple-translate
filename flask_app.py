import os

import torch
from flask import Flask, jsonify, request

from model_configs import model_configs, tokenizer
from simple_translate import SimpleTranslate

FILENAME_MODEL_WEIGHTS = "model_for_app_cpu.pt"


app = Flask(__name__)
model = SimpleTranslate(**model_configs)
model.load_state_dict(torch.load(FILENAME_MODEL_WEIGHTS, weights_only=True))


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "Server is running."})


@app.route("/translate", methods=["POST"])
def translate():
    """Translate an input text from English to French"""
    data = request.get_json()
    text = data["text"]
    temperature = data.get("temperature")
    beams = data.get("beams")
    tokens_source = tokenizer(
        text,
        truncation=True,
        max_length=model.max_sequence_length,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_tensors="pt",
    )["input_ids"]
    if temperature is not None and beams is not None:
        return (
            jsonify(
                {
                    "error": "Invalid input",
                    "details": "Only temperature or beams can be passed as input, but not both!",
                }
            ),
            400,
        )
    elif temperature is not None:
        temperature = max(1e-3, temperature)  # temperature must be positive
        tokens_destination = model.generate_with_temp(
            tokens_source, temperature=temperature
        )
    elif beams is not None:
        tokens_destination = model.generate_with_beams(tokens_source, beam_width=beams)
    else:
        tokens_destination = model.generate_with_temp(tokens_source, temperature=1e-3)
    translation = tokenizer.decode(tokens_destination[0], skip_special_tokens=True)
    return jsonify({"translation": translation})


if __name__ == "__main__":
    server = os.environ.get("SIMPLE_TRANSLATE_SERVER")
    app.run(debug=server == "local", host="0.0.0.0", port=8080)
