import streamlit as st
import torch
import torch.nn.functional as F

from model_configs import model_configs, tokenizer
from simple_translate import SimpleTranslate

# FILENAME_MODEL_WEIGHTS = "model_for_app.pt"
FILENAME_MODEL_WEIGHTS = "model_2024-12-05T04_17.pt"


def translate(
    text_source,
    model,
    tokenizer,
    temperature: float,
) -> torch.tensor:
    """Generate translation for a single input example. Batches not handled."""
    temperature = max(1e-3, temperature)  # temperature must be positive
    tokens_source = tokenizer(
        text_source,
        truncation=True,
        max_length=model.max_sequence_length,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_tensors="pt",
    )["input_ids"]
    tokens_destination = model.generate(tokens_source, temperature=temperature)
    translation = tokenizer.decode(tokens_destination[0], skip_special_tokens=True)
    return translation


@st.cache_resource
def load_model() -> SimpleTranslate:
    model = SimpleTranslate(**model_configs)
    model.load_state_dict(
        torch.load(FILENAME_MODEL_WEIGHTS, map_location=torch.device("cpu"))
    )
    return model


st.set_page_config(page_title="Simple Translate", layout="centered", page_icon="🌎")
model = load_model()

st.title("Simple Translate 🌎")
with st.expander("How it works"):
    st.markdown("This app demos a simple neural machine translation model.")

temperature = st.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    value=0.7,
)

col1, col2 = st.columns(2)
with col1:
    st.header("English 🇬🇧")
    text_input = st.text_area(
        "english_input",
        placeholder="Enter English text here",
        label_visibility="hidden",
    )
with col2:
    st.header("French 🇫🇷")
    st.write("#####")
    text_output = st.empty()  # Empty in order to define it before the button

if st.button("Translate", type="primary", use_container_width=True):
    translation = translate(text_input, model, tokenizer, temperature)
    text_output.markdown("\n\n" + translation)
