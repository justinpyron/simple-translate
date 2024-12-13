import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from model_configs import model_configs, tokenizer
from simple_translate import SimpleTranslate

FILENAME_MODEL_WEIGHTS = "model_for_app.pt"
SEED_OPTIONS = [
    "text_seeds/george_washington.csv",
    "text_seeds/thomas_jefferson.csv",
    "text_seeds/abraham_lincoln.csv",
]


@st.cache_resource
def load_model() -> SimpleTranslate:
    model = SimpleTranslate(**model_configs)
    model.load_state_dict(
        torch.load(FILENAME_MODEL_WEIGHTS, map_location=torch.device("cpu"))
    )
    return model


@st.cache_resource
def fetch_seed_text(filename: str) -> pd.Series:
    return pd.read_csv(filename)["sentences"]


def translate(
    text_source,
    model,
    tokenizer,
    temperature: float,
    beams: int,
) -> torch.tensor:
    """Generate translation for a single input example. Batches not handled."""
    tokens_source = tokenizer(
        text_source,
        truncation=True,
        max_length=model.max_sequence_length,
        return_attention_mask=False,
        return_token_type_ids=False,
        return_tensors="pt",
    )["input_ids"]
    if temperature is not None:
        temperature = max(1e-3, temperature)  # temperature must be positive
        tokens_destination = model.generate_with_temp(
            tokens_source, temperature=temperature
        )
    if beams is not None:
        tokens_destination = model.generate_with_beams(tokens_source, beam_width=beams)
    translation = tokenizer.decode(tokens_destination[0], skip_special_tokens=True)
    return translation


what_is_this_app = """
This app demos a neural machine translation model that was trained from scratch.

It uses an [encoder-decoder transformer architecture](https://github.com/justinpyron/simple-translate/blob/main/simple_translate.py) inspired by _[Attention Is All You Need](https://arxiv.org/abs/1706.03762)_.

It was trained on 10 million English/French sentence pairs from the [main dataset](https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset) of the 2015 Workshop on Statistical Machine Translation.
It was trained for roughly 20 hours on an Nvidia L4 GPU on a Google Compute Engine VM.

Source code 👉 [GitHub](https://github.com/justinpyron/simple-translate)
"""
st.set_page_config(page_title="Simple Translate", layout="centered", page_icon="🌎")
model = load_model()
if "text_input" not in st.session_state:
    st.session_state["text_input"] = None
if "text_output" not in st.session_state:
    st.session_state["text_output"] = ""
st.title("Simple Translate 🌎")
with st.expander("What is this app?"):
    st.markdown(what_is_this_app)
col1, col2 = st.columns([1, 2])
with col1:
    gen_options = ["Sample with temperature", "Beam search"]
    gen_strategy = st.radio(
        label="Generation Strategy",
        options=gen_options,
        help="How to generate the translation.\n\nBeam search is deterministic. Sampling with temperature is not.",
    )
with col2:
    temperature = None
    beams = None
    if gen_strategy == gen_options[0]:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            value=0.1,
            help="Controls randomness of generated translation. Lower values are less random.",
        )
    else:
        beams = st.slider(
            "Number of beams",
            min_value=1,
            max_value=20,
            step=1,
            value=5,
            help="Controls how wide of a search to conduct when (greedily) computing most likely translation.",
        )
col1, col2 = st.columns(2)
with col1:
    st.header("English 🇬🇧")
    seed_toggle = st.toggle("Seed with examples")
    if seed_toggle:
        source = st.radio(
            label="Source",
            options=SEED_OPTIONS,
            index=0,
            format_func=lambda x: x.split("/")[-1]
            .replace(".csv", "")
            .split("_")[-1]
            .capitalize(),
            horizontal=True,
            help="Text taken from Wikipedia articles",
        )
        seed_button = st.button("Seed", use_container_width=True)
        if seed_button:
            seeds = fetch_seed_text(source)
            seed = seeds.sample().values[0]
            st.session_state["text_input"] = seed
with col2:
    st.header("French 🇫🇷")
col1, col2 = st.columns(2)
with col1:
    text_input = st.text_area(
        "english_input",
        # value=seed if seed_toggle and seed_button else None,
        value=st.session_state["text_input"],
        height=200,
        placeholder="Enter English text here",
        label_visibility="hidden",
    )
    if text_input is not None:
        st.session_state["text_input"] = text_input
with col2:
    st.write("#####")
    text_output = st.empty()  # Empty in order to define it before the button
if st.button("Translate", type="primary", use_container_width=True):
    translation = translate(
        st.session_state["text_input"], model, tokenizer, temperature, beams
    )
    st.session_state["text_output"] = translation
text_output.markdown("\n\n" + st.session_state["text_output"])
