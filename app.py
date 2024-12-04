import streamlit as st
import torch

from model_configs import model_configs, tokenizer
from simple_translate import SimpleTranslate

FILENAME_MODEL_WEIGHTS = "model_2024-12-04T04_00.pt"


@st.cache_resource
def load_model() -> SimpleTranslate:
    model = SimpleTranslate(**model_configs)
    model.load_state_dict(
        torch.load(FILENAME_MODEL_WEIGHTS, map_location=torch.device("cpu"))
    )
    return model


st.set_page_config(page_title="Simple Translate", layout="centered", page_icon="🌎")
rmodel = load_model()

st.title("Simple Translate 🌎")
with st.expander("How it works"):
    st.markdown("This app demos a simple neural machine translation model.")

temperature = st.slider(
    "Temperature",
    min_value=0.05,
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
    text_output = st.empty()  # Empty in order to define it before the button
    text_output.text_area(
        "french_output",
        placeholder="French translation goes here",
        label_visibility="hidden",
        disabled=True,
    )

if st.button("Translate", type="primary", use_container_width=True):
    text_output.text_area(
        "french_output",
        value=text_input,
        label_visibility="hidden",
        disabled=True,
    )
