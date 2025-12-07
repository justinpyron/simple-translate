import httpx
import streamlit as st

from interfaces import TranslateRequest, TranslateResponse

# TODO: Update with proper URL
SERVER_URL = "https://your-modal-app-url.modal.run"
SERVER_ENDPOINT_PATH = "translate"


def translate(
    text_source: str,
    temperature: float,
    beams: int,
) -> str:
    """
    Generate translation by calling the Modal inference server.

    Args:
        text_source: The English text to translate
        temperature: Temperature for sampling (if using temperature-based generation)
        beams: Number of beams for beam search (if using beam search)

    Returns:
        The translated French text
    """
    request = TranslateRequest(
        text_source=text_source,
        temperature=temperature,
        beams=beams,
    )

    try:
        response = httpx.post(
            f"{SERVER_URL}/{SERVER_ENDPOINT_PATH}",
            json=request.model_dump(),
            timeout=30.0,
        )
        response.raise_for_status()
        translate_response = TranslateResponse.model_validate(response.json())
        return translate_response.translation
    except httpx.HTTPError as e:
        st.error(f"Error calling translation server: {e}")
        return ""


what_is_this_app = """
This app demos a neural machine translation model that was trained from scratch.

It uses an [encoder-decoder transformer architecture](https://github.com/justinpyron/simple-translate/blob/main/simple_translate.py) inspired by _[Attention Is All You Need](https://arxiv.org/abs/1706.03762)_.

It was trained on 10 million English/French sentence pairs from the [main dataset](https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset) of the 2015 Workshop on Statistical Machine Translation.
It was trained for roughly 20 hours on an Nvidia L4 GPU on a Google Compute Engine VM.

Source code 👉 [GitHub](https://github.com/justinpyron/simple-translate)
"""
st.set_page_config(page_title="Simple Translate", layout="centered", page_icon="🌎")
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
with col2:
    st.header("French 🇫🇷")
col1, col2 = st.columns(2)
with col1:
    text_input = st.text_area(
        "english_input",
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
    translation = translate(st.session_state["text_input"], temperature, beams)
    st.session_state["text_output"] = translation
text_output.markdown("\n\n" + st.session_state["text_output"])
