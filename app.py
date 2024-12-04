import streamlit as st
import torch
import torch.nn.functional as F

from model_configs import model_configs, tokenizer
from simple_translate import SimpleTranslate

FILENAME_MODEL_WEIGHTS = "model_2024-12-04T04_00.pt"


def translate(
    text_source,
    model,
    tokenizer,
    temperature: float = 0.01,  # Default is most likely token
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
    model.eval()
    tokens_destination = torch.tensor([[model.token_id_bos]])
    with torch.no_grad():
        for i in range(model.max_sequence_length - 1):
            logits = model.forward(tokens_source, tokens_destination)
            logits_final_token = logits[:, -1, :]
            probability = F.softmax(logits_final_token / temperature, dim=-1)
            next_token = torch.multinomial(probability, num_samples=1)
            tokens_destination = torch.cat((tokens_destination, next_token), dim=-1)
            if next_token[0, 0] == model.token_id_eos:
                break
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
    st.write("#####")
    text_output = st.empty()  # Empty in order to define it before the button

if st.button("Translate", type="primary", use_container_width=True):
    translation = translate(text_input, model, tokenizer, temperature)
    text_output.markdown("\n\n" + translation)
