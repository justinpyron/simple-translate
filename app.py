import os

import dash_bootstrap_components as dbc
import httpx
from dash import Dash, Input, Output, State, dcc, html

from schemas import TranslateRequest, TranslateResponse, TranslationDirection

SERVER_URL = os.environ.get("SIMPLE_TRANSLATE_SERVER_URL")
SERVER_ENDPOINT_PATH = "translate"
PORT = 8080

WHAT_IS_THIS_APP = """
This app demos a Neural Machine Translation (NMT) model built from scratch.

It uses an [encoder-decoder transformer architecture](https://github.com/justinpyron/simple-translate/blob/main/architecture.py) inspired by _[Attention Is All You Need](https://arxiv.org/abs/1706.03762)_. It was trained on millions of English/French sentence pairs from the [main dataset](https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset) of the 2015 Workshop on Statistical Machine Translation.

Source code 👉 [GitHub](https://github.com/justinpyron/simple-translate)
"""


def translate(
    text_source: str,
    direction: TranslationDirection,
    temperature: float | None = None,
) -> str:
    """
    Generate translation by calling the Modal inference server.

    Args:
        text_source: The source text to translate
        direction: Translation direction (en2fr or fr2en)
        temperature: Temperature for sampling

    Returns:
        The translated target text
    """
    request = TranslateRequest(
        text_source=text_source,
        direction=direction,
        temperature=temperature,
    )

    try:
        response = httpx.post(
            f"{SERVER_URL}/{SERVER_ENDPOINT_PATH}",
            json=request.model_dump(),
            timeout=120.0,
        )
        response.raise_for_status()
        translate_response = TranslateResponse.model_validate(response.json())
        return translate_response.translation
    except httpx.HTTPError as e:
        return f"Error: {e}"


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Simple Translate 🌎",
)

app.layout = dbc.Container(
    [
        dcc.Store(id="direction-store", data="en2fr"),
        html.Div("🌎 Simple Translate", className="app-title"),
        html.Div(
            [
                html.Button(
                    [
                        html.Span("ℹ", className="info-icon"),
                        html.Span("About the Project"),
                    ],
                    id="about-toggle",
                    n_clicks=0,
                    className="secondary-pill",
                ),
                dbc.Collapse(
                    html.Div(
                        dcc.Markdown(WHAT_IS_THIS_APP, link_target="_blank"),
                        className="about-content",
                    ),
                    id="about-collapse",
                    is_open=False,
                ),
            ],
            className="about-wrapper",
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div(
                        "English 🏴󠁧󠁢󠁥󠁮󠁧󠁿",
                        id="source-label",
                        className="lang-label",
                    ),
                    width=5,
                ),
                dbc.Col(
                    html.Button(
                        "⇄",
                        id="swap-btn",
                        className="swap-button",
                        n_clicks=0,
                        title="Swap languages",
                    ),
                    width=2,
                    className="d-flex justify-content-center align-items-center",
                ),
                dbc.Col(
                    html.Div(
                        "French 🇫🇷",
                        id="target-label",
                        className="lang-label",
                    ),
                    width=5,
                ),
            ],
            className="mb-2 align-items-center",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Textarea(
                        id="source-input",
                        placeholder="Enter English text here",
                        style={"height": "220px"},
                        className="mb-3",
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Textarea(
                        id="target-output",
                        placeholder="Translation will appear here",
                        style={"height": "220px"},
                        readOnly=True,
                        className="mb-3",
                    ),
                    width=6,
                ),
            ]
        ),
        html.Div(
            dbc.Button(
                dcc.Loading(
                    html.Span("Translate", id="translate-btn-text"),
                    id="translate-btn-loading",
                    type="dot",
                    color="#ffffff",
                    overlay_style={"visibility": "visible", "filter": "blur(0)"},
                ),
                id="translate-btn",
                color="primary",
                size="lg",
                className="translate-button",
            ),
            className="translate-button-row",
        ),
        html.Div(
            [
                html.Button(
                    [
                        html.Span("⚙", className="gear"),
                        html.Span("Settings"),
                    ],
                    id="settings-toggle",
                    n_clicks=0,
                    className="secondary-pill",
                ),
                dbc.Collapse(
                    html.Div(
                        [
                            html.Label(
                                "Temperature",
                                className="lang-label d-block mb-2 text-center",
                            ),
                            dcc.Slider(
                                id="temp-slider",
                                min=0.0,
                                max=1.0,
                                step=0.1,
                                value=0.1,
                                marks={i / 10: str(i / 10) for i in range(11)},
                                allow_direct_input=False,
                            ),
                        ],
                        className="settings-inline",
                    ),
                    id="settings-collapse",
                    is_open=False,
                ),
            ],
            className="settings-wrapper",
        ),
    ],
    fluid=True,
    className="app-container",
)


@app.callback(
    Output("about-collapse", "is_open"),
    Output("about-toggle", "className"),
    Input("about-toggle", "n_clicks"),
    State("about-collapse", "is_open"),
)
def toggle_about(n_clicks, is_open):
    new_is_open = (not is_open) if n_clicks else is_open
    class_name = "secondary-pill active" if new_is_open else "secondary-pill"
    return new_is_open, class_name


@app.callback(
    Output("settings-collapse", "is_open"),
    Output("settings-toggle", "className"),
    Input("settings-toggle", "n_clicks"),
    State("settings-collapse", "is_open"),
)
def toggle_settings(n_clicks, is_open):
    new_is_open = (not is_open) if n_clicks else is_open
    class_name = "secondary-pill active" if new_is_open else "secondary-pill"
    return new_is_open, class_name


@app.callback(
    Output("direction-store", "data"),
    Output("source-label", "children"),
    Output("target-label", "children"),
    Output("source-input", "placeholder"),
    Output("source-input", "value"),
    Output("target-output", "value"),
    Input("swap-btn", "n_clicks"),
    State("direction-store", "data"),
    State("source-input", "value"),
    State("target-output", "value"),
    prevent_initial_call=True,
)
def swap_direction(n_clicks, current_direction, source_text, target_text):
    if current_direction == "en2fr":
        new_direction = "fr2en"
        source_label = "French 🇫🇷"
        target_label = "English 🏴󠁧󠁢󠁥󠁮󠁧󠁿"
        placeholder = "Enter French text here"
    else:
        new_direction = "en2fr"
        source_label = "English 🏴󠁧󠁢󠁥󠁮󠁧󠁿"
        target_label = "French 🇫🇷"
        placeholder = "Enter English text here"

    # Swap the content of the boxes
    return (
        new_direction,
        source_label,
        target_label,
        placeholder,
        target_text,
        source_text,
    )


@app.callback(
    Output("target-output", "value", allow_duplicate=True),
    Input("translate-btn", "n_clicks"),
    State("source-input", "value"),
    State("direction-store", "data"),
    State("temp-slider", "value"),
    prevent_initial_call=True,
)
def handle_translate(n_clicks, source_text, direction, temperature):
    if not source_text:
        return ""

    translation = translate(source_text, direction, temperature)
    return translation


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=PORT)
