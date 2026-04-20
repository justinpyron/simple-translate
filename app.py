import os

import dash_bootstrap_components as dbc
import httpx
from dash import Dash, Input, Output, State, dcc, html

from interfaces import TranslateRequest, TranslateResponse, TranslationDirection

SERVER_URL = os.environ.get("SIMPLE_TRANSLATE_SERVER_URL")
SERVER_ENDPOINT_PATH = "translate"

WHAT_IS_THIS_APP = """
This app demos a neural machine translation model that was built and trained from scratch.

It uses an [encoder-decoder transformer architecture](https://github.com/justinpyron/simple-translate/blob/main/simple_translate.py) inspired by _[Attention Is All You Need](https://arxiv.org/abs/1706.03762)_.

It was trained on 10 million English/French sentence pairs from the [main dataset](https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset) of the 2015 Workshop on Statistical Machine Translation.
It was trained for roughly 20 hours on an Nvidia L4 GPU on a Google Compute Engine VM.

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
        beams=None,  # Beam search is removed per requirements
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
    external_stylesheets=[dbc.themes.SLATE],
    title="Simple Translate 🌎",
)

app.layout = dbc.Container(
    [
        dcc.Store(id="direction-store", data="en2fr"),
        dbc.Row(
            dbc.Col(
                html.H1("Simple Translate 🌎", className="text-center my-4"),
                width=12,
            )
        ),
        dbc.Row(
            dbc.Col(
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            dcc.Markdown(WHAT_IS_THIS_APP),
                            title="What is this app?",
                        )
                    ],
                    start_collapsed=True,
                    className="mb-4",
                ),
                width={"size": 8, "offset": 2},
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.H3(
                        "English 🏴󠁧󠁢󠁥󠁮󠁧󠁿", id="source-label", className="text-center"
                    ),
                    width=5,
                ),
                dbc.Col(
                    dbc.Button(
                        "⇄",
                        id="swap-btn",
                        color="secondary",
                        className="w-100",
                        size="lg",
                    ),
                    width=2,
                    className="d-flex align-items-end",
                ),
                dbc.Col(
                    html.H3("French 🇫🇷", id="target-label", className="text-center"),
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
                        style={"height": "200px"},
                        className="mb-3",
                    ),
                    width=6,
                ),
                dbc.Col(
                    dbc.Textarea(
                        id="target-output",
                        placeholder="Translation will appear here",
                        style={"height": "200px"},
                        readOnly=True,
                        className="mb-3",
                    ),
                    width=6,
                ),
            ]
        ),
        dbc.Row(
            dbc.Col(
                dbc.Button(
                    "Translate",
                    id="translate-btn",
                    color="primary",
                    size="lg",
                    className="w-100 mb-3",
                ),
                width={"size": 6, "offset": 3},
            )
        ),
        dbc.Row(
            dbc.Col(
                [
                    dbc.Button(
                        "Settings",
                        id="settings-toggle",
                        color="link",
                        size="sm",
                        className="p-0 text-muted",
                    ),
                    dbc.Collapse(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Label("Temperature", className="small mb-1"),
                                    dcc.Slider(
                                        id="temp-slider",
                                        min=0.0,
                                        max=1.0,
                                        step=0.05,
                                        value=0.1,
                                        marks={i / 10: str(i / 10) for i in range(11)},
                                    ),
                                ]
                            ),
                            className="mt-2 border-secondary",
                        ),
                        id="settings-collapse",
                        is_open=False,
                    ),
                ],
                width={"size": 4, "offset": 4},
                className="text-center",
            )
        ),
    ],
    fluid=True,
    className="px-5 py-3",
)


@app.callback(
    Output("settings-collapse", "is_open"),
    Input("settings-toggle", "n_clicks"),
    State("settings-collapse", "is_open"),
)
def toggle_settings(n_clicks, is_open):
    if n_clicks:
        return not is_open
    return is_open


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
    # Get port from environment or default to 8050
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=True, host="0.0.0.0", port=port)
