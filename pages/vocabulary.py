"""Words used more or less than their pre-ChatGPT trend predicts, as a word cloud."""

from pathlib import Path

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, ctx, dash_table, dcc, html
from PIL import ImageDraw, ImageFont, Image
from wordcloud import WordCloud

from charts import trend_figure
from theme import CHROME, LESS, MORE, base_layout
from ui import control
from vocab import BASES, LANGUAGES, MIN_DOCS, frequencies, predict, top_words, unexpected_words

dash.register_page(__name__, path="/vocabulary", name="AI-era vocabulary", order=1,
                   title="CUNI theses: unexpected vocabulary")

FONT_PATH = str(Path(__file__).parent.parent / "assets" / "DroidSansMono.ttf")
CLOUD_FONT = '"Droid Sans Mono", monospace'  # same face the layout is measured with, see style.css
CLOUD_W, CLOUD_H = 1030, 560
CHATGPT = 2022.87  # release, 30 November 2022
MIN_SHARES = [0, 0.1, 0.5, 1, 2, 5, 10]  # minimum-use cutoffs offered, % of theses

_, TOTALS = frequencies("cs")
LAST_YEAR = int(TOTALS.index.max())
FIRST_YEAR = 2006

layout = html.Div([
    html.H1("Which words do theses use more since ChatGPT?"),
    html.P(
        "For every word, a straight line is fitted to how often it was used in the training years "
        "and extended to the compared year. Words whose actual use departs most from that trend are shown: "
        "red ones are used more than expected, blue ones less. Word size shows how big the departure is.",
        className="lede",
    ),
    html.Div([
        control("Language of the thesis", dcc.RadioItems(
            id="vocab-lang", options=[{"label": v, "value": k} for k, v in LANGUAGES.items()], value="cs",
        )),
        control("Compared year", dcc.Dropdown(
            id="vocab-year", clearable=False, value=min(2025, LAST_YEAR),
            options=[{"label": f"{y} (partial)" if y == LAST_YEAR else str(y), "value": y}
                     for y in range(2023, LAST_YEAR + 1)],
        )),
        control("Word frequency", dcc.RadioItems(
            id="vocab-basis", value="n_docs",
            options=[{"label": label, "value": basis} for basis, (label, _) in BASES.items()],
        )),
        control("Change measured as", dcc.RadioItems(
            id="vocab-measure", value="absolute", options=[
                {"label": "Difference from expected", "value": "absolute"},
                {"label": "Ratio to expected", "value": "relative"},
            ],
        )),
        control("Show", dcc.RadioItems(
            id="vocab-direction", value="both", options=[
                {"label": "Both", "value": "both"},
                {"label": "More used than expected", "value": "more"},
                {"label": "Less used than expected", "value": "less"},
            ],
        )),
        control("Training years (trend fitted on)", dcc.RangeSlider(
            id="vocab-train", min=FIRST_YEAR, max=2022, step=1, value=[2013, 2022], pushable=2,
            marks={y: str(y) for y in range(FIRST_YEAR, 2023, 4)},
        )),
        control("Words per direction", dcc.Slider(
            id="vocab-n", min=10, max=100, step=10, value=50, marks={n: str(n) for n in (10, 50, 100)},
        )),
        control("Minimum use (% of theses in the compared year)", dcc.Slider(
            id="vocab-min-share", min=0, max=len(MIN_SHARES) - 1, step=None, value=MIN_SHARES.index(1),
            marks={i: f"{v:g} %" if v else "any" for i, v in enumerate(MIN_SHARES)},
        )),
    ], className="filters"),
    html.P(id="vocab-note", className="note"),
    html.Div(dcc.Graph(id="vocab-cloud", config={"displayModeBar": False, "responsive": False},
                       style={"width": CLOUD_W, "height": CLOUD_H}),
             className="cloud-scroll"),
    html.Div([
        html.H2(id="vocab-detail-title"),
        dcc.Input(id="vocab-search", type="search", debounce=True, placeholder="Look up any word, e.g. klíčový",
                  className="word-search", spellCheck=False, autoComplete="off"),
    ], className="detail-head"),
    html.P(id="vocab-summary", className="note"),
    dcc.Graph(id="vocab-detail", config={"displaylogo": False}),
    html.Details([
        html.Summary("Show data table"),
        dash_table.DataTable(id="vocab-table", page_size=25, sort_action="native", style_as_list_view=True,
                             css=[{"selector": ".dash-spreadsheet", "rule": "font-family: inherit"}]),
    ]),
    html.P(id="vocab-method", className="method"),
    dcc.Store(id="vocab-words"),
    dcc.Store(id="vocab-selected"),
])


def fmt(value, basis):
    """A frequency in its basis' unit, for hover text."""
    return f"{value:.1f} %" if basis == "n_docs" else f"{value:,.0f} per million words"


def comparison(row, measure, basis):
    """Actual use, and expected use with the change, as two phrases."""
    if measure == "relative":
        change = f"{2 ** row.score:.2f}× expected"
    else:
        change = f"{row.score:+.1f} pp" if basis == "n_docs" else f"{row.score:+,.0f} per million"
    used = f"used in {fmt(row.actual, basis)} of theses" if basis == "n_docs" else fmt(row.actual, basis)
    return used, f"expected {fmt(row.expected, basis)} · {change}"


def describe(row, measure, basis):
    return "<br>".join([f"<b>{row.Index}</b>", *comparison(row, measure, basis)])


def cloud_figure(words, measure, basis, theme):
    c = CHROME[theme]
    fig = go.Figure(layout=base_layout(
        theme, width=CLOUD_W, height=CLOUD_H, margin=dict(l=0, r=0, t=0, b=0), hovermode="closest",
        hoverdistance=-1, showlegend=False,
        xaxis=dict(range=[0, CLOUD_W], visible=False, fixedrange=True),
        yaxis=dict(range=[CLOUD_H, 0], visible=False, fixedrange=True),
    ))
    if words.empty:
        return fig

    cloud = WordCloud(width=CLOUD_W, height=CLOUD_H, font_path=FONT_PATH, prefer_horizontal=1.0,
                      max_font_size=64, min_font_size=11, relative_scaling=0.7, margin=6,
                      random_state=7, mode="RGBA", background_color=None)
    cloud.generate_from_frequencies(words.score.abs().to_dict())

    draw = ImageDraw.Draw(Image.new("L", (1, 1)))
    xs, ys, sizes, texts = [], [], [], []
    for (word, _), size, (top, left), _, _ in cloud.layout_:
        x0, y0, x1, y1 = draw.textbbox((0, 0), word, font=ImageFont.truetype(FONT_PATH, size), anchor="lt")
        xs.append(left + (x1 - x0) / 2)
        ys.append(top + (y1 - y0) / 2)
        sizes.append(size)
        texts.append(word)

    placed = words.loc[texts]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, text=texts, mode="text", textposition="middle center",
        textfont=dict(family=CLOUD_FONT, size=sizes,
                      color=[MORE[theme] if s > 0 else LESS[theme] for s in placed.score]),
        hovertext=[describe(r, measure, basis) for r in placed.itertuples()], hoverinfo="text",
        customdata=texts,
    ))
    fig.update_layout(hoverlabel=dict(bgcolor=c["surface"]))
    return fig


@callback(
    Output("vocab-cloud", "figure"), Output("vocab-table", "data"), Output("vocab-table", "columns"),
    Output("vocab-note", "children"), Output("vocab-method", "children"), Output("vocab-words", "data"),
    Input("vocab-lang", "value"), Input("vocab-year", "value"), Input("vocab-basis", "value"),
    Input("vocab-measure", "value"), Input("vocab-direction", "value"), Input("vocab-train", "value"),
    Input("vocab-n", "value"), Input("vocab-min-share", "value"), Input("theme", "data"),
)
def update_cloud(lang, year, basis, measure, direction, train, n, min_share, theme):
    min_share = MIN_SHARES[min_share]
    scored = unexpected_words(lang, tuple(train), year, measure, basis, min_share)
    words = top_words(scored, n, direction)
    fig = cloud_figure(words, measure, basis, theme)

    unit = BASES[basis][1]
    table = words.reset_index()
    table.columns = ["word", f"expected ({unit})", f"actual ({unit})",
                     "log2 ratio" if measure == "relative" else "change (pp)" if basis == "n_docs"
                     else "change (per million)"]
    columns = [{"name": c, "id": c} if c == "word" else
               {"name": c, "id": c, "type": "numeric", "format": {"specifier": ".2f"}} for c in table.columns]

    _, totals = frequencies(lang)
    note = (f"{LANGUAGES[lang]} theses: trend fitted on {train[0]}–{train[1]}, compared with {year}"
            + (" (year still in progress)" if year == LAST_YEAR else "") + "."
            + (" Ratios of occurrences mostly reflect topics: a few theses that repeat a word can dominate it."
               if basis == "n_tokens" and measure == "relative" else "")
            + (" Differences of occurrences favour very frequent words, such as conjunctions and prepositions."
               if basis == "n_tokens" and measure == "absolute" else ""))
    cutoff = f"{min_share:g} % and at least {MIN_DOCS}" if min_share else f"at least {MIN_DOCS}"
    method = (f"{len(scored):,} words considered: used in {cutoff} of the {int(totals.n_docs[year]):,} "
              f"{LANGUAGES[lang]} theses from {year}, or expected to be. Word forms are lowercased and not lemmatised, so "
              "inflected forms count separately. "
              + ("Every occurrence counts, so words a thesis repeats weigh more. " if basis == "n_tokens" else
                 "A thesis counts once however often it uses the word. ")
              + "The data also flags topical shifts (theses about AI, the end of "
              "COVID-19), not only changes in writing style.")
    return fig, table.round(2).to_dict("records"), columns, note, method, list(words.index)


@callback(Output("vocab-selected", "data"), Input("vocab-cloud", "clickData"), Input("vocab-search", "value"))
def select_word(click, search):
    """The word to show in detail: whichever was picked last, a clicked one or a typed one."""
    if ctx.triggered_id == "vocab-cloud" and click:
        return click["points"][0]["customdata"]
    return (search or "").strip().lower() or None


def summarize(word, scored, year, measure, basis, min_share):
    """One line on how the word departs from its trend and where it ranks."""
    if word not in scored.index:
        cutoff = f"{min_share:g} % of theses and " if min_share else ""
        return f"Below the minimum use in {year} (at least {cutoff}{MIN_DOCS} theses), so not ranked."
    row = scored.loc[word]
    used, expected = comparison(row, measure, basis)
    side = scored[np.sign(scored.score) == np.sign(row.score)]
    rank = int((side.score.abs() > abs(row.score)).sum()) + 1
    direction = "more" if row.score > 0 else "less"
    return (f"In {year}: {used}, {expected}. Rank {rank:,} of {len(side):,} words used {direction} "
            "than expected.")


@callback(
    Output("vocab-detail", "figure"), Output("vocab-detail-title", "children"), Output("vocab-summary", "children"),
    Input("vocab-selected", "data"), Input("vocab-words", "data"),
    Input("vocab-lang", "value"), Input("vocab-year", "value"), Input("vocab-basis", "value"),
    Input("vocab-measure", "value"), Input("vocab-train", "value"), Input("vocab-min-share", "value"),
    Input("theme", "data"),
)
def update_detail(selected, words, lang, year, basis, measure, train, min_share, theme):
    min_share = MIN_SHARES[min_share]
    tables, _ = frequencies(lang)
    word = selected or (words[0] if words else None)
    fig = go.Figure(layout=base_layout(theme, height=380))
    if word is None:
        return fig, "", "Click a word in the cloud, or look one up."
    if word not in tables[basis].index:
        return fig, f"“{word}” not found", (
            f"Not in the data for {LANGUAGES[lang]} theses. Only single lowercase word forms used in at least "
            "20 theses across all years are included, and inflected forms are separate words.")

    freq = tables[basis].loc[[word]]
    fit_years = np.arange(train[0], year + 1)
    fitted = np.array([predict(freq, tuple(train), y)[0] for y in fit_years])
    docs = basis == "n_docs"
    fig = trend_figure(
        theme, freq.iloc[0].astype("float64"), fit_years, fitted, train,
        event=CHATGPT, event_label="ChatGPT released", expected_label="Expected (linear trend)",
        y_title=f"% of {LANGUAGES[lang]} theses using the word" if docs else "Occurrences per million words",
        y_format=".1f" if docs else ",.0f", y_suffix=" % of theses" if docs else " per million words",
        ticksuffix=" %" if docs else "", train_label="training years",
    )
    scored = unexpected_words(lang, tuple(train), year, measure, basis, min_share)
    summary = summarize(word, scored, year, measure, basis, min_share)
    return fig, f"“{word}” over time", summary
