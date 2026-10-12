"""Words used more or less than their pre-ChatGPT trend predicts, as a word cloud."""

from pathlib import Path

import dash
import numpy as np
import plotly.graph_objects as go
import snowballstemmer
from dash import Input, Output, callback, clientside_callback, ctx, dash_table, dcc, html
from PIL import Image, ImageDraw, ImageFont
from wordcloud import WordCloud

from charts import trend_figure
from data import FACULTIES, FIRST_YEAR
from derived import STEMMERS
from theme import CHROME, LESS, MORE, base_layout
from trends import fit, interval
from ui import control, parse_query, url_store, url_sync
from vocab import BASES, KIND, LANGUAGES, MIN_DOCS, SCALE, frequencies, last_year, top_words, unexpected_words

dash.register_page(__name__, path="/vocabulary", name="AI-era vocabulary", order=1,
                   title="CUNI theses: unexpected vocabulary")

FONT_PATH = str(Path(__file__).parent.parent / "assets" / "DroidSansMono.ttf")
CLOUD_FONT = '"Droid Sans Mono", monospace'  # same face the layout is measured with, see style.css
CLOUD_W, CLOUD_H = 1030, 560  # at full width; narrower screens get a smaller cloud with smaller words
CHATGPT = 2022.87  # release, 30 November 2022
MIN_SHARES = [0, 0.1, 0.5, 1, 2, 5, 10]  # minimum-use cutoffs offered, % of theses
LAST_YEAR = last_year()
MEASURES = {
    "absolute": "Difference from expected",
    "relative": "Ratio to expected",
    "z": "Surprise (standard errors from the trend)",
}

SPEC = {
    "lang": ("choice", "cs", LANGUAGES),
    "year": ("int", min(2025, LAST_YEAR), (2023, LAST_YEAR)),
    "basis": ("choice", "n_docs", BASES),
    "measure": ("choice", "absolute", MEASURES),
    "direction": ("choice", "both", ("both", "more", "less")),
    "train": ("range", [2013, 2022], (FIRST_YEAR, 2022)),
    "n": ("int", 50, (10, 100)),
    "min": ("int", MIN_SHARES.index(1), (0, len(MIN_SHARES) - 1)),
    "faculties": ("list", list(FACULTIES), FACULTIES),
    "forms": ("list", [], ("stem",)),
    "word": ("text", None, None),
}


def layout(**query):
    """The page; settings in the URL query (e.g. ?measure=z&forms=stem) preset the controls."""
    q = parse_query(query, SPEC)
    return html.Div([
        html.H1("Which words do theses use more since ChatGPT?"),
        html.P(
            "For every word, a straight line is fitted to how often it was used in the training years "
            "and extended to the compared year. Words whose actual use departs most from that trend are shown: "
            "red ones are used more than expected, blue ones less. Word size shows how big the departure is.",
            className="lede",
        ),
        html.Div([
            control("Faculties", dcc.Dropdown(
                id="vocab-faculties", multi=True, value=q["faculties"], placeholder="Select faculties…",
                options=[{"label": n, "value": f} for f, n in sorted(FACULTIES.items(), key=lambda kv: kv[1])],
            ), wide=True),
            control("Language of the thesis", dcc.RadioItems(
                id="vocab-lang", options=[{"label": v, "value": k} for k, v in LANGUAGES.items()], value=q["lang"],
            )),
            control("Compared year", dcc.Dropdown(
                id="vocab-year", clearable=False, value=q["year"],
                options=[{"label": f"{y} (partial)" if y == LAST_YEAR else str(y), "value": y}
                         for y in range(2023, LAST_YEAR + 1)],
            )),
            control("Word frequency", dcc.RadioItems(
                id="vocab-basis", value=q["basis"],
                options=[{"label": label, "value": basis} for basis, (label, _) in BASES.items()],
            )),
            control("Change measured as", dcc.RadioItems(
                id="vocab-measure", value=q["measure"], options=[{"label": v, "value": k} for k, v in MEASURES.items()],
            )),
            control("Show", dcc.RadioItems(
                id="vocab-direction", value=q["direction"], options=[
                    {"label": "Both", "value": "both"},
                    {"label": "More used than expected", "value": "more"},
                    {"label": "Less used than expected", "value": "less"},
                ],
            )),
            control("Word forms", dcc.Checklist(
                id="vocab-forms", value=q["forms"],
                options=[{"label": "Merge inflected forms (klíčový, klíčové, klíčovou…)", "value": "stem"}],
            )),
            control("Training years (trend fitted on)", dcc.RangeSlider(
                id="vocab-train", min=FIRST_YEAR, max=2022, step=1, value=q["train"], pushable=2,
                marks={y: str(y) for y in range(FIRST_YEAR, 2023, 4)},
            ), slider=True),
            control("Words per direction", dcc.Slider(
                id="vocab-n", min=10, max=100, step=10, value=q["n"], marks={n: str(n) for n in (10, 50, 100)},
            ), slider=True),
            control("Minimum use (% of theses in the compared year)", dcc.Slider(
                id="vocab-min-share", min=0, max=len(MIN_SHARES) - 1, step=None, value=q["min"],
                marks={i: f"{v:g} %" if v else "any" for i, v in enumerate(MIN_SHARES)},
            ), slider=True),
        ], className="filters"),
        html.P(id="vocab-note", className="note"),
        html.Div(dcc.Graph(id="vocab-cloud", config={"displayModeBar": False, "responsive": False}),
                 className="cloud-scroll"),
        html.Div([
            html.H2(id="vocab-detail-title"),
            dcc.Input(id="vocab-search", type="search", debounce=True, value=q["word"],
                      placeholder="Look up any word, e.g. klíčový", className="word-search",
                      spellCheck=False, autoComplete="off"),
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
        dcc.Store(id="vocab-selected", data=q["word"]),
        dcc.Store(id="vocab-width"),
        url_store("vocab-url"),
    ])


# The cloud is laid out on the server at the width the viewer's screen allows.
clientside_callback(
    "function(_) { return Math.min(window.innerWidth - 40, %d); }" % CLOUD_W,
    Output("vocab-width", "data"),
    Input("vocab-width", "id"),
)


def source(faculties, forms):
    """Arguments selecting the data: None and "word" are the exact counts of all faculties."""
    faculties = tuple(sorted(faculties or []))
    all_faculties = set(faculties) == set(FACULTIES)
    unit = "stem" if "stem" in (forms or []) else "word"
    return (None if all_faculties else faculties), unit


def fmt(value, basis):
    """A frequency in its basis' unit, for hover text."""
    return f"{value:.1f} %" if basis == "n_docs" else f"{value:,.0f} per million words"


def comparison(row, measure, basis):
    """Actual use, and expected use with the change, as two phrases."""
    if measure == "relative":
        change = f"{2 ** row.score:.2f}× expected"
    elif measure == "z":
        change = f"{row.score:+.1f} standard errors"
    else:
        change = f"{row.actual - row.expected:+.1f} pp" if basis == "n_docs" else f"{row.actual - row.expected:+,.0f} per million"
    used = f"used in {fmt(row.actual, basis)} of theses" if basis == "n_docs" else fmt(row.actual, basis)
    return used, f"expected {fmt(row.expected, basis)} · {change}"


def describe(row, measure, basis, forms):
    lines = [f"<b>{row.label}</b>", *comparison(row, measure, basis)]
    if forms is not None and len(forms):
        lines.append("forms: " + ", ".join(forms[:6]) + (" …" if len(forms) > 6 else ""))
    return "<br>".join(lines)


def cloud_figure(words, measure, basis, theme, width, forms):
    c = CHROME[theme]
    width = int(max(300, min(width or CLOUD_W, CLOUD_W)))
    height = int(max(360, CLOUD_H * width / CLOUD_W))
    fig = go.Figure(layout=base_layout(
        theme, width=width, height=height, margin=dict(l=0, r=0, t=0, b=0), hovermode="closest",
        hoverdistance=-1, showlegend=False,
        xaxis=dict(range=[0, width], visible=False, fixedrange=True),
        yaxis=dict(range=[height, 0], visible=False, fixedrange=True),
    ))
    if words.empty:
        return fig

    scale = width / CLOUD_W
    cloud = WordCloud(width=width, height=height, font_path=FONT_PATH, prefer_horizontal=1.0,
                      max_font_size=max(24, int(64 * scale)), min_font_size=max(8, int(11 * scale)),
                      relative_scaling=0.7, margin=max(3, int(6 * scale)), random_state=7, mode="RGBA",
                      background_color=None)
    # Lay out labels, keeping keys distinct (two stems can share a most-used form only in theory).
    label_of = dict(zip(words.index, words.label))
    cloud.generate_from_frequencies({label_of[k]: abs(s) for k, s in words.score.items()})
    key_of = {label: key for key, label in label_of.items()}

    draw = ImageDraw.Draw(Image.new("L", (1, 1)))
    xs, ys, sizes, keys = [], [], [], []
    for (label, _), size, (top, left), _, _ in cloud.layout_:
        x0, y0, x1, y1 = draw.textbbox((0, 0), label, font=ImageFont.truetype(FONT_PATH, size), anchor="lt")
        xs.append(left + (x1 - x0) / 2)
        ys.append(top + (y1 - y0) / 2)
        sizes.append(size)
        keys.append(key_of[label])

    placed = words.loc[keys]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, text=list(placed.label), mode="text", textposition="middle center",
        textfont=dict(family=CLOUD_FONT, size=sizes,
                      color=[MORE[theme] if s > 0 else LESS[theme] for s in placed.score]),
        hovertext=[describe(r, measure, basis, forms.get(r.Index) if forms is not None else None)
                   for r in placed.itertuples()],
        hoverinfo="text", customdata=keys,
    ))
    fig.update_layout(hoverlabel=dict(bgcolor=c["surface"]))
    return fig


@callback(
    Output("vocab-cloud", "figure"), Output("vocab-table", "data"), Output("vocab-table", "columns"),
    Output("vocab-note", "children"), Output("vocab-method", "children"), Output("vocab-words", "data"),
    Input("vocab-lang", "value"), Input("vocab-year", "value"), Input("vocab-basis", "value"),
    Input("vocab-measure", "value"), Input("vocab-direction", "value"), Input("vocab-train", "value"),
    Input("vocab-n", "value"), Input("vocab-min-share", "value"), Input("vocab-faculties", "value"),
    Input("vocab-forms", "value"), Input("vocab-width", "data"), Input("theme", "data"),
)
def update_cloud(lang, year, basis, measure, direction, train, n, min_share, faculties, forms, width, theme):
    if not faculties:
        return go.Figure(layout=base_layout(theme, height=200)), [], [], "Select at least one faculty.", "", []
    min_share = MIN_SHARES[min_share]
    facs, unit = source(faculties, forms)
    f = frequencies(lang, facs, unit)
    scored = unexpected_words(lang, tuple(train), year, measure, basis, min_share, facs, unit)
    words = top_words(scored, n, direction)
    fig = cloud_figure(words, measure, basis, theme, width, f.forms if unit == "stem" else None)

    value_unit = BASES[basis][1]
    table = words[["label", "expected", "actual", "lower", "upper", "score"]].reset_index(drop=True)
    table.columns = ["word", f"expected ({value_unit})", f"actual ({value_unit})", "95 % interval from",
                     "95 % interval to", MEASURES[measure].lower()]
    columns = [{"name": c, "id": c} if c == "word" else
               {"name": c, "id": c, "type": "numeric", "format": {"specifier": ".2f"}} for c in table.columns]

    scope = "all faculties" if facs is None else (FACULTIES[facs[0]] if len(facs) == 1 else f"{len(facs)} faculties")
    note = (f"{LANGUAGES[lang]} theses, {scope}: trend fitted on {train[0]}–{train[1]}, compared with {year}"
            + (" (year still in progress)" if year == LAST_YEAR else "") + "."
            + (" Ratios of occurrences mostly reflect topics: a few theses that repeat a word can dominate it."
               if basis == "n_tokens" and measure == "relative" else "")
            + (" Differences of occurrences favour very frequent words, such as conjunctions and prepositions."
               if basis == "n_tokens" and measure == "absolute" else "")
            + (" Surprise divides the difference by the trend's own year-to-year scatter, so words that were "
               "steady before stand out." if measure == "z" else ""))
    cutoff = f"{min_share:g} % and at least {MIN_DOCS}" if min_share else f"at least {MIN_DOCS}"
    per_thesis = ("Because faculties are selected or forms merged, this uses per-thesis counts, which only "
                  "include word forms a thesis uses at least twice; shares are lower than in the default view. "
                  if f.per_thesis else "")
    merged = ("Inflected forms are merged by a light stemmer and shown by their most used form. " if unit == "stem"
              else "Word forms are lowercased and not lemmatised, so inflected forms count separately. ")
    method = (f"{len(scored):,} words considered: used in {cutoff} of the {int(f.totals.n_docs[year]):,} "
              f"{LANGUAGES[lang]} theses from {year}, or expected to be. {merged}{per_thesis}"
              + ("Every occurrence counts, so words a thesis repeats weigh more. " if basis == "n_tokens" else
                 "A thesis counts once however often it uses the word. ")
              + "The data also flags topical shifts (theses about AI, the end of COVID-19), not only changes "
              "in writing style.")
    return fig, table.round(2).to_dict("records"), columns, note, method, list(words.index)


@callback(Output("vocab-selected", "data"), Input("vocab-cloud", "clickData"), Input("vocab-search", "value"),
          prevent_initial_call=True)
def select_word(click, search):
    """The word to show in detail: whichever was picked last, a clicked one or a typed one."""
    if ctx.triggered_id == "vocab-cloud" and click:
        return click["points"][0]["customdata"]
    return (search or "").strip().lower() or None


def summarize(key, label, scored, year, measure, basis, min_share):
    """One line on how the word departs from its trend and where it ranks."""
    if key not in scored.index:
        cutoff = f"{min_share:g} % of theses and " if min_share else ""
        return f"Below the minimum use in {year} (at least {cutoff}{MIN_DOCS} theses), so not ranked."
    row = scored.loc[key]
    used, expected = comparison(row, measure, basis)
    side = scored[np.sign(scored.score) == np.sign(row.score)]
    rank = int((side.score.abs() > abs(row.score)).sum()) + 1
    direction = "more" if row.score > 0 else "less"
    unusual = ("outside" if not row.lower <= row.actual <= row.upper else "within") + " the 95 % prediction interval"
    return (f"In {year}: {used}, {expected}, {unusual}. Rank {rank:,} of {len(side):,} words used {direction} "
            "than expected.")


def lookup_key(word, lang, unit):
    """The table key for a typed or clicked word: the word itself, or its stem when forms are merged."""
    if unit == "stem" and lang in STEMMERS:
        return snowballstemmer.stemmer(STEMMERS[lang]).stemWord(word)
    return word


@callback(
    Output("vocab-detail", "figure"), Output("vocab-detail-title", "children"), Output("vocab-summary", "children"),
    Input("vocab-selected", "data"), Input("vocab-words", "data"),
    Input("vocab-lang", "value"), Input("vocab-year", "value"), Input("vocab-basis", "value"),
    Input("vocab-measure", "value"), Input("vocab-train", "value"), Input("vocab-min-share", "value"),
    Input("vocab-faculties", "value"), Input("vocab-forms", "value"), Input("theme", "data"),
)
def update_detail(selected, words, lang, year, basis, measure, train, min_share, faculties, forms, theme):
    fig = go.Figure(layout=base_layout(theme, height=380))
    if not faculties:
        return fig, "", ""
    min_share = MIN_SHARES[min_share]
    facs, unit = source(faculties, forms)
    f = frequencies(lang, facs, unit)
    table = f.tables[basis]
    if selected:  # a clicked cloud word is already a key; a typed one may need stemming
        key = selected if selected in table.index else lookup_key(selected, lang, unit)
    else:
        key = words[0] if words else None
    if key is None:
        return fig, "", "Click a word in the cloud, or look one up."
    if key not in table.index:
        return fig, f"“{selected}” not found", (
            f"Not in the data for {LANGUAGES[lang]} theses. Only single lowercase word forms used in at least "
            "20 theses across all years are included" + ("" if unit == "stem" else ", and inflected forms are "
                                                         "separate words unless you merge them") + ".")

    freq = table.loc[[key]]
    years = freq.columns.to_numpy()
    fit_years = np.arange(train[0], year + 1)
    baseline = fit(freq.to_numpy(), years, tuple(train))
    expected, lower, upper = (a[0] for a in interval(baseline, fit_years, f.totals[basis].reindex(fit_years),
                                                     SCALE[basis], KIND[basis]))
    docs = basis == "n_docs"
    fig = trend_figure(
        theme, freq.iloc[0].astype("float64"), fit_years, expected, train,
        event=CHATGPT, event_label="ChatGPT released", expected_label="Expected (linear trend)",
        y_title=f"% of {LANGUAGES[lang]} theses using the word" if docs else "Occurrences per million words",
        y_format=".1f" if docs else ",.0f", y_suffix=" % of theses" if docs else " per million words",
        ticksuffix=" %" if docs else "", train_label="training years", lower=lower, upper=upper,
    )
    label = f.labels.get(key, key)
    scored = unexpected_words(lang, tuple(train), year, measure, basis, min_share, facs, unit)
    summary = summarize(key, label, scored, year, measure, basis, min_share)
    if unit == "stem" and key in f.forms.index and f.forms[key] is not None:
        summary += " Forms merged: " + ", ".join(list(f.forms[key])[:8]) + "."
    return fig, f"“{label}” over time", summary


url_sync("vocab-url", SPEC, {
    "lang": ("vocab-lang", "value"), "year": ("vocab-year", "value"), "basis": ("vocab-basis", "value"),
    "measure": ("vocab-measure", "value"), "direction": ("vocab-direction", "value"),
    "train": ("vocab-train", "value"), "n": ("vocab-n", "value"), "min": ("vocab-min-share", "value"),
    "faculties": ("vocab-faculties", "value"), "forms": ("vocab-forms", "value"), "word": ("vocab-selected", "data"),
})
