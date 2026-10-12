"""AI-use declarations and AI mentions in theses and reviews, and how declaring theses write."""

from functools import lru_cache

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html

from data import FACULTIES
from derived import MARKERS, build
from theme import CHROME, SEQUENTIAL, SERIES, base_layout
from ui import control, parse_query, url_store, url_sync

dash.register_page(__name__, path="/ai", name="AI declarations", order=3, title="CUNI theses: AI declarations")

FROM_YEAR = 2018
STYLE_FROM = 2015  # the style index is shown from here, to show the pre-ChatGPT level
LANGS = {"all": "All languages", "cs": "Czech", "en": "English", "sk": "Slovak"}
# Indicator -> (label, test on the per-thesis table)
INDICATORS = {
    "other": ("AI terms outside the body (mostly declarations)", lambda t: t.ai_mentions_other > 0),
    "body": ("AI terms in the body", lambda t: t.ai_mentions_body > 0),
    "reviews": ("A reviewer mentions AI", lambda t: t.reviews_mention_ai.fillna(False).astype(bool)),
    "form": ("The review form asks about AI", lambda t: t.review_form_mentions_ai.fillna(False).astype(bool)),
}


@lru_cache(maxsize=None)
def theses() -> pd.DataFrame:
    t = pd.read_parquet(build("thesis_style"))
    flags = pd.DataFrame({k: test(t) for k, (_, test) in INDICATORS.items()})
    return pd.concat([t, flags], axis=1).assign(marker_rate=lambda d: d.marker_tokens / d.prose_n_words * 1e4)


@lru_cache(maxsize=None)
def markers() -> pd.DataFrame:
    return pd.read_parquet(build("markers"))


LAST_YEAR = int(theses().year.max())

SPEC = {
    "faculties": ("list", list(FACULTIES), FACULTIES),
    "lang": ("choice", "all", LANGS),
}


def layout(**query):
    """The page; settings in the URL query preset the controls."""
    q = parse_query(query, SPEC)
    return html.Div([
        html.H1("Do theses declare AI use?"),
        html.P(
            "Theses defended from 2025 on often state whether AI tools were used. The dataset has no flag for this, "
            "but it counts mentions of AI terms (ChatGPT, GPT, umělá inteligence, jazykový model, LLM, nástroje AI, …) "
            "outside the main body, where they are mostly such declarations. They include declarations that no AI "
            "was used, so they show that AI is addressed, not that it was used.",
            className="lede",
        ),
        html.Div([
            control("Faculties", dcc.Dropdown(
                id="ai-faculties", multi=True, value=q["faculties"], placeholder="Select faculties…",
                options=[{"label": n, "value": f} for f, n in sorted(FACULTIES.items(), key=lambda kv: kv[1])],
            ), wide=True),
            control("Language of the thesis", dcc.RadioItems(
                id="ai-lang", value=q["lang"], options=[{"label": v, "value": k} for k, v in LANGS.items()],
            )),
        ], className="filters"),
        html.H2("AI in theses and their reviews"),
        html.P(id="ai-trend-note", className="note"),
        dcc.Graph(id="ai-trend", config={"displaylogo": False}),
        html.H2("AI terms outside the body, by faculty"),
        html.P("Share of a faculty's theses that mention AI outside the body. Hover a cell for the counts.",
               className="note"),
        dcc.Graph(id="ai-heatmap", config={"displaylogo": False}),
        html.H2("Do theses that address AI write differently?"),
        html.P(id="ai-style-note", className="note"),
        dcc.Graph(id="ai-style", config={"displaylogo": False}),
        html.Details([
            html.Summary("The marker words"),
            html.P(id="ai-markers", className="method"),
        ]),
        html.Details([
            html.Summary("Show data table"),
            dash_table.DataTable(id="ai-table", page_size=25, sort_action="native", style_as_list_view=True,
                                 css=[{"selector": ".dash-spreadsheet", "rule": "font-family: inherit"}]),
        ]),
        html.P(
            "AI mentions are keyword matches, so a thesis about AI counts too. Reviews are the supervisor's and "
            "opponent's reports; the review form counts when the faculty's form template itself has an AI question. "
            "The style index counts uses of the marker words in forms used at least twice in a thesis.",
            className="method",
        ),
        url_store("ai-url"),
    ])


def selection(faculties, lang) -> pd.DataFrame:
    t = theses()
    t = t[t.fac.isin(faculties or []) & (t.year >= FROM_YEAR)]
    return t if lang == "all" else t[t.lang_group == lang]


@callback(
    Output("ai-trend", "figure"), Output("ai-trend-note", "children"), Output("ai-table", "data"),
    Output("ai-table", "columns"),
    Input("ai-faculties", "value"), Input("ai-lang", "value"), Input("theme", "data"),
)
def update_trend(faculties, lang, theme):
    t = selection(faculties, lang)
    c = CHROME[theme]
    fig = go.Figure(layout=base_layout(theme, height=440, margin=dict(l=64, r=200, t=24, b=56)))
    if t.empty:
        return fig, "Select at least one faculty.", [], []
    shares = t.groupby("year")[list(INDICATORS)].mean() * 100
    counts = t.groupby("year").size()
    for i, (key, (label, _)) in enumerate(INDICATORS.items()):
        colour = SERIES[theme][i]
        fig.add_trace(go.Scatter(
            x=shares.index, y=shares[key], mode="lines+markers", name=label,
            line=dict(color=colour, width=2), marker=dict(size=8, line=dict(width=2, color=c["surface"])),
            customdata=np.c_[t.groupby("year")[key].sum().reindex(shares.index), counts.reindex(shares.index)],
            hovertemplate=f"<b>%{{y:.1f}} %</b> {label.lower()} (%{{customdata[0]:,}} of %{{customdata[1]:,}})<extra></extra>",
        ))
        fig.add_annotation(x=shares.index[-1], y=shares[key].iloc[-1], text=label.split(" (")[0], xanchor="left",
                           xshift=14, showarrow=False, font=dict(color=c["ink2"], size=12))
    fig.add_vrect(x0=LAST_YEAR - 0.5, x1=LAST_YEAR + 0.5, fillcolor=c["partial"], line_width=0, layer="below",
                  annotation_text="partial year", annotation_position="bottom left",
                  annotation_font=dict(color=c["muted"], size=12))
    fig.update_xaxes(dtick=1, range=[FROM_YEAR - 0.3, LAST_YEAR + 0.5])
    fig.update_yaxes(title="% of theses", ticksuffix=" %")
    latest = shares.loc[shares.index.max()]
    note = (f"{len(t):,} theses since {FROM_YEAR}. In {shares.index.max()} so far, {latest.other:.0f} % mention AI "
            f"outside the body, against about {shares.loc[:2022, 'other'].mean():.0f} % before 2023.")
    table = shares.round(1).reset_index()
    table.columns = ["year"] + [label for label, _ in INDICATORS.values()]
    columns = [{"name": col, "id": col} for col in table.columns]
    return fig, note, table.to_dict("records"), columns


@callback(Output("ai-heatmap", "figure"), Input("ai-faculties", "value"), Input("ai-lang", "value"),
          Input("theme", "data"))
def update_heatmap(faculties, lang, theme):
    t = selection(faculties, lang)
    t = t[t.year >= 2022]
    fig = go.Figure(layout=base_layout(theme, height=max(260, 34 * len(faculties or []) + 90),
                                       margin=dict(l=280, r=24, t=16, b=48), hovermode="closest"))
    if t.empty:
        return fig
    grid = t.groupby(["fac", "year"]).other.agg(["mean", "sum", "size"]).reset_index()
    order = grid.groupby("fac")["mean"].max().sort_values().index  # most declarations at the top
    share = grid.pivot(index="fac", columns="year", values="mean").reindex(order) * 100
    hits = grid.pivot(index="fac", columns="year", values="sum").reindex(order)
    size = grid.pivot(index="fac", columns="year", values="size").reindex(order)
    ramp = SEQUENTIAL[theme]
    fig.add_trace(go.Heatmap(
        z=share.to_numpy(), x=share.columns, y=[FACULTIES[f] for f in share.index],
        colorscale=[[i / (len(ramp) - 1), col] for i, col in enumerate(ramp)], zmin=0, zmax=100,
        xgap=2, ygap=2, customdata=np.dstack([hits.to_numpy(), size.to_numpy()]),
        hovertemplate="<b>%{z:.1f} %</b> · %{y}, %{x}<br>%{customdata[0]:,.0f} of %{customdata[1]:,.0f} theses"
                      "<extra></extra>",
        colorbar=dict(ticksuffix=" %", thickness=12, outlinewidth=0, tickfont=dict(color=CHROME[theme]["muted"])),
    ))
    fig.update_xaxes(dtick=1, showspikes=False, showgrid=False)
    fig.update_yaxes(showgrid=False, rangemode=None)
    return fig


@callback(
    Output("ai-style", "figure"), Output("ai-style-note", "children"), Output("ai-markers", "children"),
    Input("ai-faculties", "value"), Input("theme", "data"),
)
def update_style(faculties, theme):
    t = theses()
    t = t[t.fac.isin(faculties or []) & (t.lang_group == MARKERS["lang"]) & (t.year >= STYLE_FROM)]
    c = CHROME[theme]
    fig = go.Figure(layout=base_layout(theme, height=420))
    words = ", ".join(markers().word)
    if t.empty:
        return fig, "Select at least one faculty.", words
    groups = {False: ("No AI terms outside the body", SERIES[theme][0]),
              True: ("AI terms outside the body", SERIES[theme][1])}
    stats = t.groupby(["other", "year"]).marker_rate.agg(["mean", "std", "size"]).reset_index()
    stats = stats[stats["size"] >= 20]  # too few theses give no meaningful mean
    for flag, (label, colour) in groups.items():
        s = stats[stats.other == flag]
        half = 1.96 * s["std"] / np.sqrt(s["size"])
        fig.add_trace(go.Scatter(x=s.year, y=s["mean"] + half, mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=s.year, y=s["mean"] - half, mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=_alpha(colour, 0.18),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=s.year, y=s["mean"], mode="lines+markers", name=label,
            line=dict(color=colour, width=2), marker=dict(size=8, line=dict(width=2, color=c["surface"])),
            customdata=np.c_[half, s["size"]],
            hovertemplate=f"<b>%{{y:.0f}}</b> marker words per 10,000 · {label.lower()} "
                          "(±%{customdata[0]:.0f}, %{customdata[1]:,} theses)<extra></extra>",
        ))
    fig.add_vline(x=2022.87, line=dict(color=c["axis"], width=1, dash="dot"), annotation_text="ChatGPT released",
                  annotation_position="top left", annotation_font=dict(color=c["muted"], size=12))
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(title="Marker words per 10,000 words")
    last = stats[stats.year == stats.year.max()].set_index("other")["mean"]
    gap = f" In {stats.year.max()}: {last.get(True, np.nan):.0f} vs {last.get(False, np.nan):.0f}." if len(last) == 2 else ""
    note = (f"Czech theses. The style index counts how often a thesis uses the {MARKERS['n']} words most "
            f"over-used in {MARKERS['year']} relative to their {MARKERS['train'][0]}–{MARKERS['train'][1]} trend "
            "(klíčový, naznačuje, zdůrazňuje, …), per 10,000 words. Bands show the 95 % confidence interval of "
            f"the mean.{gap} Theses that address AI may also differ in topic and faculty, so this is a "
            "correlation, not proof of use.")
    return fig, note, words


def _alpha(hex_colour: str, alpha: float) -> str:
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


url_sync("ai-url", SPEC, {"faculties": ("ai-faculties", "value"), "lang": ("ai-lang", "value")})
