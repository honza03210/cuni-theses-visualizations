"""World map: which countries theses mention more or less since 2022 than the baseline predicts."""

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html

from charts import trend_figure
from countries import COUNTRIES
from geo import LANGUAGES, MIN_MENTIONS, compare, expected_rates, mentions, rates
from theme import CHROME, base_layout
from ui import control

dash.register_page(__name__, path="/countries", name="Countries mentioned", order=2,
                   title="CUNI theses: countries mentioned")

FIRST_YEAR = 2006
PERIOD_START = 2022
_, TOTALS, _ = mentions("cs")
LAST_YEAR = int(TOTALS.index.max())
RATIO_CAP = 2  # colour scale for ratios stops at 4x / one quarter (beyond shows as the end colour)

# Diverging scale: blue (mentioned less) - neutral grey - red (mentioned more).
DIVERGING = {
    "light": [[0, "#1c5cab"], [0.25, "#9ec5f4"], [0.5, "#f0efec"], [0.75, "#f2aaa9"], [1, "#c8302f"]],
    "dark": [[0, "#3987e5"], [0.25, "#30598f"], [0.5, "#4a4a46"], [0.75, "#9a4545"], [1, "#e66767"]],
}

layout = html.Div([
    html.H1("Which countries do theses mention more since 2022?"),
    html.P(
        "Mentions of each country (its name, adjective and inhabitants, in any grammatical form) per million "
        "words of thesis text. The baseline years predict how often a country would be mentioned in the "
        "compared years; red countries are mentioned more than that, blue ones less.",
        className="lede",
    ),
    html.Div([
        control("Language of the thesis", dcc.RadioItems(
            id="geo-lang", options=[{"label": v, "value": k} for k, v in LANGUAGES.items()], value="cs",
        )),
        control("Expected from", dcc.RadioItems(
            id="geo-baseline", value="trend", options=[
                {"label": "Linear trend of the baseline years", "value": "trend"},
                {"label": "Average of the baseline years", "value": "mean"},
            ],
        )),
        control("Change measured as", dcc.RadioItems(
            id="geo-measure", value="relative", options=[
                {"label": "Ratio to expected", "value": "relative"},
                {"label": "Difference (mentions per million words)", "value": "absolute"},
            ],
        )),
        control("Region", dcc.RadioItems(
            id="geo-scope", value="world", options=[{"label": "World", "value": "world"},
                                                    {"label": "Europe", "value": "europe"}],
        )),
        control("Baseline years", dcc.RangeSlider(
            id="geo-train", min=FIRST_YEAR, max=PERIOD_START - 1, step=1, value=[2013, PERIOD_START - 1],
            pushable=2, marks={y: str(y) for y in range(FIRST_YEAR, PERIOD_START, 3)},
        ), slider=True),
        control("Compared years", dcc.RangeSlider(
            id="geo-period", min=PERIOD_START, max=LAST_YEAR, step=1, value=[PERIOD_START, LAST_YEAR],
            marks={y: f"{y}*" if y == LAST_YEAR else str(y) for y in range(PERIOD_START, LAST_YEAR + 1)},
        ), slider=True),
    ], className="filters"),
    html.P(id="geo-note", className="note"),
    dcc.Graph(id="geo-map", config={"displaylogo": False, "scrollZoom": False}),
    html.H2(id="geo-detail-title"),
    html.P(id="geo-summary", className="note"),
    dcc.Graph(id="geo-detail", config={"displaylogo": False}),
    html.Details([
        html.Summary("Show data table"),
        dash_table.DataTable(id="geo-table", page_size=25, sort_action="native", style_as_list_view=True,
                             css=[{"selector": ".dash-spreadsheet", "rule": "font-family: inherit"}]),
    ]),
    html.P(id="geo-method", className="method"),
])


def years_label(period):
    label = f"{period[0]}–{period[1]}" if period[0] != period[1] else str(period[0])
    return label + (f" ({LAST_YEAR} still in progress)" if period[1] == LAST_YEAR else "")


def change_text(score, measure):
    return f"{2 ** score:.2f}× expected" if measure == "relative" else f"{score:+,.1f} per million words"


def map_figure(df, measure, scope, theme):
    c = CHROME[theme]
    if measure == "relative":
        z = df.score.clip(-RATIO_CAP, RATIO_CAP)
        limit = RATIO_CAP
        ticks = list(range(-RATIO_CAP, RATIO_CAP + 1))
        colorbar = dict(tickvals=ticks, ticktext=[f"{2 ** t:g}×" if t >= 0 else f"1/{2 ** -t:g}" for t in ticks])
    else:
        z = df.score
        limit = max(df.score.abs().max(), 1e-9)
        colorbar = dict(tickformat=",.0f", ticksuffix=" pm")
    hover = [f"<b>{r.country}</b><br>{r.actual:,.1f} mentions per million words<br>"
             f"expected {r.expected:,.1f} · {change_text(r.score, measure)}" for r in df.itertuples()]
    fig = go.Figure(go.Choropleth(
        locations=df.index, z=z, locationmode="ISO-3", colorscale=DIVERGING[theme], zmid=0,
        zmin=-limit, zmax=limit, marker=dict(line=dict(color=c["axis"], width=0.5)),
        text=hover, hovertemplate="%{text}<extra></extra>", customdata=df.index,
        colorbar=dict(colorbar, thickness=12, len=0.7, outlinewidth=0,
                      tickfont=dict(color=c["muted"]), x=1.0),
    ))
    fig.update_layout(base_layout(theme, height=540, margin=dict(l=0, r=0, t=0, b=0), hovermode="closest"))
    fig.update_geos(
        scope=scope, projection_type="natural earth", showframe=False, showcoastlines=False,
        showcountries=True, countrycolor=c["axis"], showland=True, landcolor=c["surface"],
        showocean=False, showlakes=False, bgcolor=c["surface"],
        lataxis_range=[-58, 85] if scope == "world" else None,
    )
    return fig


@callback(
    Output("geo-map", "figure"), Output("geo-table", "data"), Output("geo-table", "columns"),
    Output("geo-note", "children"), Output("geo-method", "children"),
    Input("geo-lang", "value"), Input("geo-baseline", "value"), Input("geo-measure", "value"),
    Input("geo-scope", "value"), Input("geo-train", "value"), Input("geo-period", "value"),
    Input("theme", "data"),
)
def update_map(lang, baseline, measure, scope, train, period, theme):
    df = compare(lang, tuple(train), tuple(period), baseline, measure).sort_values("score", ascending=False)
    fig = map_figure(df, measure, scope, theme)

    overall = df.actual.sum() / df.expected.sum()
    note = (f"{LANGUAGES[lang]} theses, {years_label(period)} against the baseline {train[0]}–{train[1]}. "
            f"All countries together are mentioned {overall:.2f}× as often as expected, so a general shift "
            "in what theses are about colours most of the map one way. Outlined countries without colour are "
            "not counted or too rarely mentioned.")

    table = df.reset_index().rename(columns={"iso": "code"})
    table = table.rename(columns={"expected": "expected per million words", "actual": "actual per million words",
                                  "score": "log2 ratio" if measure == "relative" else "change per million words"})
    columns = [{"name": c, "id": c} if c in ("code", "country") else
               {"name": c, "id": c, "type": "numeric", "format": {"specifier": ",.0f" if c == "mentions" else ",.2f"}}
               for c in table.columns]
    shown = len(df)
    method = (f"{shown} of {len(COUNTRIES)} countries shown: those mentioned, or expected to be, at least "
              f"{MIN_MENTIONS} times in the compared years. A country counts as mentioned when a word form matches "
              "its pattern (e.g. Ukrajina, Ukrajině, ukrajinský, Ukrajinci); ambiguous forms such as “anglický” "
              "or “americký” are left out, North and South Korea cannot be told apart, and Czechia itself is "
              "not shown. Single words only: names like “Spojené státy” count through “USA” or “Američané”.")
    return fig, table.round(2).to_dict("records"), columns, note, method


@callback(
    Output("geo-detail", "figure"), Output("geo-detail-title", "children"), Output("geo-summary", "children"),
    Input("geo-map", "clickData"), Input("geo-lang", "value"), Input("geo-baseline", "value"),
    Input("geo-measure", "value"), Input("geo-train", "value"), Input("geo-period", "value"),
    Input("theme", "data"),
)
def update_detail(click, lang, baseline, measure, train, period, theme):
    df = compare(lang, tuple(train), tuple(period), baseline, measure)
    clicked = click["points"][0]["location"] if click else None
    iso = clicked if clicked in df.index else df.score.idxmax()

    r = rates(lang)
    fit_years = np.arange(train[0], period[1] + 1)
    fitted = expected_rates(lang, tuple(train), fit_years, baseline)[r.index.get_loc(iso)]
    row = df.loc[iso]
    fig = trend_figure(
        theme, r.loc[iso], fit_years, fitted, train, above=row.score > 0,
        expected_label="Expected (linear trend)" if baseline == "trend" else "Expected (baseline average)",
        y_title="Mentions per million words", y_format=",.1f", y_suffix=" per million words",
        highlight=period,
    )
    _, _, forms = mentions(lang)
    top_forms = ", ".join(forms[iso].index[:8])
    summary = (f"{years_label(period)}: {row.actual:,.1f} mentions per million words "
               f"({int(row.mentions):,} mentions), expected {row.expected:,.1f} · {change_text(row.score, measure)}. "
               f"Most counted forms: {top_forms}.")
    return fig, f"{row.country} over time", summary
