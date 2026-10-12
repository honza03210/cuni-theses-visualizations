"""World map: which countries theses mention more or less since 2022 than the baseline predicts."""

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, callback, dash_table, dcc, html

from charts import trend_figure
from countries import COUNTRIES
from data import FIRST_YEAR
from geo import COUNTS, LANGUAGES, compare, forms, groups, history, overall
from theme import CHROME, base_layout
from ui import control, parse_query, url_store, url_sync

dash.register_page(__name__, path="/countries", name="Countries mentioned", order=2,
                   title="CUNI theses: countries mentioned")

PERIOD_START = 2022
LAST_YEAR = int(groups("cs", "mentions", False, False).years.max())
RATIO_CAP = 2  # colour scale for ratios stops at 4x / one quarter (beyond shows as the end colour)
Z_CAP = 6      # and for surprise at six standard errors

# Diverging scale: blue (mentioned less) - neutral grey - red (mentioned more).
DIVERGING = {
    "light": [[0, "#1c5cab"], [0.25, "#9ec5f4"], [0.5, "#f0efec"], [0.75, "#f2aaa9"], [1, "#c8302f"]],
    "dark": [[0, "#3987e5"], [0.25, "#30598f"], [0.5, "#4a4a46"], [0.75, "#9a4545"], [1, "#e66767"]],
}
MEASURES = {
    "relative": "Ratio to expected",
    "absolute": "Difference from expected",
    "z": "Surprise (standard errors from expected)",
}


def layout(**query):
    """The page; settings in the URL query (e.g. ?lang=en&baseline=mean) preset the controls."""
    q = parse_query(query, SPEC)
    return html.Div([
        html.H1("Which countries do theses mention more since 2022?"),
        html.P(
            "Mentions of each country (its name, adjective and inhabitants, in any grammatical form). The baseline "
            "years predict how often a country would be mentioned in the compared years; red countries are "
            "mentioned more than that, blue ones less.",
            className="lede",
        ),
        html.Div([
            control("Language of the thesis", dcc.RadioItems(
                id="geo-lang", value=q["lang"], options=[{"label": v, "value": k} for k, v in LANGUAGES.items()],
            )),
            control("Count", dcc.RadioItems(
                id="geo-count", value=q["count"],
                options=[{"label": label, "value": k} for k, (label, *_) in COUNTS.items()],
            )),
            control("Expected from", dcc.RadioItems(
                id="geo-baseline", value=q["baseline"], options=[
                    {"label": "Linear trend of the baseline years", "value": "trend"},
                    {"label": "Average of the baseline years", "value": "mean"},
                ],
            )),
            control("Change measured as", dcc.RadioItems(
                id="geo-measure", value=q["measure"], options=[{"label": v, "value": k} for k, v in MEASURES.items()],
            )),
            control("Adjust", dcc.Checklist(
                id="geo-adjust", value=q["adjust"], options=[
                    {"label": "Compare each faculty with its own baseline", "value": "faculty"},
                    {"label": "Relative to all country mentions", "value": "relative"},
                ],
            )),
            control("Region", dcc.RadioItems(
                id="geo-scope", value=q["scope"], options=[{"label": "World", "value": "world"},
                                                          {"label": "Europe", "value": "europe"}],
            )),
            control("Baseline years", dcc.RangeSlider(
                id="geo-train", min=FIRST_YEAR, max=PERIOD_START - 1, step=1, value=q["train"],
                pushable=2, marks={y: str(y) for y in range(FIRST_YEAR, PERIOD_START, 3)},
            ), slider=True),
            control("Compared years", dcc.RangeSlider(
                id="geo-period", min=PERIOD_START, max=LAST_YEAR, step=1, value=q["period"],
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
        dcc.Store(id="geo-selected", data=q["country"]),
        url_store("geo-url"),
    ])


SPEC = {
    "lang": ("choice", "cs", LANGUAGES),
    "count": ("choice", "mentions", COUNTS),
    "baseline": ("choice", "trend", ("trend", "mean")),
    "measure": ("choice", "relative", MEASURES),
    "adjust": ("list", [], ("faculty", "relative")),
    "scope": ("choice", "world", ("world", "europe")),
    "train": ("range", [2013, PERIOD_START - 1], (FIRST_YEAR, PERIOD_START - 1)),
    "period": ("range", [PERIOD_START, LAST_YEAR], (PERIOD_START, LAST_YEAR)),
    "country": ("choice", None, COUNTRIES),
}


def years_label(period):
    label = f"{period[0]}–{period[1]}" if period[0] != period[1] else str(period[0])
    return label + (f" ({LAST_YEAR} still in progress)" if period[1] == LAST_YEAR else "")


def change_text(row, measure, unit):
    if measure == "relative":
        return f"{2 ** row.score:.2f}× expected"
    if measure == "absolute":
        return f"{row.score:+,.1f} {unit}"
    return f"{row.z:+.1f} standard errors"


def unusual_text(row):
    return "outside the 95 % prediction interval" if row.unusual else "within normal variation"


def map_figure(df, measure, scope, unit, theme):
    c = CHROME[theme]
    if measure == "relative":
        z, limit = df.score.clip(-RATIO_CAP, RATIO_CAP), RATIO_CAP
        ticks = list(range(-RATIO_CAP, RATIO_CAP + 1))
        colorbar = dict(tickvals=ticks, ticktext=[f"{2 ** t:g}×" if t >= 0 else f"1/{2 ** -t:g}" for t in ticks])
    elif measure == "z":
        z, limit = df.score.clip(-Z_CAP, Z_CAP), Z_CAP
        colorbar = dict(tickvals=[-6, -4, -2, 0, 2, 4, 6], ticktext=["−6", "−4", "−2", "0", "+2", "+4", "+6"])
    else:
        z, limit = df.score, max(df.score.abs().max(), 1e-9)
        colorbar = dict(tickformat=",.1f")
    hover = [f"<b>{r.country}</b><br>{r.actual:,.2f} {unit}<br>expected {r.expected:,.2f} · "
             f"{change_text(r, measure, unit)}<br>{unusual_text(r)}" for r in df.itertuples()]
    fig = go.Figure(go.Choropleth(
        locations=df.index, z=z, locationmode="ISO-3", colorscale=DIVERGING[theme], zmid=0,
        zmin=-limit, zmax=limit, marker=dict(line=dict(color=c["axis"], width=0.5)),
        text=hover, hovertemplate="%{text}<extra></extra>",
        colorbar=dict(colorbar, thickness=12, len=0.7, outlinewidth=0, tickfont=dict(color=c["muted"]), x=1.0),
    ))
    fig.update_layout(base_layout(theme, height=540, margin=dict(l=0, r=0, t=0, b=0), hovermode="closest"))
    fig.update_geos(
        scope=scope, projection_type="natural earth", showframe=False, showcoastlines=False,
        showcountries=True, countrycolor=c["axis"], showland=True, landcolor=c["surface"],
        showocean=False, showlakes=False, bgcolor=c["surface"],
        lataxis_range=[-58, 85] if scope == "world" else None,
    )
    return fig


SETTINGS = [Input("geo-lang", "value"), Input("geo-count", "value"), Input("geo-baseline", "value"),
            Input("geo-measure", "value"), Input("geo-adjust", "value"), Input("geo-train", "value"),
            Input("geo-period", "value")]


def options(lang, count, baseline, measure, adjust, train, period):
    adjust = adjust or []
    return dict(lang=lang, train=tuple(train), period=tuple(period), baseline=baseline, measure=measure,
                count=count, by_faculty="faculty" in adjust, relative="relative" in adjust)


@callback(
    Output("geo-map", "figure"), Output("geo-table", "data"), Output("geo-table", "columns"),
    Output("geo-note", "children"), Output("geo-method", "children"),
    *SETTINGS, Input("geo-scope", "value"), Input("theme", "data"),
)
def update_map(lang, count, baseline, measure, adjust, train, period, scope, theme):
    o = options(lang, count, baseline, measure, adjust, train, period)
    df = compare(**o).sort_values("score", ascending=False)
    unit = groups(lang, count, o["by_faculty"], o["relative"]).unit
    fig = map_figure(df, measure, scope, unit, theme)

    ratio = overall(lang, o["train"], o["period"], baseline, count, o["by_faculty"])
    note = (f"{LANGUAGES[lang]} theses, {years_label(period)} against the baseline {train[0]}–{train[1]}. "
            f"All countries together are {'mentioned' if count == 'mentions' else 'named in theses'} "
            f"{ratio:.2f}× as often as expected"
            + ("; the map shows each country's share of all mentions, so this overall shift is taken out."
               if o["relative"] else ", so a general shift in what theses are about colours most of the map one way.")
            + f" {int(df.unusual.sum())} of {len(df)} countries are outside their 95 % prediction interval."
            " Outlined countries without colour are not counted or too rarely mentioned.")

    table = df.drop(columns=["se"]).reset_index().rename(columns={
        "iso": "code", "expected": f"expected ({unit})", "actual": f"actual ({unit})",
        "count": "mentions" if count == "mentions" else "theses", "z": "standard errors",
        "unusual": "outside 95 % interval", "score": MEASURES[measure].lower()})
    table["outside 95 % interval"] = table["outside 95 % interval"].map({True: "yes", False: "no"})
    text_columns = ("code", "country", "outside 95 % interval")
    columns = [{"name": c, "id": c} if c in text_columns else
               {"name": c, "id": c, "type": "numeric", "format": {"specifier": ",.0f" if c in ("mentions", "theses") else ",.2f"}}
               for c in table.columns]
    source = ("Mentions come from the dataset's exact yearly word counts." if count == "mentions" and not o["by_faculty"]
              else "This view uses per-thesis word counts, so a thesis mentions a country when it uses one of its "
                   "word forms at least twice.")
    method = (f"{len(df)} of {len(COUNTRIES)} countries shown: those mentioned, or expected to be, at least "
              f"{COUNTS[count][4]} times in the compared years. {source} Comparing each faculty with its own "
              "baseline and adding the results up removes the effect of faculties growing or shrinking. "
              "Ambiguous forms such as “anglický” or “americký” are left out, North and South Korea cannot be "
              "told apart, and Czechia itself is not shown.")
    return fig, table.round(2).to_dict("records"), columns, note, method


@callback(Output("geo-selected", "data"), Input("geo-map", "clickData"), prevent_initial_call=True)
def select_country(click):
    return click["points"][0]["location"] if click else dash.no_update


@callback(
    Output("geo-detail", "figure"), Output("geo-detail-title", "children"), Output("geo-summary", "children"),
    Input("geo-selected", "data"), *SETTINGS, Input("theme", "data"),
)
def update_detail(selected, lang, count, baseline, measure, adjust, train, period, theme):
    o = options(lang, count, baseline, measure, adjust, train, period)
    df = compare(**o)
    iso = selected if selected in df.index else df.score.idxmax()
    row = df.loc[iso]
    g = groups(lang, count, o["by_faculty"], o["relative"])
    h = history(lang, iso, o["train"], period[1], baseline, count, o["by_faculty"], o["relative"])
    fitted = h.expected.dropna()
    fig = trend_figure(
        theme, h.actual, fitted.index.to_numpy(), fitted.to_numpy(), train, above=row.score > 0,
        expected_label="Expected (linear trend)" if baseline == "trend" else "Expected (baseline average)",
        y_title=g.unit[0].upper() + g.unit[1:], y_format=",.2f" if o["relative"] or count == "theses" else ",.1f",
        y_suffix=f" {g.unit}", highlight=period,
        lower=h.lower.dropna().to_numpy(), upper=h.upper.dropna().to_numpy(),
    )
    noun = "mentions" if count == "mentions" else "theses"
    summary = (f"{years_label(period)}: {row.actual:,.2f} {g.unit} ({int(row['count']):,} {noun}), expected "
               f"{row.expected:,.2f} · {change_text(row, measure, g.unit)}, {unusual_text(row)}. "
               f"Most counted forms: {', '.join(forms(lang)[iso].index[:8])}.")
    return fig, f"{row.country} over time", summary


url_sync("geo-url", SPEC, {
    "lang": ("geo-lang", "value"), "count": ("geo-count", "value"), "baseline": ("geo-baseline", "value"),
    "measure": ("geo-measure", "value"), "adjust": ("geo-adjust", "value"), "scope": ("geo-scope", "value"),
    "train": ("geo-train", "value"), "period": ("geo-period", "value"), "country": ("geo-selected", "data"),
})
