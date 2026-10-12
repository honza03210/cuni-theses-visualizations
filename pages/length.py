"""Average length of CUNI bachelor theses through the years, by faculty."""

import dash
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html

from data import FACULTIES, MEASURES, load_theses, subset_for, yearly_stats
from theme import CHROME, SERIES, base_layout
from ui import control, parse_query, url_store, url_sync

dash.register_page(__name__, path="/", name="Thesis length", order=0, title="CUNI theses: length over time")

THESES = load_theses()
LAST_YEAR = int(THESES.year.max())  # the current, incomplete year

MAX_COMPARED = 8


SPEC = {
    "faculties": ("list", list(FACULTIES), FACULTIES),
    "measure": ("choice", "body_n_words", MEASURES),
    "stat": ("choice", "mean", ("mean", "median")),
    "view": ("choice", "combined", ("combined", "compare")),
}


def layout(**query):
    """The page; settings in the URL query preset the controls."""
    q = parse_query(query, SPEC)
    return html.Div([
        html.H1("How long are Charles University bachelor theses?"),
        html.P([
            "Length of bachelor theses defended at Charles University by year of defence. Data: ",
            html.A("honza03210/cuni-bachelor-theses-style",
                   href="https://huggingface.co/datasets/honza03210/cuni-bachelor-theses-style"),
            ".",
        ], className="lede"),
        html.Div([
            control("Faculties", dcc.Dropdown(
                id="faculties",
                options=[{"label": name, "value": fac} for fac, name in sorted(FACULTIES.items(), key=lambda kv: kv[1])],
                value=q["faculties"], multi=True, placeholder="Select faculties…",
            ), wide=True),
            control("Length", dcc.RadioItems(
                id="measure", options=[{"label": label, "value": col} for col, label in MEASURES.items()],
                value=q["measure"],
            )),
            control("Average", dcc.RadioItems(
                id="stat", options=[{"label": "Mean", "value": "mean"}, {"label": "Median", "value": "median"}],
                value=q["stat"],
            )),
            control("View", dcc.RadioItems(
                id="view", options=[{"label": "Selected faculties combined", "value": "combined"},
                                    {"label": "Compare faculties", "value": "compare"}],
                value=q["view"],
            )),
        ], className="filters"),
        html.P(id="note", className="note"),
        dcc.Graph(id="chart", config={"displaylogo": False}),
        html.Details([
            html.Summary("Show data table"),
            dash_table.DataTable(id="table", page_size=25, sort_action="native",
                                 style_as_list_view=True, css=[{"selector": ".dash-spreadsheet", "rule": "font-family: inherit"}]),
        ]),
        html.P(id="method", className="method"),
        dcc.Store(id="slots", data={}),
        url_store("length-url"),
    ])


@callback(Output("slots", "data"), Input("faculties", "value"), State("slots", "data"))
def assign_slots(selected, slots):
    """Give each selected faculty a colour slot that it keeps while selected,
    so adding or removing faculties never repaints the others."""
    selected = selected or []
    slots = {fac: slot for fac, slot in slots.items() if fac in selected}
    for fac in selected:
        free = sorted(set(range(MAX_COMPARED)) - set(slots.values()))
        if fac not in slots and free:
            slots[fac] = free[0]
    return slots


def base_figure(theme, measure, stat):
    c = CHROME[theme]
    fig = go.Figure()
    fig.add_vrect(x0=LAST_YEAR - 0.5, x1=LAST_YEAR + 0.5, fillcolor=c["partial"], line_width=0, layer="below",
                  annotation_text="partial year", annotation_position="top left",
                  annotation_font=dict(color=c["muted"], size=12))
    fig.update_layout(base_layout(theme, height=520))
    fig.update_xaxes(dtick=2)
    fig.update_yaxes(title=f"{stat.capitalize()} {MEASURES[measure].split(' (')[0].lower()}", tickformat=",")
    return fig


@callback(
    Output("chart", "figure"), Output("table", "data"), Output("table", "columns"),
    Output("note", "children"), Output("method", "children"),
    Input("slots", "data"), Input("measure", "value"), Input("stat", "value"),
    Input("view", "value"), Input("theme", "data"), State("faculties", "value"),
)
def update(slots, measure, stat, view, theme, selected):
    selected = selected or []
    rows = subset_for(THESES[THESES.fac.isin(selected)], measure)
    fig = base_figure(theme, measure, stat)
    note = ""

    if not selected:
        note = "Select at least one faculty."
        return fig, [], [], note, ""

    if view == "combined":
        stats = yearly_stats(rows, measure, by_faculty=False)
        colour = SERIES[theme][0]
        fig.add_trace(go.Scatter(x=stats.year, y=stats.q3, mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=stats.year, y=stats.q1, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor=CHROME[theme]["band"],
                                 name="Middle 50 % of theses", hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=stats.year, y=stats[stat], mode="lines+markers", name=stat.capitalize(),
            line=dict(color=colour, width=2), marker=dict(size=8, line=dict(width=2, color=CHROME[theme]["surface"])),
            customdata=stats[["n", "q1", "q3"]],
            hovertemplate="<b>%{y:,.0f} words</b><br>%{customdata[0]:,} theses · "
                          "middle 50 %: %{customdata[1]:,.0f}–%{customdata[2]:,.0f}<extra></extra>",
        ))
        label = "all faculties" if len(selected) == len(FACULTIES) else f"{len(selected)} selected faculties"
        note = f"{stat.capitalize()} across {label}; the shaded band spans the 25th to 75th percentile."
        table = stats
    else:
        stats = yearly_stats(rows, measure, by_faculty=True)
        shown = sorted(slots, key=slots.get)
        for fac in shown:
            s = stats[stats.fac == fac]
            fig.add_trace(go.Scatter(
                x=s.year, y=s[stat], mode="lines+markers", name=FACULTIES[fac],
                line=dict(color=SERIES[theme][slots[fac]], width=2),
                marker=dict(size=8, line=dict(width=2, color=CHROME[theme]["surface"])),
                customdata=s[["n"]],
                hovertemplate="<b>%{y:,.0f}</b> · %{fullData.name} (%{customdata[0]:,} theses)<extra></extra>",
            ))
        if len(selected) > len(shown):
            note = (f"Comparing {len(shown)} of {len(selected)} selected faculties — at most {MAX_COMPARED} "
                    "can be compared at once. Remove some to bring in others.")
        table = stats[stats.fac.isin(shown)].assign(fac=lambda d: d.fac.map(FACULTIES))

    table = table.rename(columns={"fac": "faculty"}).round(0)
    columns = [{"name": col, "id": col} if col in ("year", "faculty") else
               {"name": col, "id": col, "type": "numeric", "format": {"specifier": ",.0f"}}
               for col in table.columns]
    method = (f"Based on {len(rows):,} usable theses since 2006"
              + (" whose main body was cleanly identified (introduction through conclusion)." if measure == "body_n_words" else ".")
              + f" Theses in all languages are included. {LAST_YEAR} is incomplete.")
    return fig, table.to_dict("records"), columns, note, method


url_sync("length-url", SPEC, {"faculties": ("faculties", "value"), "measure": ("measure", "value"),
                              "stat": ("stat", "value"), "view": ("view", "value")})
