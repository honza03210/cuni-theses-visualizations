"""Charts shared by the pages."""

import plotly.graph_objects as go

from theme import CHROME, LESS, MORE, base_layout


def trend_figure(theme, history, fit_years, fitted, train, *, expected_label, y_title, y_format, y_suffix,
                 event=None, event_label=None, ticksuffix="", highlight=None, train_label="baseline years",
                 above=None):
    """A yearly series against what the baseline predicts.

    The actual line is red when it is above the prediction and blue when below, matching the
    colours of the overview: `above` says which, by default judged at the last year. `highlight`
    optionally shades the compared years and `event` marks a moment with a dotted line.
    """
    c = CHROME[theme]
    fig = go.Figure(layout=base_layout(theme, height=380))
    if above is None:
        above = history[fit_years[-1]] > fitted[-1]
    colour = MORE[theme] if above else LESS[theme]
    fig.add_vrect(x0=train[0] - 0.5, x1=train[1] + 0.5, fillcolor=c["partial"], line_width=0, layer="below",
                  annotation_text=train_label, annotation_position="top left",
                  annotation_font=dict(color=c["muted"], size=12))
    if highlight:
        fig.add_vrect(x0=highlight[0] - 0.5, x1=highlight[1] + 0.5, fillcolor=c["band"], line_width=0,
                      layer="below", annotation_text="compared years", annotation_position="top right",
                      annotation_font=dict(color=c["muted"], size=12))
    if event is not None:
        fig.add_vline(x=event, line=dict(color=c["axis"], width=1, dash="dot"),
                      annotation_text=event_label, annotation_position="bottom right",
                      annotation_font=dict(color=c["muted"], size=12))
    fig.add_trace(go.Scatter(
        x=fit_years, y=fitted, mode="lines", name=expected_label,
        line=dict(color=c["muted"], width=2, dash="dash"),
        hovertemplate=f"expected %{{y:{y_format}}}{y_suffix}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=history.index, y=history.values, mode="lines+markers", name="Actual",
        line=dict(color=colour, width=2), marker=dict(size=8, line=dict(width=2, color=c["surface"])),
        hovertemplate=f"<b>%{{y:{y_format}}}{y_suffix}</b><extra></extra>",
    ))
    fig.update_xaxes(dtick=2)
    fig.update_yaxes(title=y_title, ticksuffix=ticksuffix, tickformat=",")
    return fig
