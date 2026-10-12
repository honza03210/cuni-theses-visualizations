"""Small layout helpers shared by the pages."""

from dash import html


def control(label, component, wide=False, slider=False):
    classes = "control" + (" control-wide" if wide else "") + (" control-slider" if slider else "")
    return html.Div([html.Label(label, className="control-label"), component], className=classes)
