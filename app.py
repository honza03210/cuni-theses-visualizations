"""Dash app: interactive visualizations of Charles University bachelor theses."""

import os
import threading

import dash
from dash import Dash, Input, Output, clientside_callback, dcc, html

import clouds

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True, title="CUNI theses visualizations")
server = app.server  # for WSGI servers: gunicorn app:server
threading.Thread(target=clouds.warm, daemon=True).start()  # in the background: startup stays fast
PAGES = list(dash.page_registry.values())


def nav_id(page):
    return "nav-" + page["module"].split(".")[-1]


app.layout = html.Main([
    dcc.Location(id="url"),
    html.Nav([
        dcc.Link(page["name"], href=page["relative_path"], className="nav-link", id=nav_id(page))
        for page in PAGES
    ], className="nav"),
    dash.page_container,
    dcc.Store(id="theme", data="light"),
])

# Follow the viewer's OS colour scheme.
clientside_callback(
    """function(_) {
        return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }""",
    Output("theme", "data"),
    Input("theme", "id"),
)


@app.callback([Output(nav_id(p), "className") for p in PAGES], Input("url", "pathname"))
def highlight_nav(pathname):
    return ["nav-link active" if p["relative_path"] == pathname else "nav-link"
            for p in PAGES]


if __name__ == "__main__":
    app.run(debug=os.environ.get("DASH_DEBUG", "1") == "1", port=int(os.environ.get("PORT", 8050)))
