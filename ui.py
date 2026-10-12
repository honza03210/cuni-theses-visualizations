"""Small layout helpers shared by the pages, and keeping page settings in the URL."""

import json

from dash import Input, Output, clientside_callback, dcc, html


def control(label, component, wide=False, slider=False):
    classes = "control" + (" control-wide" if wide else "") + (" control-slider" if slider else "")
    return html.Div([html.Label(label, className="control-label"), component], className=classes)


# URL state
#
# Each page describes its settings as {param: (kind, default, allowed)}; kinds are "choice" (one
# of `allowed`), "list" (a subset of `allowed`), "range" (two ints within `allowed` = (lo, hi)),
# "int" (within `allowed`) and "text". The page's layout reads them from the query string with
# parse_query, and url_sync writes changes back with history.pushState, so links can be shared
# and the back button steps through earlier settings (Dash re-renders the page from the URL).
# Settings at their default value are left out of the URL.


def encode(value, kind) -> str:
    if value is None:
        return ""
    if kind == "range":
        return f"{value[0]}-{value[1]}"
    if kind == "list":
        return ",".join(map(str, value))
    return str(value)


def parse_query(query: dict, spec: dict) -> dict:
    """Setting values from URL query parameters; anything missing or invalid gets its default."""
    values = {}
    for param, (kind, default, allowed) in spec.items():
        raw, value = query.get(param), default
        try:
            if raw is None:
                pass
            elif kind == "choice" and raw in allowed:
                value = raw
            elif kind == "list":
                items = [item for item in raw.split(",") if item in allowed] if raw else []
                value = items
            elif kind == "range":
                lo, hi = (int(v) for v in raw.split("-"))
                if allowed[0] <= lo <= hi <= allowed[1]:
                    value = [lo, hi]
            elif kind == "int" and allowed[0] <= int(raw) <= allowed[1]:
                value = int(raw)
            elif kind == "text":
                value = raw[:100]
        except ValueError:
            pass
        values[param] = value
    return values


def url_sync(sink: str, spec: dict, inputs: dict):
    """Write the page's settings to the URL whenever they change.

    `inputs` maps each param of `spec` to the (component id, property) holding it; `sink` is the
    id of a dcc.Store in the page's layout that receives the query string.
    """
    params = list(inputs)
    kinds = [spec[p][0] for p in params]
    defaults = [encode(spec[p][1], spec[p][0]) for p in params]
    clientside_callback(
        f"""function(...values) {{
            const params = {json.dumps(params)}, kinds = {json.dumps(kinds)}, defaults = {json.dumps(defaults)};
            const query = new URLSearchParams();
            values.forEach((v, i) => {{
                let s = '';
                if (v === null || v === undefined) s = '';
                else if (kinds[i] === 'range') s = v[0] + '-' + v[1];
                else if (kinds[i] === 'list') s = v.join(',');
                else s = String(v);
                if (s !== defaults[i]) query.set(params[i], s);
            }});
            const search = query.toString() ? '?' + query.toString() : '';
            if (search !== window.location.search) {{
                window.history.pushState(null, '', window.location.pathname + search);
            }}
            return search;
        }}""",
        Output(sink, "data"),
        [Input(component, prop) for component, prop in inputs.values()],
    )


def url_store(sink: str):
    return dcc.Store(id=sink)
