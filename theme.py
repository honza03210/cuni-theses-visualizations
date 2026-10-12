"""Colours and shared Plotly styling, in light and dark variants."""

SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
# Diverging poles: more used than expected / less used than expected.
MORE = {"light": "#e34948", "dark": "#e66767"}
LESS = {"light": "#2a78d6", "dark": "#3987e5"}

CHROME = {
    "light": {"ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "axis": "#c3c2b7",
              "band": "rgba(42,120,214,0.14)", "partial": "rgba(137,135,129,0.08)", "surface": "#fcfcfb",
              "interval": "rgba(137,135,129,0.22)"},
    "dark": {"ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a", "axis": "#383835",
             "band": "rgba(57,135,229,0.20)", "partial": "rgba(137,135,129,0.10)", "surface": "#1a1a19",
             "interval": "rgba(195,194,183,0.16)"},
}
# Sequential (one hue, light -> dark) for magnitudes such as shares in a heatmap.
SEQUENTIAL = {
    "light": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    "dark": ["#0d366b", "#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#b7d3f6"],  # near zero recedes
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def base_layout(theme, **overrides):
    """Layout shared by all charts: surfaces, ink, recessive grid, unified hover."""
    c = CHROME[theme]
    axis = dict(gridcolor=c["grid"], linecolor=c["axis"], tickfont=dict(color=c["muted"]),
                title_font=dict(color=c["muted"]), zeroline=False)
    layout = dict(
        margin=dict(l=64, r=24, t=24, b=56),
        paper_bgcolor=c["surface"], plot_bgcolor=c["surface"],
        font=dict(family=FONT, color=c["ink"], size=13),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=c["surface"], bordercolor=c["axis"], font=dict(color=c["ink"])),
        legend=dict(orientation="h", y=-0.14, x=0, font=dict(color=c["ink"])),
        xaxis=dict(axis, showgrid=False, showspikes=True, spikemode="across", spikethickness=1,
                   spikecolor=c["axis"], spikedash="solid"),
        yaxis=dict(axis, rangemode="tozero", showline=False),
    )
    layout.update(overrides)
    return layout
