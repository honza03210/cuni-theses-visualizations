"""How often countries are mentioned per year, and how that departs from the baseline."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from countries import COUNTRIES
from derived import build
from trends import fit, pooled, prediction_se, t_quantile

LANGUAGES = {"cs": "Czech", "en": "English"}

# What is counted: name -> (label, unit, scale, sampling kind, minimum count to be shown)
COUNTS = {
    "mentions": ("Mentions per million words", "per million words", 1_000_000, "count", 20),
    "theses": ("Share of theses mentioning it", "% of theses", 100, "share", 10),
}
RELATIVE_UNIT = "% of all country mentions"


@dataclass(frozen=True)
class Groups:
    """Counts (country x year) and denominators (per year) for each group that gets its own baseline."""
    counts: dict      # group -> DataFrame country x year
    sizes: dict       # group -> Series year -> denominator
    years: np.ndarray
    scale: float
    kind: str
    unit: str


@lru_cache(maxsize=None)
def _faculty_totals(lang: str) -> pd.DataFrame:
    t = pd.read_parquet(build("faculty_totals"))
    return t[t.lang_group == lang]


@lru_cache(maxsize=32)
def groups(lang: str, count: str, by_faculty: bool, relative: bool) -> Groups:
    """The data behind one combination of options.

    Mentions of all faculties come from the exact yearly word counts. Counting theses, or
    splitting by faculty, needs per-thesis counts, where a thesis mentions a country when it uses
    one of its word forms at least twice.
    """
    _, _, scale, kind, _ = COUNTS[count]
    totals = _faculty_totals(lang)
    years = np.sort(totals.year.unique())
    column = "n_tokens" if count == "mentions" else "n_docs"
    if count == "mentions" and not by_faculty:
        exact = pd.read_parquet(build("countries_exact"))
        exact = exact[exact.lang_group == lang]
        counts = {"all": exact.pivot_table(index="iso", columns="year", values="n_tokens", aggfunc="sum")}
        sizes = {"all": totals.groupby("year")[column].sum()}
    else:
        per_fac = pd.read_parquet(build("countries_by_faculty"))
        per_fac = per_fac[per_fac.lang_group == lang]
        if by_faculty:
            counts = {f: g.pivot_table(index="iso", columns="year", values=column, aggfunc="sum")
                      for f, g in per_fac.groupby("fac")}
            sizes = {f: g.set_index("year")[column] for f, g in totals.groupby("fac")}
        else:
            counts = {"all": per_fac.pivot_table(index="iso", columns="year", values=column, aggfunc="sum")}
            sizes = {"all": totals.groupby("year")[column].sum()}
    isos = [iso for iso in COUNTRIES if any(iso in c.index for c in counts.values())]
    counts = {g: c.reindex(index=isos, columns=years).fillna(0) for g, c in counts.items()}
    sizes = {g: s.reindex(years).fillna(0) for g, s in sizes.items() if g in counts}
    unit = COUNTS[count][1]
    if relative:
        sizes = {g: c.sum(axis=0) for g, c in counts.items()}
        scale, kind, unit = 100, "share", RELATIVE_UNIT
    return Groups(counts, sizes, years, scale, kind, unit)


def _as_list(g: Groups):
    return [(g.counts[k].to_numpy(dtype="float64"), g.sizes[k].to_numpy(dtype="float64"), g.years) for k in g.counts]


@lru_cache(maxsize=64)
def compare(lang: str, train: tuple[int, int], period: tuple[int, int], baseline: str, measure: str,
            count: str = "mentions", by_faculty: bool = False, relative: bool = False) -> pd.DataFrame:
    """Actual and expected rates over the compared years, per country, with a signed score.

    measure "absolute": actual - expected; "relative": log2(actual / expected); "z": the
    difference in standard errors. `z` and `unusual` (outside the 95 % prediction interval)
    are always included.
    """
    g = groups(lang, count, by_faculty, relative)
    actual, expected, se = pooled(_as_list(g), train, period, g.scale, g.kind, baseline)
    isos = next(iter(g.counts.values())).index
    in_period = (g.years >= period[0]) & (g.years <= period[1])
    n_actual = sum(c.to_numpy()[:, in_period].sum(axis=1) for c in g.counts.values())
    size = sum(s.to_numpy()[in_period].sum() for s in g.sizes.values())
    df = pd.DataFrame({"country": [COUNTRIES[iso][0] for iso in isos], "expected": expected, "actual": actual,
                       "count": n_actual, "se": se}, index=pd.Index(isos, name="iso"))
    df["z"] = (df.actual - df.expected) / np.maximum(df.se, 1e-12)
    dof = (train[1] - train[0] + 1) - (2 if baseline == "trend" else 1)
    df["unusual"] = df.z.abs() > t_quantile(dof)
    expected_count = df.expected * size / g.scale
    df = df[np.maximum(df["count"], expected_count) >= COUNTS[count][4]]
    if measure == "absolute":
        df["score"] = df.actual - df.expected
    elif measure == "relative":
        eps = g.scale / max(size, 1) * 5  # five mentions (or theses) keep ratios of rarely named countries finite
        df["score"] = np.log2((df.actual + eps) / (df.expected + eps))
    else:
        df["score"] = df.z
    return df


def history(lang: str, iso: str, train: tuple[int, int], last: int, baseline: str, count: str,
            by_faculty: bool, relative: bool) -> pd.DataFrame:
    """Actual and expected rate per year for one country, with the 95 % prediction interval.

    With by_faculty, each faculty's prediction is weighted by its size in that year.
    """
    g = groups(lang, count, by_faculty, relative)
    years = g.years[(g.years >= train[0]) & (g.years <= last)]
    exp_count = var = 0
    act_count = size = 0
    for key in g.counts:
        c = g.counts[key].to_numpy(dtype="float64")
        s = g.sizes[key].to_numpy(dtype="float64")
        b = fit(c / np.maximum(s, 1) * g.scale, g.years, train, baseline)
        row = g.counts[key].index.get_loc(iso)
        w = g.sizes[key].reindex(years).to_numpy(dtype="float64")
        sub = type(b)(b.intercept[[row]], b.slope[[row]], b.resid_sd[[row]], b.n, b.xbar, b.sxx)
        exp_count = exp_count + sub.predict(years)[0] * w
        var = var + (prediction_se(sub, years, w, g.scale, g.kind)[0] * w) ** 2
        act_count = act_count + g.counts[key].loc[iso].reindex(g.years).to_numpy()
        size = size + s
    w_total = sum(g.sizes[k].reindex(years).to_numpy(dtype="float64") for k in g.counts)
    expected = exp_count / np.maximum(w_total, 1)
    half = t_quantile((train[1] - train[0] + 1) - (2 if baseline == "trend" else 1)) * np.sqrt(var) / np.maximum(w_total, 1)
    actual = pd.Series(act_count / np.maximum(size, 1) * g.scale, index=g.years)
    return pd.DataFrame({"expected": expected, "lower": np.clip(expected - half, 0, None), "upper": expected + half},
                        index=years).join(actual.rename("actual"), how="right")


@lru_cache(maxsize=None)
def forms(lang: str) -> dict[str, pd.Series]:
    """The word forms counted for each country, most used first."""
    f = pd.read_parquet(build("country_forms"))
    f = f[f.lang_group == lang]
    return {iso: g.set_index("word").n_tokens.sort_values(ascending=False) for iso, g in f.groupby("iso")}


def overall(lang: str, train, period, baseline, count, by_faculty) -> float:
    """All countries together: actual over expected, to show a general shift."""
    g = groups(lang, count, by_faculty, False)
    totals = [(c.sum(axis=0, keepdims=True), s, y) for c, s, y in _as_list(g)]
    actual, expected, _ = pooled(totals, train, period, g.scale, g.kind, baseline)
    return float(actual[0] / max(expected[0], 1e-12))

