"""Expected vs actual word usage: per-word baselines extrapolated past the training years."""

from dataclasses import dataclass
from functools import lru_cache

import duckdb
import numpy as np
import pandas as pd

from data import FACULTIES
from derived import DERIVED_DIR, MIN_TOTAL_DOCS, TABLES, build
from trends import fit, interval, prediction_se

LANGUAGES = {"cs": "Czech", "en": "English", "sk": "Slovak"}

# How a word's frequency in a year is measured: column of the word tables -> (label, unit).
BASES = {
    "n_docs": ("Share of theses using the word", "% of theses"),
    "n_tokens": ("Occurrences per million words", "per million words"),
}
SCALE = {"n_docs": 100, "n_tokens": 1_000_000}  # frequencies are stored in the unit above
KIND = {"n_docs": "share", "n_tokens": "count"}  # for the sampling noise of each basis

MIN_DOCS = 10  # a word must be used by this many theses in the compared year, or be expected to be


@dataclass(frozen=True)
class Frequencies:
    tables: dict           # basis -> DataFrame, word (or stem) x year, in the basis' unit
    totals: pd.DataFrame   # year -> n_docs (theses), n_tokens (words)
    labels: pd.Series      # key -> word shown (for a stem, its most used form)
    forms: pd.Series       # key -> the word forms merged into it (stems only)
    per_thesis: bool       # True when built from per-thesis counts (forms used at least twice)


def _load(table: str, key: str, lang: str, faculties) -> pd.DataFrame:
    """Yearly counts of `key` summed over the faculties, from a derived table (fast, in DuckDB)."""
    fac_filter = f"AND fac IN {tuple(faculties) + ('',)}" if faculties is not None else ""
    con = duckdb.connect()
    return con.sql(f"""
        SELECT {key}, year, sum(n_docs)::DOUBLE AS n_docs, sum(n_tokens)::DOUBLE AS n_tokens
        FROM read_parquet('{build(table)}') WHERE lang_group = '{lang}' {fac_filter}
        GROUP BY ALL
    """).df()


def _as_tables(long: pd.DataFrame, key: str, totals: pd.DataFrame) -> dict:
    wide = long.set_index([key, "year"]).unstack("year", fill_value=0)
    tables = {b: wide[b].reindex(columns=totals.index, fill_value=0) for b in BASES}
    keep = tables["n_docs"].sum(axis=1) >= MIN_TOTAL_DOCS
    return {b: (t[keep] / totals[b] * SCALE[b]).astype("float32") for b, t in tables.items()}


def _totals(lang: str, faculties) -> pd.DataFrame:
    totals = pd.read_parquet(build("faculty_totals"))
    totals = totals[(totals.lang_group == lang) & totals.fac.isin(faculties)]
    return totals.groupby("year")[list(BASES)].sum().sort_index()


def last_year() -> int:
    return int(pd.read_parquet(build("faculty_totals")).year.max())


@lru_cache(maxsize=16)
def frequencies(lang: str, faculties: tuple | None = None, unit: str = "word") -> Frequencies:
    """Yearly word frequencies; the all-faculty tables are cached on disk (see `_frequencies`)."""
    if faculties is not None:
        return _frequencies(lang, faculties, unit)
    source = TABLES["vocab" if unit == "word" else "stems_by_faculty"]
    path = DERIVED_DIR / f"frequencies_{lang}_{unit}_{source}"
    if not path.exists():
        f = _frequencies(lang, None, unit)
        wide = pd.concat({b: t for b, t in f.tables.items()}, axis=1)
        wide.columns = [f"{b}:{y}" for b, y in wide.columns]
        wide.assign(label=f.labels, forms=f.forms.reindex(wide.index) if unit == "stem" else None).to_parquet(path)
    wide = pd.read_parquet(path)
    tables = {b: wide[[c for c in wide.columns if c.startswith(b + ":")]].rename(columns=lambda c: int(c.split(":")[1]))
              for b in BASES}
    forms = wide.forms if unit == "stem" else pd.Series(dtype=object)
    return Frequencies(tables, _totals(lang, FACULTIES), wide.label, forms, per_thesis=unit == "stem")


def _frequencies(lang: str, faculties: tuple | None = None, unit: str = "word") -> Frequencies:
    """Yearly word frequencies for a language.

    With all faculties and single word forms these are the dataset's exact yearly counts. Picking
    faculties or merging inflected forms (unit "stem") needs per-thesis counts, which only include
    forms a thesis uses at least twice.
    """
    if faculties is None and unit == "word":
        long = _load("vocab", "word", lang, None)
        totals = _totals(lang, FACULTIES)
        tables = _as_tables(long, "word", totals)
        keys = tables["n_docs"].index
        return Frequencies(tables, totals, pd.Series(keys, index=keys), pd.Series(dtype=object), per_thesis=False)

    facs = list(faculties or FACULTIES)
    source = "words_by_faculty" if unit == "word" else "stems_by_faculty"
    long = _load(source, unit, lang, facs)
    totals = _totals(lang, facs)
    tables = _as_tables(long, unit, totals)
    keys = tables["n_docs"].index
    if unit == "word":
        return Frequencies(tables, totals, pd.Series(keys, index=keys), pd.Series(dtype=object), per_thesis=True)
    forms = duckdb.connect().sql(f"""
        SELECT stem, list(word ORDER BY n_tokens DESC) AS forms FROM read_parquet('{build("stem_forms")}')
        WHERE lang_group = '{lang}' GROUP BY stem
    """).df().set_index("stem").forms.reindex(keys)
    return Frequencies(tables, totals, forms.str[0].fillna(pd.Series(keys, index=keys)), forms, per_thesis=True)


def years_of(freq: pd.DataFrame) -> np.ndarray:
    return freq.columns.to_numpy()


@lru_cache(maxsize=32)
def unexpected_words(lang: str, train: tuple[int, int], year: int, measure: str, basis: str,
                     min_share: float, faculties: tuple | None = None, unit: str = "word") -> pd.DataFrame:
    """Every eligible word with its expected and actual frequency in `year` and a signed score.

    measure "absolute": actual - expected, in the basis' unit (percentage points of theses,
    or occurrences per million words); "relative": log2(actual / expected); "z": the difference
    in standard errors of the prediction (how unusual it is given the baseline's scatter).
    A word is eligible when at least `min_share` percent of the year's theses, and at least
    MIN_DOCS theses, use it or are expected to. Eligibility is always decided by the number of
    theses using the word, so a word repeated many times in a single thesis cannot qualify on its own.
    """
    f = frequencies(lang, faculties, unit)
    docs = f.tables["n_docs"]
    years = years_of(docs)
    docs_expected = fit(docs.to_numpy(), years, train).predict(year)[:, 0]
    doc_share = np.maximum(docs[year].to_numpy(dtype="float64"), docs_expected) / SCALE["n_docs"]
    eligible = (doc_share * f.totals.n_docs[year] >= MIN_DOCS) & (doc_share * 100 >= min_share)

    freq = f.tables[basis][eligible]
    baseline = fit(freq.to_numpy(), years, train)
    size = [f.totals[basis][year]]
    expected, lower, upper = (a[:, 0] for a in interval(baseline, [year], size, SCALE[basis], KIND[basis]))
    df = pd.DataFrame({"expected": expected, "actual": freq[year].to_numpy(dtype="float64"),
                       "lower": lower, "upper": upper}, index=freq.index)
    if measure == "absolute":
        df["score"] = df.actual - df.expected
    elif measure == "relative":
        # Smoothing keeps ratios of near-zero frequencies finite: one thesis, or one occurrence per million words.
        eps = SCALE["n_docs"] / f.totals.n_docs[year] if basis == "n_docs" else 1.0
        df["score"] = np.log2((df.actual + eps) / (df.expected + eps))
    else:
        se = prediction_se(baseline, [year], size, SCALE[basis], KIND[basis])[:, 0]
        df["score"] = (df.actual - df.expected) / np.maximum(se, 1e-9)
    df.insert(0, "label", f.labels.reindex(df.index).to_numpy())
    return df


def top_words(df: pd.DataFrame, n: int, direction: str) -> pd.DataFrame:
    """The n most over-used and/or n most under-used words."""
    parts = []
    if direction in ("both", "more"):
        parts.append(df[df.score > 0].nlargest(n, "score"))
    if direction in ("both", "less"):
        parts.append(df[df.score < 0].nsmallest(n, "score"))
    return pd.concat(parts)
