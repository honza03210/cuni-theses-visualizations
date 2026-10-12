"""Expected vs actual word usage: per-word linear trends extrapolated past the training years."""

from functools import lru_cache

import numpy as np
import pandas as pd

from data import download

LANGUAGES = {"cs": "Czech", "en": "English", "sk": "Slovak"}

# How a word's frequency in a year is measured: column of the word tables -> (label, unit).
BASES = {
    "n_docs": ("Share of theses using the word", "% of theses"),
    "n_tokens": ("Occurrences per million words", "per million words"),
}
SCALE = {"n_docs": 100, "n_tokens": 1_000_000}  # frequencies are stored in the unit above

MIN_TOTAL_DOCS = 20  # drop words used in fewer theses over all years
MIN_DOCS = 10        # a word must be used by this many theses in the compared year, or be expected to be


@lru_cache(maxsize=None)
def frequencies(lang: str) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Word x year frequency tables for each basis (in its unit), and per-year totals."""
    totals = pd.read_parquet(download("word_freq_totals"))
    totals = totals[totals.lang_group == lang].set_index("year")[list(BASES)].sort_index()
    words = pd.read_parquet(download("word_freq_by_year"), columns=["year", "word", *BASES],
                            filters=[("lang_group", "==", lang)])
    tables = {}
    for basis in BASES:
        counts = words.pivot_table(index="word", columns="year", values=basis, fill_value=0)
        tables[basis] = counts.reindex(columns=totals.index, fill_value=0)
    keep = tables["n_docs"].sum(axis=1) >= MIN_TOTAL_DOCS
    tables = {basis: (t[keep] / totals[basis] * SCALE[basis]).astype("float32") for basis, t in tables.items()}
    return tables, totals


def trend(freq: pd.DataFrame, train: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares intercept and slope of each word's frequency over the training years."""
    years = np.arange(train[0], train[1] + 1)
    y = freq[years].to_numpy(dtype="float64")
    x = years - years.mean()
    slope = (y - y.mean(axis=1, keepdims=True)) @ x / (x @ x)
    intercept = y.mean(axis=1) - slope * years.mean()
    return intercept, slope


def predict(freq: pd.DataFrame, train: tuple[int, int], year: int) -> np.ndarray:
    intercept, slope = trend(freq, train)
    return np.clip(intercept + slope * year, 0, None)


@lru_cache(maxsize=32)
def unexpected_words(lang: str, train: tuple[int, int], year: int, measure: str, basis: str,
                     min_share: float) -> pd.DataFrame:
    """Every eligible word with its expected and actual frequency in `year` and a signed score.

    measure "absolute": actual - expected, in the basis' unit (percentage points of theses,
    or occurrences per million words).
    measure "relative": log2(actual / expected).
    A word is eligible when at least `min_share` percent of the year's theses, and at least
    MIN_DOCS theses, use it or are expected to. Eligibility is always decided by the number of
    theses using the word, so a word repeated many times in a single thesis cannot qualify on its own.
    """
    tables, totals = frequencies(lang)
    docs = tables["n_docs"]
    doc_share = np.maximum(docs[year].to_numpy(dtype="float64"), predict(docs, train, year)) / SCALE["n_docs"]
    eligible = (doc_share * totals.n_docs[year] >= MIN_DOCS) & (doc_share * 100 >= min_share)

    freq = tables[basis][eligible]
    df = pd.DataFrame({"expected": predict(freq, train, year), "actual": freq[year].to_numpy(dtype="float64")},
                      index=freq.index)
    if measure == "absolute":
        df["score"] = df.actual - df.expected
    else:
        # Smoothing keeps ratios of near-zero frequencies finite: one thesis, or one occurrence per million words.
        eps = SCALE["n_docs"] / totals.n_docs[year] if basis == "n_docs" else 1.0
        df["score"] = np.log2((df.actual + eps) / (df.expected + eps))
    return df


def top_words(df: pd.DataFrame, n: int, direction: str) -> pd.DataFrame:
    """The n most over-used and/or n most under-used words."""
    parts = []
    if direction in ("both", "more"):
        parts.append(df[df.score > 0].nlargest(n, "score"))
    if direction in ("both", "less"):
        parts.append(df[df.score < 0].nsmallest(n, "score"))
    return pd.concat(parts)
