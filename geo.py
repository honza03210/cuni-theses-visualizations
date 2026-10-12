"""How often countries are mentioned per year, and how that departs from the baseline."""

import hashlib
from functools import lru_cache

import numpy as np
import pandas as pd

from countries import COUNTRIES
from data import CACHE_DIR, download
from vocab import trend

LANGUAGES = {"cs": "Czech", "en": "English"}
PATTERN = {"cs": 1, "en": 2}  # position of each language's pattern in COUNTRIES
MIN_MENTIONS = 20  # countries mentioned (or expected to be) fewer times in the compared years are not shown


@lru_cache(maxsize=None)
def mentions(lang: str) -> tuple[pd.DataFrame, pd.Series, dict[str, pd.Series]]:
    """Mentions of each country per year, words in all theses per year, and the counted word forms.

    Matching the patterns against every word form takes half a minute, so the result is cached
    in data/, keyed by the patterns.
    """
    totals = pd.read_parquet(download("word_freq_totals"))
    totals = totals[totals.lang_group == lang].set_index("year").n_tokens.sort_index()
    key = hashlib.sha1(repr(sorted(COUNTRIES.items())).encode()).hexdigest()[:10]
    cache = CACHE_DIR / f"country_mentions_{lang}_{key}.parquet"
    if not cache.exists():
        match_forms(lang).to_parquet(cache)
    matched = pd.read_parquet(cache)
    counts = matched.pivot_table(index="iso", columns="year", values="n_tokens", aggfunc="sum")
    counts = counts.reindex(index=[iso for iso in COUNTRIES if iso in counts.index], columns=totals.index)
    forms = {iso: g.groupby("word").n_tokens.sum().sort_values(ascending=False) for iso, g in matched.groupby("iso")}
    return counts.fillna(0), totals, forms


def match_forms(lang: str) -> pd.DataFrame:
    """Rows (iso, word, year, n_tokens) for every word form matching a country's pattern."""
    words = pd.read_parquet(download("word_freq_by_year"), columns=["year", "word", "n_tokens"],
                            filters=[("lang_group", "==", lang)])
    vocabulary = words.word.drop_duplicates()
    parts = []
    for iso, spec in COUNTRIES.items():
        pattern = spec[PATTERN[lang]]
        if pattern is not None:
            matched = words[words.word.isin(vocabulary[vocabulary.str.fullmatch(pattern)])]
            parts.append(matched.assign(iso=iso))
    return pd.concat(parts, ignore_index=True)[["iso", "word", "year", "n_tokens"]]


def rates(lang: str) -> pd.DataFrame:
    """Mentions per million words, country x year."""
    counts, totals, _ = mentions(lang)
    return counts / totals * 1_000_000


def expected_rates(lang: str, train: tuple[int, int], years: np.ndarray, baseline: str) -> np.ndarray:
    """Expected mentions per million words, country x year, from the baseline years."""
    r = rates(lang)
    if baseline == "trend":
        intercept, slope = trend(r, train)
        return np.clip(intercept[:, None] + slope[:, None] * years[None, :], 0, None)
    counts, totals, _ = mentions(lang)
    span = list(range(train[0], train[1] + 1))
    mean = counts[span].sum(axis=1) / totals[span].sum() * 1_000_000
    return np.repeat(mean.to_numpy()[:, None], len(years), axis=1)


def compare(lang: str, train: tuple[int, int], period: tuple[int, int], baseline: str,
            measure: str) -> pd.DataFrame:
    """Actual and expected mentions per million words over the compared years, per country.

    Years are pooled by their number of words, so a partial year weighs less.
    measure "absolute": actual - expected; "relative": log2(actual / expected).
    """
    counts, totals, _ = mentions(lang)
    years = np.arange(period[0], period[1] + 1)
    words = totals[years].to_numpy()
    expected = expected_rates(lang, train, years, baseline) @ words / words.sum()
    actual = counts[years].sum(axis=1) / words.sum() * 1_000_000
    df = pd.DataFrame({"expected": expected, "actual": actual.to_numpy(),
                       "mentions": counts[years].sum(axis=1).to_numpy()}, index=counts.index)
    df.insert(0, "country", [COUNTRIES[iso][0] for iso in df.index])
    df = df[np.maximum(df.actual, df.expected) * words.sum() / 1_000_000 >= MIN_MENTIONS]
    if measure == "absolute":
        df["score"] = df.actual - df.expected
    else:
        eps = 1_000_000 / words.sum() * 5  # five mentions keep ratios of rarely named countries finite
        df["score"] = np.log2((df.actual + eps) / (df.expected + eps))
    return df
