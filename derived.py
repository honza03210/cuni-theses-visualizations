"""Derived tables, built once from the dataset into data/derived/ (see prepare.py).

Two kinds of source data feed the pages:

* the exact yearly word tables of the dataset (`word_freq_by_year`), which count every use of a
  word form in every thesis, and
* per-thesis word counts (`word_counts`), which allow splitting by faculty, merging inflected
  forms of a word and counting theses, but only list a word form when a thesis uses it at least
  twice. Everything built from them is labelled that way on the pages.

Every table is rebuilt when missing, so the app also works without running prepare.py first;
it only starts slower.
"""

import hashlib
import re
from functools import lru_cache
from pathlib import Path

import duckdb
import pandas as pd
import snowballstemmer

from countries import COUNTRIES
from data import CACHE_DIR, FACULTIES, FIRST_YEAR, download

DERIVED_DIR = CACHE_DIR / "derived"
LANG_GROUPS = ("cs", "en", "sk")
MIN_TOTAL_DOCS = 20  # words used by fewer theses over all years are dropped
STEMMERS = {"cs": "czech", "sk": "czech", "en": "english"}  # Slovak is close enough to Czech for a light stemmer
PATTERN = {"cs": 1, "en": 2, "sk": 1}  # which pattern of COUNTRIES each language uses


def _key(*parts) -> str:
    """Short hash of whatever a table depends on, so changing it rebuilds the table."""
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:8]


COUNTRY_KEY = _key(sorted(COUNTRIES.items()))
# AI-style markers: the words most over-used in one year relative to the pre-ChatGPT trend. Chosen on
# 2024 so that the theses of 2025-2026, where declarations are common, did not pick them.
MARKERS = dict(lang="cs", train=(2013, 2022), year=2024, measure="absolute", basis="n_docs", min_share=1.0, n=100)
TABLES = {
    "vocab": f"vocab_{_key(MIN_TOTAL_DOCS)}.parquet",
    "faculty_totals": f"faculty_totals_{_key(FIRST_YEAR, sorted(FACULTIES))}.parquet",
    "words_by_faculty": f"words_by_faculty_{_key(MIN_TOTAL_DOCS, FIRST_YEAR)}.parquet",
    "stems_by_faculty": f"stems_by_faculty_{_key(MIN_TOTAL_DOCS, FIRST_YEAR, STEMMERS)}.parquet",
    "stem_forms": f"stem_forms_{_key(MIN_TOTAL_DOCS, STEMMERS)}.parquet",
    "countries_exact": f"countries_exact_{COUNTRY_KEY}.parquet",
    "countries_by_faculty": f"countries_by_faculty_{_key(COUNTRY_KEY, FIRST_YEAR)}.parquet",
    "country_forms": f"country_forms_{COUNTRY_KEY}.parquet",
    "thesis_style": f"thesis_style_{_key(sorted(MARKERS.items()), FIRST_YEAR)}.parquet",
    "markers": f"markers_{_key(sorted(MARKERS.items()))}.parquet",
}


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    con.execute(f"""
        CREATE TEMP TABLE theses AS
        SELECT item_id, year, fac, prose_n_words,
               CASE WHEN lang IN {LANG_GROUPS} THEN lang ELSE 'other' END AS lang_group
        FROM read_parquet('{download("theses")}')
        WHERE usable AND year >= {FIRST_YEAR} AND fac IN {tuple(FACULTIES)}
    """)
    return con


def _words() -> str:
    return f"read_parquet('{download('word_counts')}')"


def build(name: str) -> Path:
    """Path of a derived table, building it (and the tables built alongside it) when missing."""
    path = DERIVED_DIR / TABLES[name]
    if not path.exists():
        DERIVED_DIR.mkdir(parents=True, exist_ok=True)
        BUILDERS[name]()
    return path


def build_vocab():
    """Exact yearly use of every word form used by >= MIN_TOTAL_DOCS theses, per language."""
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    con.execute(f"""
        COPY (
            WITH w AS (SELECT * FROM read_parquet('{download("word_freq_by_year")}')
                       WHERE lang_group IN {LANG_GROUPS})
            SELECT w.lang_group, w.word, w.year, w.n_docs, w.n_tokens FROM w
            JOIN (SELECT lang_group, word FROM w GROUP BY ALL HAVING sum(n_docs) >= {MIN_TOTAL_DOCS})
            USING (lang_group, word)
        ) TO '{DERIVED_DIR / TABLES["vocab"]}' (FORMAT parquet)
    """)


def build_faculty_tables():
    """Per-thesis word counts aggregated by language, faculty and year: word forms and stems."""
    con = _connect()
    con.execute(f"""
        COPY (SELECT lang_group, fac, year, count(*) AS n_docs, sum(prose_n_words)::BIGINT AS n_tokens
              FROM theses GROUP BY ALL) TO '{DERIVED_DIR / TABLES["faculty_totals"]}' (FORMAT parquet)
    """)
    con.execute(f"""
        CREATE TEMP TABLE counts AS
        SELECT t.lang_group, t.fac, t.year, w.item_id, w.word, w.count
        FROM {_words()} w JOIN theses t USING (item_id)
    """)
    _copy_by_faculty(con, "word", "counts", TABLES["words_by_faculty"])

    vocabulary = con.sql("SELECT DISTINCT lang_group, word FROM counts WHERE lang_group <> 'other'").df()
    stemmers = {lang: snowballstemmer.stemmer(algo) for lang, algo in STEMMERS.items()}
    vocabulary["stem"] = [stemmers[lang].stemWord(word) for lang, word in zip(vocabulary.lang_group, vocabulary.word)]
    con.register("stems", vocabulary)
    con.execute("""
        CREATE TEMP TABLE stem_counts AS
        SELECT c.lang_group, c.fac, c.year, c.item_id, s.stem, sum(c.count) AS count
        FROM counts c JOIN stems s USING (lang_group, word) GROUP BY ALL
    """)
    _copy_by_faculty(con, "stem", "stem_counts", TABLES["stems_by_faculty"])
    con.execute(f"""
        COPY (
            WITH forms AS (
                SELECT c.lang_group, s.stem, c.word, sum(c.count) AS n_tokens
                FROM counts c JOIN stems s USING (lang_group, word) GROUP BY c.lang_group, s.stem, c.word)
            SELECT lang_group, stem, word, n_tokens FROM forms
            QUALIFY row_number() OVER (PARTITION BY lang_group, stem ORDER BY n_tokens DESC) <= 8
        ) TO '{DERIVED_DIR / TABLES["stem_forms"]}' (FORMAT parquet)
    """)


def _copy_by_faculty(con, key: str, source: str, table: str):
    con.execute(f"""
        COPY (
            WITH g AS (SELECT lang_group, fac, year, {key}, count(*) AS n_docs, sum(count) AS n_tokens
                       FROM {source} WHERE lang_group <> 'other' GROUP BY ALL)
            SELECT g.* FROM g
            JOIN (SELECT lang_group, {key} FROM g GROUP BY ALL HAVING sum(n_docs) >= {MIN_TOTAL_DOCS})
            USING (lang_group, {key})
        ) TO '{DERIVED_DIR / table}' (FORMAT parquet)
    """)


@lru_cache(maxsize=None)
def country_matcher(lang: str) -> re.Pattern:
    """One regular expression with a named group per country, for a single pass over the vocabulary."""
    groups = [f"(?P<{iso}>{spec[PATTERN[lang]]})" for iso, spec in COUNTRIES.items() if spec[PATTERN[lang]]]
    return re.compile("|".join(groups))


def match_countries(words, lang: str) -> pd.DataFrame:
    """(word, iso) for every word form that names a country."""
    matcher = country_matcher(lang)
    rows = []
    for word in words:
        m = matcher.fullmatch(word)
        if m:
            rows.append((word, m.lastgroup))
    return pd.DataFrame(rows, columns=["word", "iso"])


def build_country_tables():
    """Country mentions: exact yearly counts, per-thesis counts by faculty, and the forms counted."""
    con = _connect()
    matches = []
    for lang in LANG_GROUPS:
        vocabulary = con.sql(f"""
            SELECT DISTINCT word FROM read_parquet('{download("word_freq_by_year")}') WHERE lang_group = '{lang}'
        """).df().word
        matches.append(match_countries(vocabulary, lang).assign(lang_group=lang))
    con.register("matches", pd.concat(matches, ignore_index=True))
    con.execute(f"""
        COPY (SELECT m.lang_group, m.iso, w.year, sum(w.n_tokens) AS n_tokens
              FROM read_parquet('{download("word_freq_by_year")}') w JOIN matches m USING (lang_group, word)
              WHERE w.year >= {FIRST_YEAR} GROUP BY ALL) TO '{DERIVED_DIR / TABLES["countries_exact"]}' (FORMAT parquet)
    """)
    con.execute(f"""
        COPY (SELECT m.lang_group, m.iso, w.word, sum(w.n_tokens) AS n_tokens
              FROM read_parquet('{download("word_freq_by_year")}') w JOIN matches m USING (lang_group, word)
              GROUP BY ALL) TO '{DERIVED_DIR / TABLES["country_forms"]}' (FORMAT parquet)
    """)
    con.execute(f"""
        COPY (
            WITH per_thesis AS (
                SELECT t.lang_group, t.fac, t.year, w.item_id, m.iso, sum(w.count) AS count
                FROM {_words()} w JOIN theses t USING (item_id) JOIN matches m USING (lang_group, word)
                GROUP BY ALL)
            SELECT lang_group, fac, year, iso, count(*) AS n_docs, sum(count) AS n_tokens
            FROM per_thesis GROUP BY ALL
        ) TO '{DERIVED_DIR / TABLES["countries_by_faculty"]}' (FORMAT parquet)
    """)


def build_thesis_style():
    """Per thesis: AI mentions in the text and reviews, and how much it uses the AI-style markers."""
    from vocab import top_words, unexpected_words  # vocab reads tables built here

    m = MARKERS
    scored = unexpected_words(m["lang"], m["train"], m["year"], m["measure"], m["basis"], m["min_share"])
    markers = top_words(scored, m["n"], "more")[["label", "expected", "actual", "score"]].reset_index(names="word")
    markers.to_parquet(DERIVED_DIR / TABLES["markers"])
    con = _connect()
    con.register("markers", markers[["word"]])
    con.execute(f"""
        COPY (
            WITH use AS (
                SELECT w.item_id, count(*) AS markers_used, sum(w.count) AS marker_tokens
                FROM {_words()} w JOIN markers USING (word) GROUP BY w.item_id)
            SELECT t.item_id, t.year, t.fac, t.lang_group, t.prose_n_words,
                   s.ai_mentions_body, s.ai_mentions_other, s.reviews_mention_ai, s.review_form_mentions_ai,
                   CASE WHEN t.lang_group = '{m["lang"]}' THEN coalesce(u.markers_used, 0) END AS markers_used,
                   CASE WHEN t.lang_group = '{m["lang"]}' THEN coalesce(u.marker_tokens, 0) END AS marker_tokens
            FROM theses t
            JOIN read_parquet('{download("theses")}') s USING (item_id)
            LEFT JOIN use u USING (item_id)
        ) TO '{DERIVED_DIR / TABLES["thesis_style"]}' (FORMAT parquet)
    """)


BUILDERS = {
    "vocab": build_vocab,
    "faculty_totals": build_faculty_tables,
    "words_by_faculty": build_faculty_tables,
    "stems_by_faculty": build_faculty_tables,
    "stem_forms": build_faculty_tables,
    "countries_exact": build_country_tables,
    "countries_by_faculty": build_country_tables,
    "country_forms": build_country_tables,
    "thesis_style": build_thesis_style,
    "markers": build_thesis_style,
}
