"""Page settings in the URL."""

from ui import encode, parse_query

SPEC = {
    "lang": ("choice", "cs", ("cs", "en")),
    "train": ("range", [2013, 2021], (2006, 2021)),
    "facs": ("list", ["a", "b"], ("a", "b", "c")),
    "n": ("int", 50, (10, 100)),
    "word": ("text", None, None),
}


def test_defaults_when_empty():
    assert parse_query({}, SPEC) == {"lang": "cs", "train": [2013, 2021], "facs": ["a", "b"], "n": 50, "word": None}


def test_valid_values_are_read():
    q = parse_query({"lang": "en", "train": "2010-2020", "facs": "c,a", "n": "30", "word": "klíčový"}, SPEC)
    assert q == {"lang": "en", "train": [2010, 2020], "facs": ["c", "a"], "n": 30, "word": "klíčový"}


def test_invalid_values_fall_back():
    q = parse_query({"lang": "xx", "train": "2030-2040", "facs": "c,zz", "n": "abc"}, SPEC)
    assert q["lang"] == "cs" and q["train"] == [2013, 2021] and q["facs"] == ["c"] and q["n"] == 50


def test_round_trip():
    values = {"lang": "en", "train": [2008, 2019], "facs": ["b"], "n": 20, "word": "hlubší"}
    query = {k: encode(v, SPEC[k][0]) for k, v in values.items()}
    assert parse_query(query, SPEC) == values
