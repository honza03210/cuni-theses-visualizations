"""Country patterns: they catch the forms they should and none of the look-alikes found in the data."""

import pytest

from derived import match_countries

MATCHES = {
    "cs": {
        "UKR": ["ukrajina", "ukrajině", "ukrajinský", "ukrajinci"],
        "RUS": ["rusko", "ruska", "ruské", "rusové"],
        "DEU": ["německo", "německé", "němci"],
        "GBR": ["británie", "britský", "anglie", "skotsku"],
        "USA": ["usa", "američané"],
        "LTU": ["litva", "litvě", "litevský"],
        "ETH": ["etiopie", "etiopský"],
        "CYP": ["kypr", "kypru", "kyperský"],
        "BRA": ["brazílie", "brazilský"],
        "ISL": ["islandu", "islandský"],
        "TUR": ["turecko", "turecký", "turci"],
        "BIH": ["bosna", "bosně", "hercegovina", "bosenský"],
        "KOR": ["korea", "korejský", "jihokorejský"],
    },
    "en": {
        "UKR": ["ukraine", "ukrainian"],
        "GBR": ["britain", "uk", "england", "scottish"],
        "USA": ["usa", "america", "americans"],
    },
}

# Word forms that look like a country but are not one; each was found in the data.
NOT_COUNTRIES = {
    "cs": ["litvínov", "litvínově", "etiopatogeneze", "kyprý", "kyprá", "island", "tureckiová", "rakušanová",
           "bosničová", "nepálených", "braziliensis", "anglický", "anglicky", "americký", "malta", "peru",
           "finance", "norma", "činnost", "rusovlasý", "řeka", "jar", "kuře"],
    "en": ["us", "african", "turkeys", "american", "english"],
}


@pytest.mark.parametrize("lang", MATCHES)
def test_country_forms_match(lang):
    for iso, words in MATCHES[lang].items():
        found = match_countries(words, lang).set_index("word").iso
        assert list(found.reindex(words)) == [iso] * len(words), (iso, found.to_dict())


@pytest.mark.parametrize("lang", NOT_COUNTRIES)
def test_look_alikes_do_not_match(lang):
    found = match_countries(NOT_COUNTRIES[lang], lang)
    assert found.empty, found.to_dict("records")


def test_each_word_names_one_country():
    # Patterns are tried in one alternation, so overlapping patterns would silently pick the first.
    import re
    from countries import COUNTRIES
    for lang, col in (("cs", 1), ("en", 2)):
        patterns = {iso: re.compile(spec[col]) for iso, spec in COUNTRIES.items() if spec[col]}
        for words in MATCHES[lang].values():
            for word in words:
                hits = [iso for iso, p in patterns.items() if p.fullmatch(word)]
                assert len(hits) == 1, (word, hits)
