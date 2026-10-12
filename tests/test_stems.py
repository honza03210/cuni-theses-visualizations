"""Merging inflected forms."""

import snowballstemmer

from derived import STEMMERS


def stems(lang, words):
    s = snowballstemmer.stemmer(STEMMERS[lang])
    return {s.stemWord(w) for w in words}


def test_czech_adjective_forms_merge():
    assert len(stems("cs", ["klíčový", "klíčová", "klíčové", "klíčových", "klíčovým", "klíčovou"])) == 1
    assert len(stems("cs", ["naznačuje", "naznačují"])) == 1


def test_different_words_stay_apart():
    assert len(stems("cs", ["klíčový", "hlubší", "bohužel"])) == 3


def test_english_forms_merge():
    assert len(stems("en", ["delve", "delves", "delving"])) == 1
