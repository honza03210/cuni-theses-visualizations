"""Download the dataset and build every derived table, so the app starts fast.

    python prepare.py            # everything (about 3 minutes and 560 MB of downloads the first time)
    python prepare.py --clean    # also delete the large raw tables afterwards (for a deploy image)

The app builds whatever is missing on first use, so running this is optional locally.
"""

import sys
import time

import derived
import vocab
from data import CACHE_DIR, download


def main(clean: bool = False):
    started = time.time()
    for table in ("theses", "word_freq_by_year", "word_freq_totals", "word_counts"):
        download(table)
        print(f"downloaded {table}")
    for name in derived.TABLES:
        derived.build(name)
        print(f"built {name}")
    for lang in vocab.LANGUAGES:
        for unit in ("word", "stem"):
            vocab.frequencies(lang, None, unit)
        print(f"cached word frequencies for {lang}")
    if clean:
        # The pages only read the theses table and the derived tables.
        for table in ("word_freq_by_year", "word_counts"):
            (CACHE_DIR / f"{table}.parquet").unlink(missing_ok=True)
        print("removed the raw word tables")
    print(f"done in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main(clean="--clean" in sys.argv)
