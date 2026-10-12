"""Word cloud layouts: computed once per set of settings and cached.

Laying out a cloud takes most of a second, and most visitors look at the same few views, so
layouts are kept in memory (for any settings) and prepare.py writes the default views to disk,
from where the app loads them at startup. A layout only depends on the words and the width;
colours and hover text are applied per request, so light and dark mode share one layout.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from PIL import Image, ImageDraw, ImageFont
from wordcloud import WordCloud

from derived import DERIVED_DIR, TABLES
from vocab import LANGUAGES, last_year, top_words, unexpected_words

FONT_PATH = str(Path(__file__).parent / "assets" / "DroidSansMono.ttf")
CLOUD_W, CLOUD_H = 1030, 560  # at full width; narrower screens get a smaller cloud with smaller words
WIDTH_STEP = 40               # widths are rounded down to this, so similar screens share a layout
MIN_SHARES = [0, 0.1, 0.5, 1, 2, 5, 10]  # minimum-use cutoffs offered, % of theses
LAYOUT_VERSION = 1            # bump when the layout parameters below change


class Settings(NamedTuple):
    lang: str
    year: int
    basis: str
    measure: str
    direction: str
    train: tuple
    n: int
    min_share: float
    faculties: tuple | None  # None: all faculties
    unit: str                # "word" or "stem"
    width: int


DEFAULT = dict(lang="cs", year=2025, basis="n_docs", measure="absolute", direction="both", train=(2013, 2022),
               n=50, min_share=MIN_SHARES[3], faculties=None, unit="word")


def width_bucket(width) -> int:
    """The layout width for a screen: at most CLOUD_W, at least 300, in steps of WIDTH_STEP."""
    width = int(width or CLOUD_W)
    return CLOUD_W if width >= CLOUD_W else max(300, width // WIDTH_STEP * WIDTH_STEP)


def height_for(width: int) -> int:
    return int(max(360, CLOUD_H * width / CLOUD_W))


def _disk_path() -> Path:
    # Layouts depend on the derived word tables and on the layout code, so both are in the name.
    key = hashlib.sha1(repr((sorted(TABLES.items()), LAYOUT_VERSION)).encode()).hexdigest()[:8]
    return DERIVED_DIR / f"cloud_layouts_{key}.json"


def _load_prepared() -> dict:
    try:
        return {tuple_key: [tuple(p) for p in placed] for tuple_key, placed in
                ((tuple(json.loads(k)), v) for k, v in json.loads(_disk_path().read_text()).items())}
    except (OSError, ValueError):
        return {}


def _key(settings: Settings) -> tuple:
    """A JSON-friendly form of the settings, used as the key on disk."""
    s = settings._asdict()
    s["train"] = list(s["train"])
    s["faculties"] = list(s["faculties"]) if s["faculties"] is not None else None
    return tuple(json.dumps(v) for v in s.values())


PREPARED = _load_prepared()


def layout(settings: Settings) -> list[tuple]:
    """(key, x, y, font size) for each placed word, centres in pixels from the top left."""
    prepared = PREPARED.get(_key(settings))
    return prepared if prepared is not None else _compute(settings)


@lru_cache(maxsize=256)
def _compute(settings: Settings) -> list[tuple]:
    s = settings
    scored = unexpected_words(s.lang, s.train, s.year, s.measure, s.basis, s.min_share, s.faculties, s.unit)
    words = top_words(scored, s.n, s.direction)
    if words.empty:
        return []
    width, height = s.width, height_for(s.width)
    scale = width / CLOUD_W
    cloud = WordCloud(width=width, height=height, font_path=FONT_PATH, prefer_horizontal=1.0,
                      max_font_size=max(24, int(64 * scale)), min_font_size=max(8, int(11 * scale)),
                      relative_scaling=0.7, margin=max(3, int(6 * scale)), random_state=7, mode="RGBA",
                      background_color=None)
    label_of = dict(zip(words.index, words.label))
    cloud.generate_from_frequencies({label_of[k]: abs(v) for k, v in words.score.items()})
    key_of = {label: key for key, label in label_of.items()}

    draw = ImageDraw.Draw(Image.new("L", (1, 1)))
    placed = []
    for (label, _), size, (top, left), _, _ in cloud.layout_:
        x0, y0, x1, y1 = draw.textbbox((0, 0), label, font=ImageFont.truetype(FONT_PATH, size), anchor="lt")
        placed.append((key_of[label], left + (x1 - x0) / 2, top + (y1 - y0) / 2, size))
    return placed


def default_views() -> list[Settings]:
    """The views most visitors see: the default settings for every language and compared year, at
    common widths (desktop, tablet, phones), and the other measures and bases for Czech on desktop."""
    views = []
    for lang in LANGUAGES:
        for year in range(2023, last_year() + 1):
            for width in (CLOUD_W, 720, 360, 320):
                views.append(Settings(**dict(DEFAULT, lang=lang, year=year), width=width))
    for change in (dict(measure="relative"), dict(measure="z"), dict(basis="n_tokens"), dict(unit="stem")):
        views.append(Settings(**dict(DEFAULT, **change), width=CLOUD_W))
    return views


def warm():
    """Score the words of the default views, so their first visitors do not wait for it."""
    for s in default_views():
        if s.width == CLOUD_W:
            unexpected_words(s.lang, s.train, s.year, s.measure, s.basis, s.min_share, s.faculties, s.unit)


def prepare():
    """Compute the default views and write them to disk for the app to load at startup."""
    layouts = {json.dumps(list(_key(s))): _compute(s) for s in default_views()}
    _disk_path().write_text(json.dumps(layouts))
    PREPARED.update(_load_prepared())
    return len(layouts)
