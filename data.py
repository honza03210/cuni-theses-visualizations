"""Loading and aggregation of the CUNI bachelor theses dataset."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

DATASET_URL = "https://huggingface.co/datasets/honza03210/cuni-bachelor-theses-style/resolve/main/data"
CACHE_DIR = Path(__file__).parent / "data"

FACULTIES = {
    "etf": "Protestant Theological Faculty",
    "faf": "Faculty of Pharmacy in Hradec Králové",
    "ff": "Faculty of Arts",
    "fhs": "Faculty of Humanities",
    "fsv": "Faculty of Social Sciences",
    "ftvs": "Faculty of Physical Education and Sport",
    "htf": "Hussite Theological Faculty",
    "ktf": "Catholic Theological Faculty",
    "lf1": "First Faculty of Medicine",
    "lf2": "Second Faculty of Medicine",
    "lf3": "Third Faculty of Medicine",
    "lfhk": "Faculty of Medicine in Hradec Králové",
    "mff": "Faculty of Mathematics and Physics",
    "pedf": "Faculty of Education",
    "prf": "Faculty of Science",
}

# Length measures offered in the UI: column -> label.
MEASURES = {
    "body_n_words": "Body words (introduction to conclusion)",
    "full_n_words": "Full text words",
}

FIRST_YEAR = 2006  # very few theses before this year
COLUMNS = ["year", "fac", "usable", "body_method", *MEASURES]


def download(table: str) -> Path:
    """Local path of a dataset table, downloading it on first use."""
    path = CACHE_DIR / f"{table}.parquet"
    if not path.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urlretrieve(f"{DATASET_URL}/{table}.parquet", path)
    return path


def load_theses() -> pd.DataFrame:
    """Return usable theses since FIRST_YEAR."""
    df = pd.read_parquet(download("theses"), columns=COLUMNS)
    df = df[df.usable & (df.year >= FIRST_YEAR) & df.fac.isin(FACULTIES)]
    return df.drop(columns="usable").reset_index(drop=True)


def subset_for(df: pd.DataFrame, measure: str) -> pd.DataFrame:
    """Rows comparable across years for the given measure.

    Body length is only reliable where both ends of the body were found
    (body_method == "intro-back"), as the dataset card recommends.
    """
    if measure == "body_n_words":
        df = df[df.body_method == "intro-back"]
    return df


def yearly_stats(df: pd.DataFrame, measure: str, by_faculty: bool) -> pd.DataFrame:
    """Per-year mean, median, quartiles and count of `measure`."""
    keys = ["fac", "year"] if by_faculty else ["year"]
    grouped = df.groupby(keys)[measure]
    stats = grouped.agg(mean="mean", median="median", n="size")
    stats["q1"] = grouped.quantile(0.25)
    stats["q3"] = grouped.quantile(0.75)
    return stats.reset_index()
