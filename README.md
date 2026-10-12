# cuni-theses-visualizations

Interactive visualizations of the
[Charles University Bachelor Theses — Style Features](https://huggingface.co/datasets/honza03210/cuni-bachelor-theses-style)
dataset (66,138 bachelor's theses defended at Charles University, 2000–2026), built with Plotly and Dash.

## Visualizations

- **Thesis length over time** — mean or median length per year of defence, filterable by
  faculty. Shows either all selected faculties combined (with the 25th–75th percentile band)
  or up to 8 faculties side by side. Length is measured either as the main body
  (introduction through conclusion) or as the full extracted text.
- **AI-era vocabulary** — a word cloud of the words whose use departs most from their
  pre-ChatGPT trend. For every word, a linear regression of its frequency is fitted on the training
  years (default 2013–2022) and extrapolated to the compared year; red words are used more than
  predicted, blue words less, sized by the size of the gap (difference from, or ratio to, the
  prediction). Frequency is either the share of theses using the word (each thesis counts once) or
  occurrences per million words (every use counts, so repeated words weigh more). Click a word in the cloud, or look up any word, to see its actual history against the trend and its rank among all words.
  Filterable by thesis language (Czech, English, Slovak), compared year and training window; a
  minimum-use cutoff (share of theses using the word) removes rarely used words.
- **Countries mentioned** — a world map of how often each country is mentioned (name, adjective
  and inhabitants, in any grammatical form; patterns in `countries.py`) per million words, in the
  compared years (default 2022–2026) against what the baseline years (default 2013–2021) predict,
  either by linear trend or by their average. Red countries are mentioned more than expected, blue
  less. Click a country for its history and the word forms counted. Czech or English theses.

## Running

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
```

Open http://127.0.0.1:8050. On first start the `theses`, `word_freq_by_year` and
`word_freq_totals` tables (~140 MB) are downloaded from Hugging Face into `data/`.

## Data notes

- Only `usable` theses (text present, not garbled) from 2006 on are included; earlier years
  have very few theses.
- Body length uses only theses whose body was cleanly cut out (`body_method == "intro-back"`),
  as the dataset card recommends for comparisons across years.
- Thesis length includes theses in all languages; the vocabulary page compares theses of one language at a time.
- Words are lowercased word forms without lemmatisation, so Czech inflected forms count separately.
- Country mentions are matched on single word forms, so ambiguous forms are left out (e.g.
  "anglický", "americký"), North and South Korea are merged and Czechia is not shown. The
  matches are cached in `data/country_mentions_*.parquet` (about 30 s to build).
- The current year is incomplete.
