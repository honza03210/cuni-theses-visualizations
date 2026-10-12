# cuni-theses-visualizations

Interactive visualizations of the
[Charles University Bachelor Theses — Style Features](https://huggingface.co/datasets/honza03210/cuni-bachelor-theses-style)
dataset (66,138 bachelor's theses defended at Charles University, 2000–2026), built with Plotly and Dash.

## Pages

- **Thesis length** — mean or median length per year of defence, filterable by faculty. Shows
  either all selected faculties combined (with the 25th–75th percentile band) or up to 8
  faculties side by side. Length is the main body (introduction through conclusion) or the full
  extracted text.
- **AI-era vocabulary** — a word cloud of the words whose use departs most from their
  pre-ChatGPT trend. For every word a linear regression of its frequency is fitted on the
  training years (default 2013–2022) and extrapolated to the compared year; red words are used
  more than predicted, blue less, sized by the difference, the ratio, or the *surprise* (the
  difference in standard errors of the prediction). Frequency is the share of theses using the
  word or occurrences per million words. Filter by faculty, merge inflected forms (a light
  stemmer: klíčový, klíčové, klíčovou… count as one word), set a minimum use. Click a word in the
  cloud, or look up any word, for its history with the 95 % prediction interval and its rank.
- **Countries mentioned** — a world map of how often each country is mentioned (name, adjective
  and inhabitants, in any grammatical form; patterns in `countries.py`) in the compared years
  (default 2022–2026) against what the baseline years (default 2013–2021) predict, by linear
  trend or average. Count mentions per million words or the share of theses mentioning a
  country; optionally compare each faculty with its own baseline (removing the effect of a
  changing faculty mix) or measure each country's share of all country mentions. Hover shows
  whether a country is outside its 95 % prediction interval; click for its history.
- **AI declarations** — the share of theses mentioning AI terms outside the main body (mostly
  AI-use declarations, including declared non-use), in the body, in the supervisor's and
  opponent's reviews and in the review form, by year and faculty; and a style index comparing
  how much theses that address AI use the words most over-used since ChatGPT.

All settings are kept in the URL, so a view can be shared as a link and the browser's back
button steps through earlier settings.

## Running

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python prepare.py   # optional: downloads the data and builds every cache (~3 min)
.venv/bin/python app.py
```

Open http://127.0.0.1:8050. Without `prepare.py`, the app downloads the dataset (~560 MB) and
builds the caches in `data/` the first time each page needs them.

### Deploying

The `Dockerfile` builds every cache at image build time and keeps only what the app reads
(about 400 MB); it serves the app with gunicorn on `$PORT`. `render.yaml` is a
[Render](https://render.com) blueprint for it. The app uses about 1.1 GB of memory once every view
has been opened.

```sh
docker build -t cuni-theses .
docker run -p 8050:8050 cuni-theses
```

### Tests

```sh
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests
```

Unit tests cover the country patterns (including every look-alike found in the data, such as
"Litvínov" for Lithuania), the trend statistics (prediction-interval coverage, pooling over
faculties), URL settings and stemming. `tests/test_browser.py` drives the app in Chrome with
Playwright: clicking and hovering words, looking up words, clicking countries, URL state and the
back button, and a phone-width layout. It is skipped if Chrome is not installed.

## Code

| file | role |
|---|---|
| `app.py` | the Dash app, navigation, theme detection, `server` for WSGI |
| `pages/` | one module per page |
| `data.py` | downloading the dataset, faculties, the theses table |
| `derived.py`, `prepare.py` | every derived table (DuckDB), cached in `data/derived/` |
| `vocab.py`, `geo.py` | word frequencies and country mentions, expected vs actual |
| `trends.py` | linear-trend and mean baselines with prediction intervals |
| `countries.py` | the word-form patterns per country |
| `charts.py`, `theme.py`, `ui.py` | shared chart, colours, controls and URL state |

## Data notes

- Only `usable` theses (text present, not garbled) from 2006 on are included; earlier years
  have very few theses. The current year is incomplete.
- Body length uses only theses whose body was cleanly cut out (`body_method == "intro-back"`),
  as the dataset card recommends for comparisons across years.
- Two kinds of word data are used. By default the pages use the dataset's exact yearly counts
  of every word form. Splitting by faculty, merging inflected forms, counting theses that mention
  a country and the style index need per-thesis counts, which the dataset only provides for word
  forms a thesis uses at least twice; those views say so, and their shares are lower.
- Predictions come with 95 % prediction intervals from the baseline's year-to-year scatter
  (never narrower than the sampling noise of the counts).
- Country mentions are matched on single word forms, so ambiguous forms are left out (e.g.
  "anglický", "americký"), North and South Korea are merged and Czechia is not shown.
- AI mentions are keyword matches, so theses about AI count too, and declarations include
  declared non-use. The style index's marker words were chosen on 2024 theses, before
  declarations became common; theses that address AI also differ in topic and faculty, so the
  comparison is a correlation.
