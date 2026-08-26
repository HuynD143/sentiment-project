# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Streamlit app that crawls comments from a YouTube or Reddit URL, classifies each into 3 sentiment classes with a locally loaded Keras model, and renders matplotlib charts + wordclouds. UI text is Vietnamese; the model is trained on English text.

## Commands

```bash
# Setup (Python 3.10/3.11 — TensorFlow does not support newer versions)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Main app (both are equivalent)
python main.py
streamlit run ui.py

# Single-text eval page
python main.py -n eval
streamlit run eval.py

# Docker (listens on 8502, not Streamlit's default 8501)
docker build -t sentiment-app .
docker run --rm -p 8502:8502 --env-file ./crawl_data/.env sentiment-app
```

There is no test suite and no linter configured. `crawl_data/test.py` is a stale duplicate of `Crawler`, not a test.

`run_ui.bat` and `setup_env.bat` are stale: they reference a `backend/UI.py` layout and a hardcoded Python 3.13 path that no longer match the repo. Prefer the commands above.

## Architecture

The pipeline is entirely inside [ui.py](ui.py) — there is no service layer:

1. **Route by URL substring** — `youtube.com`/`youtu.be` → `Crawler`, `reddit.com` → `CrawlReddit`. Anything else is rejected.
2. **Crawl** → both crawlers are normalized to two parallel lists, `content` (comment text) and `authors`. Note the differing return shapes: `Crawler.get_youtube_comments()` returns dicts (`author`, `text`, `likes`, `published_at`, `updated_at`); `CrawlReddit.get_comments()` returns `(author, body)` tuples.
3. **Inference** — `predict_local()` tokenizes with the saved tokenizer, pads to `MAX_LEN`, `argmax` over 3 softmax outputs. Class order is fixed by training: `0=negative, 1=neutral, 2=positive`.
4. **Render** — `draw_basic_stats`, `draw_distribution_and_share`, `draw_wordclouds`, selected by the radio button. Each independently re-buckets `content` by label into a `comments_by` dict keyed by the Vietnamese label strings, so those strings (`"Tích cực"`, `"Trung lập"`, `"Tiêu cực"`) are load-bearing and must stay in sync with `idx2vi` in `predict_local`.

Crawler imports are wrapped in a try/except that sets `_CRAWLER_OK`; if `crawl_data` fails to import, the app degrades to a warning instead of crashing.

[eval.py](eval.py) is a standalone page that reloads model/tokenizer with its own hardcoded relative paths (`notebook/models/...`), so it only works when run from the repo root. Its `process()` reloads the model on every call — the `@st.cache_resource` helper next to it is defined but unused for the actual prediction.

## Model artifacts

- `notebook/models/btlpython3.h5` + `notebook/models/tokenizer.json` are what the app loads (paths are `MODEL_PATH`/`TOKENIZER_PATH` in [ui.py](ui.py), duplicated as literals in [eval.py](eval.py)).
- Architecture (from [notebook/scripts/train_model.ipynb](notebook/scripts/train_model.ipynb)): `Embedding(29766, 100)` → `Bidirectional(LSTM(128))` → `Dense(32, relu)` → `Dense(3, softmax)`, AdamW lr=1e-4, sparse categorical crossentropy. Plain `Sequential` with no custom layers, so `load_model()` needs no `custom_objects`. The `PositionEncode`/`EncodeBlock` transformer classes in the notebook are defined but commented out of the final model.
- Training data is pulled from the HuggingFace dataset `Sp1786/multiclass-sentiment-analysis-dataset`, **not** from `data/train.csv`. The CSVs in `data/` are a different (Kaggle tweet) dataset with a `sentiment` string column and are not used by the app.
- **Train/serve length skew:** the notebook pads to `maxlen=100`; `ui.py` and `eval.py` pad to `MAX_LEN = 30`. BiLSTM tolerates it so nothing errors, but predictions are made at a sequence length the model was not trained on. Change `MAX_LEN` in both files together if aligning them.
- The notebook saves `btlpython2.h5`; `btlpython3.h5` is not produced by any committed code.

## Credentials

- YouTube: `Crawler.__init__` calls `load_dotenv()` and reads **`API_KEY`** — that exact name. The root `.env.txt` documents `YOUTUBE_API_KEY`/`REDDIT_*` instead, which no code reads; `crawl_data/.env` has the correct `API_KEY`.
- Reddit: `client_id`/`client_secret` are hardcoded in [crawl_data/crawl_reddit.py](crawl_data/crawl_reddit.py); there is no env-var path.
- Both `.env.txt` and `crawl_data/.env` are committed to git with live-looking keys — treat them as compromised rather than as a template to copy.

## Repo state

- `.gitignore` lists `*.h5`, but the three `.h5` files were already tracked before it was added and remain in the index (~38 MB each). `git rm --cached` is needed if the intent is to actually drop them.
- `backend_api/main_api.py` and `backend_api/requirements.txt` are empty placeholders.
