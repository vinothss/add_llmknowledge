# Adding Knowledge to LLMs: self-paced workshop

Hands-on course: curate data, evaluate, customise (CPT / SFT / DPO) and optimise LLMs.
Labs run on **Google Colab** (free T4 core track, Colab Pro bonus track).

## Repository layout

| Path | What |
|---|---|
| `docs/` | Portal content (MkDocs Material) |
| `notebooks/` | Colab labs (generated, don't hand-edit) |
| `scripts/make_m0*_notebook.py` | Source of each module notebook; edit here, then regenerate |
| `labkit/` | Shared helpers: runtime detection, LLM wrapper, auto-checks, progress |
| `labkit/solutions/` | Reference solutions (`USE_SOLUTIONS = True` in a notebook) |
| `data/raw/` | Fictional Lumora Energy corpus + answer key used by checks |
| `data/train/` | Reference training data for Module 3: CPT corpus + paraphrases, SFT QA, DPO style prompts (`scripts/build_train_data.py`) |
| `data/eval/` | Held-out Lumora QA benchmark + human-labelled judge calibration set (`scripts/build_eval_sets.py`) |
| `scripts/build_raw_corpus.py` | Regenerates the corpus |
| `requirements/` | Per-module pip extras installed by the notebook setup cell |

## Publish it

1. Create a public GitHub repo and push this folder to it.
2. `python scripts/set_repo.py <your-user>/<repo>` to rewrite every Colab badge and clone URL, then
   regenerate the notebooks (`python scripts/make_m0N_notebook.py` for each module) and commit.
3. In GitHub: **Settings -> Pages -> Source: `gh-pages` branch**. The workflow in
   `.github/workflows/docs.yml` builds the portal on each push to `main`.

## Develop locally

```bash
pip install mkdocs-material nbformat nbconvert
mkdocs serve                                  # portal at http://127.0.0.1:8000
for f in scripts/make_m0*_notebook.py; do python $f; done   # regenerate notebooks
# smoke-test the notebook end to end with reference solutions:
cd notebooks && LAB_SMOKE=1 LAB_USE_SOLUTIONS=1 jupyter nbconvert --execute --to notebook 01_data_curation.ipynb --output /tmp/out.ipynb
```

Lumora Energy and all its people, products and data are fictional.
