# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project intent

UpliftBench is a portfolio project benchmarking causal inference / uplift modeling techniques on the **Criteo Uplift Prediction Dataset v2** (~13.9M rows, RCT-style randomized treatment assignment, from Criteo AI Lab). The repo currently contains only README.md and LICENSE — all code is yet to be written on branch `claude/causal-inference-uplift-x0DZw`.

The intended scope (from the original task brief) is:

- **DoWhy four-step pipeline**: model → identify → estimate → refute.
- **CausalML meta-learners**: S-learner, T-learner, X-learner.
- **EconML Double Machine Learning** and a **DR-learner**.
- Base learner is **CPU-bound LightGBM** with chunked dataset streaming (the dataset does not fit comfortably in memory on a laptop, and the project is explicitly designed to run on CPU in hours-per-estimator, not GPU).
- Evaluation: **Qini curves**, **AUUC**, and segmentation of the population into **persuadables / sure-things / lost-causes / do-not-disturb**.

## Deliverables (drive architecture)

Four artifacts are expected — keep the code organized so each one is straightforward to produce:

1. **Streamlit Community Cloud demo** with a treatment-budget slider that updates the persuadable segment live. The Streamlit app should load a pre-trained model + a precomputed scored sample, not retrain on launch (Community Cloud has tight RAM/CPU limits).
2. **Kaggle public notebook** (SEO visibility) — self-contained, runnable on Kaggle's environment.
3. **GitHub repo** — this repo; reproducible training scripts + saved model artifacts.
4. **Short technical blog post**.

Because the same logic feeds the notebook, the Streamlit app, and the training scripts, factor shared code (data loading, feature prep, scoring, Qini/AUUC, segmentation) into a small library module and have the three entry points (training script, notebook, Streamlit app) import from it rather than duplicating.

## Data handling constraints

- The Criteo v2 CSV is large; **do not load it whole in memory**. Use chunked reading (`pandas.read_csv(chunksize=...)`) or `pyarrow`/parquet conversion as a one-time preprocessing step, then read parquet for training.
- Treatment assignment is randomized — preserve that property when sampling; do not stratify by treatment in ways that destroy the RCT structure used for refutation.
- Keep raw data out of git (add to `.gitignore` when adding it). The repo should be cloneable without the dataset; provide a download/preprocess script instead.

## Working in this repo

- All work goes on branch `claude/causal-inference-uplift-x0DZw` (already checked out).
- No build, lint, or test commands exist yet — add them to this file as soon as a `pyproject.toml` / `requirements.txt` / test suite is introduced.
- When picking libraries, the brief explicitly calls for **DoWhy**, **CausalML**, **EconML**, and **LightGBM**. Prefer these over alternatives unless there's a concrete reason to deviate.
