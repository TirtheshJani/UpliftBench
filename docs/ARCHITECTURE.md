# Architecture

UpliftBench is a single library (`src/upliftbench/`) plus three thin entry points (training scripts, Streamlit app, Kaggle notebook). The library is the single source of truth; every entry point imports from it.

## Module map

| Module | Public surface | Notes |
|---|---|---|
| `config.py` | `FEATURES`, paths, `LIGHTGBM_PARAMS`, `SEED`, segment thresholds | One place to change feature names, paths, hyperparameters |
| `features.py` | `CRITEO_SCHEMA` (pyarrow), `CSV_COLUMN_DTYPES` | Used by `data/prepare.py` to enforce dtype downcast on read |
| `data/prepare.py` | `csv_to_parquet(csv_path, parquet_path, batch_rows)` | Streaming, peak RSS bounded by `batch_rows` |
| `data/loader.py` | `load_parquet_iter`, `train_test_split_rct(n, test_frac, seed)` | RCT split is **not** stratified on treatment, by design |
| `estimators/base.py` | `BaseUpliftEstimator` Protocol | `fit`, `predict_cate`, `predict_baseline` |
| `estimators/{s,t,x,dr,dml}_learner.py` | One class each | Hyperparameters pinned in `config.py` |
| `estimators/__init__.py` | `ESTIMATOR_REGISTRY`, `get_estimator(name)` | The train CLI dispatches via this |
| `eval/qini.py` | `qini_curve`, `qini_coefficient` | TDD'd against synthetic uplift with known signal |
| `eval/auuc.py` | `uplift_curve`, `auuc` | Same normalization style as Qini |
| `eval/topk.py` | `top_k_uplift` | Empirical ATE in the top-k by CATE |
| `eval/harness.py` | `evaluate_estimator(t, y, cate)` | Single dict of all metrics + curves |
| `segmentation.py` | `score_and_segment`, `budget_allocation`, segment constants | **Imported by training, notebook, and Streamlit identically.** No heavy ML deps. |
| `refute/dowhy_pipeline.py` | `run_dowhy`, `REFUTER_NAMES` | All four standard refuters on a stratified sample |
| `persistence.py` | `save_model`, `load_model`, `ModelPaths` | Joblib + sibling JSON metadata |
| `plotting.py` | `qini_plot`, `auuc_bar`, `segment_bar` | matplotlib helpers |

## Entry points

```
scripts/download_data.py   ─►  data/raw/criteo-uplift-v2.1.csv.gz
scripts/prepare_data.py    ─►  data/processed/criteo.parquet
scripts/train.py           ─►  artifacts/models/{est}_{date}_{sha}.{joblib,json}
scripts/run_dowhy.py       ─►  artifacts/dowhy_refutation.json
scripts/evaluate_all.py    ─►  artifacts/leaderboard.parquet, best_estimator.txt
scripts/score_sample.py    ─►  artifacts/scored_sample.parquet (Git LFS)
streamlit_app/app.py       ─►  reads scored_sample + leaderboard + refutation (no inference)
```

The Streamlit app's import list is intentionally minimal:

```python
import pandas, pyarrow, numpy, matplotlib, streamlit
from upliftbench.config import ...
from upliftbench.segmentation import budget_allocation, ALL_SEGMENTS, SEGMENT_PERSUADABLE
```

No `lightgbm`, no `dowhy`, no `causalml`, no `econml`. The slim `streamlit` extra in `pyproject.toml` excludes those so the Streamlit Community Cloud install stays under its 1 GB RAM cap.

## Shared contracts

Two contracts let independent pieces compose:

1. **`BaseUpliftEstimator` Protocol.** Every estimator exposes `fit(X, T, Y) -> None`, `predict_cate(X) -> ndarray`, `predict_baseline(X) -> ndarray`. The train CLI, `evaluate_all`, and `score_sample` all rely on exactly this surface.
2. **`segmentation.score_and_segment` and `budget_allocation`.** These work on a `DataFrame` with columns `pred_cate`, `pred_baseline`, `segment`, and they are the **only** path the Streamlit app uses to transform predictions into the four-way customer mix. Any logic change here lands everywhere at once.

## Determinism

A single seed (`config.SEED = 42`) drives:

- the random train/test split in `train_test_split_rct`,
- LightGBM's `bagging` and `feature_fraction` sampling,
- DoWhy's stratified subsample for refutation,
- the 200k-row sample drawn by `score_sample`.

Every training run hashes the held-out test row indices and writes the digest into the model's sibling `.json` so `evaluate_all` can confirm every estimator was scored on the same rows.
