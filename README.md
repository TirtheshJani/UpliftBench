# UpliftBench

> Causal-inference benchmark on the **Criteo Uplift Prediction Dataset v2**.
> Five estimators, Qini + AUUC evaluation, four-way customer segmentation, DoWhy refutation, and a live Streamlit demo.

[![CI](https://github.com/TirtheshJani/UpliftBench/actions/workflows/ci.yml/badge.svg)](https://github.com/TirtheshJani/UpliftBench/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/built%20with-uv-261230.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Table of contents

1. [What this is](#what-this-is)
2. [Why](#why-causal-inference-now)
3. [Deliverables](#deliverables)
4. [Quickstart](#quickstart)
5. [Repo layout](#repo-layout)
6. [Architecture at a glance](#architecture-at-a-glance)
7. [The five estimators](#the-five-estimators)
8. [Evaluation](#evaluation)
9. [The four-way segmentation](#the-four-way-segmentation)
10. [DoWhy refutation](#dowhy-refutation)
11. [Hardware and memory budget](#hardware-and-memory-budget)
12. [Reproducibility](#reproducibility)
13. [Documentation](#documentation)
14. [Bundled skills](#bundled-skills)
15. [Repo conventions](#repo-conventions)
16. [Citation](#citation)
17. [License](#license)

## What this is

A portfolio-grade benchmark that takes the public **Criteo Uplift Prediction Dataset v2** (~13.9M rows, RCT-style randomized treatment, free from Criteo AI Lab) and runs five canonical uplift estimators end-to-end:

- **S-learner** (one model, treatment as feature)
- **T-learner** (two models, one per arm)
- **X-learner** (Kunzel et al., 2019; implemented from scratch with LightGBM)
- **DR-learner** (EconML, doubly robust, 3-fold cross-fit)
- **Double Machine Learning** (EconML LinearDML)

Each estimator implements the same `BaseUpliftEstimator` protocol so the eval harness, the Streamlit app, and the Kaggle notebook all consume them identically. The strongest estimator (by Qini) is wrapped in DoWhy's four-step pipeline and stress-tested with all four standard refuters.

## Why causal inference, now

Causal inference / uplift modeling is the fastest-growing required skill in 2026 senior ML/DS job descriptions: Snap, Booking, Wayfair, Uber, DoorDash, and Spotify's Experimentation Platform team all hire for it. The Criteo Uplift v2 dataset is the canonical public benchmark in this space (real RCT-style randomized ad-targeting decisions). Doing the work on it, end-to-end and reproducibly, signals the right thing.

## Deliverables

| # | Artifact | Path |
|---|---|---|
| 1 | This GitHub repo | https://github.com/TirtheshJani/UpliftBench |
| 2 | Streamlit Community Cloud demo (budget slider over pre-scored sample, no inference at runtime) | `streamlit_app/app.py` |
| 3 | Kaggle public notebook (self-contained, 2M-row subsample, fits the 9-hour CPU wall) | `notebooks/kaggle_end_to_end.ipynb` |
| 4 | Technical blog post | `blog/post.md` |

## Quickstart

```bash
# 1. Install
uv sync --extra dev                     # core + dev (tests, lint)
uv sync --extra dev --extra streamlit   # add the slim Streamlit-only deps

# 2. Get and prepare data (one shot, ~15 minutes)
make data       # download criteo-uplift-v2.1.csv.gz (~700 MB, with HF mirror fallback)
make prepare    # CSV.gz to parquet, dtype-optimized, batched

# 3. Train and evaluate
make train-all  # S, T, X, DR, DML (~2-3 hours on 8 CPU cores)
make dowhy      # 4-step pipeline + 4 refuters on a 1M stratified sample
make eval       # build artifacts/leaderboard.parquet
make score      # build artifacts/scored_sample.parquet (Git LFS)

# 4. Demo and tests
make app        # Streamlit local
make test       # pytest -q
make ci         # lint + type + test
```

If you only have 60 seconds and just want to read the math, open
[`docs/EVALUATION.md`](docs/EVALUATION.md) and
[`src/upliftbench/eval/qini.py`](src/upliftbench/eval/qini.py).

## Repo layout

```
UpliftBench/
├── README.md, LICENSE, CLAUDE.md, Makefile
├── pyproject.toml, uv.lock                # uv-managed env (Python 3.11+)
├── .pre-commit-config.yaml                # ruff format + check, em-dash forbidder
├── .github/workflows/ci.yml               # GitHub Actions, ubuntu-latest only
├── .streamlit/config.toml                 # theme + headless settings
├── .claude/skills/                        # five bundled skills (auto-loaded by Claude Code)
├── .gitattributes                         # Git LFS rules for three artifacts
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATA.md
│   ├── ESTIMATORS.md
│   ├── EVALUATION.md
│   ├── REFUTATION.md
│   └── CONTRIBUTING.md
├── src/upliftbench/
│   ├── config.py                          # paths, FEATURES, LIGHTGBM_PARAMS, thresholds
│   ├── features.py                        # Criteo schema (pyarrow), dtype map
│   ├── data/
│   │   ├── download.py                    # primary URL + HF mirror fallback
│   │   ├── prepare.py                     # streaming CSV.gz to parquet (pyarrow)
│   │   └── loader.py                      # chunked iterator + train_test_split_rct
│   ├── estimators/
│   │   ├── base.py                        # BaseUpliftEstimator Protocol
│   │   ├── s_learner.py, t_learner.py     # LightGBM
│   │   ├── x_learner.py                   # LightGBM from scratch (no causalml dep)
│   │   ├── dr_learner.py, dml.py          # EconML wrappers
│   │   └── __init__.py                    # ESTIMATOR_REGISTRY + get_estimator
│   ├── eval/
│   │   ├── qini.py, auuc.py, topk.py
│   │   └── harness.py                     # evaluate_estimator() entry point
│   ├── refute/dowhy_pipeline.py           # 4 steps + all 4 standard refuters
│   ├── segmentation.py                    # SHARED API (no heavy deps)
│   ├── persistence.py                     # save/load + sibling JSON metadata
│   └── plotting.py
├── scripts/                               # Typer CLIs
│   ├── download_data.py, prepare_data.py
│   ├── train.py, evaluate_all.py
│   ├── run_dowhy.py, score_sample.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_s_t_learners.ipynb
│   ├── 03_x_dr_dml.ipynb
│   ├── 04_dowhy_refutation.ipynb
│   ├── 05_comparison.ipynb
│   └── kaggle_end_to_end.ipynb            # self-contained, fits 9h Kaggle CPU
├── streamlit_app/app.py                   # NO heavy ML imports (slim Streamlit Cloud)
├── tests/                                 # pytest tree mirrors src/
├── artifacts/                             # LFS only: scored_sample, leaderboard, refutation
└── blog/post.md
```

## Architecture at a glance

The library is the **single source of truth**. Three entry points (training scripts, Kaggle notebook, Streamlit app) all import the same modules so the scoring math is identical across them.

```
                       ┌──────────────────────────────────────────┐
   data/raw/  ─CSV─►   │  scripts/prepare_data.py  (one shot)     │
                       │     pyarrow streaming                    │
                       └──────────────────┬───────────────────────┘
                                          │
                                          ▼
                       ┌──────────────────────────────────────────┐
   data/processed/  ◄──┤   data/processed/criteo.parquet          │
                       └──────────────────┬───────────────────────┘
                                          │
        ┌─────────────────────────────────┼──────────────────────────────────┐
        │                                 │                                  │
        ▼                                 ▼                                  ▼
 scripts/train.py                  scripts/run_dowhy.py            scripts/score_sample.py
  (per estimator)                   (best estimator)                (all estimators, 200k)
        │                                 │                                  │
        ▼                                 ▼                                  ▼
 artifacts/models/*.joblib       artifacts/dowhy_refutation.json   artifacts/scored_sample.parquet
        │                                                                    │
        ▼                                                                    │
 scripts/evaluate_all.py                                                     │
        │                                                                    │
        ▼                                                                    ▼
 artifacts/leaderboard.parquet  ─────────────────────────────►  streamlit_app/app.py
                                                                (NO heavy ML imports)
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for module responsibilities.

## The five estimators

All five implement the same protocol:

```python
class BaseUpliftEstimator(Protocol):
    name: str
    def fit(self, X, T, Y) -> None: ...
    def predict_cate(self, X) -> np.ndarray: ...
    def predict_baseline(self, X) -> np.ndarray: ...
```

They register themselves into `ESTIMATOR_REGISTRY`, which the train CLI dispatches against. See [`docs/ESTIMATORS.md`](docs/ESTIMATORS.md) for theory and implementation notes per estimator.

## Evaluation

Qini and AUUC are implemented from scratch in `src/upliftbench/eval/{qini,auuc}.py` and TDD'd against synthetic uplift data where the true per-row treatment effect is known analytically. A perfect ranker (sort by ground-truth) gives positive Qini > 0.05 on the synthetic fixture; a random ranker stays within ±0.03 of zero.

See [`docs/EVALUATION.md`](docs/EVALUATION.md) for the math.

## The four-way segmentation

The shared `segmentation.score_and_segment` function labels each row as one of:

| Segment | predicted CATE | predicted baseline | meaning |
|---|---|---|---|
| persuadable | > 0 | low | treatment converts them |
| sure_thing | ≤ 0 | high | converts anyway |
| lost_cause | ≤ 0 | low | treatment will not help |
| do_not_disturb | > 0 | high | classical uplift literature flags as negative ROI |

The Streamlit slider re-allocates the treatment budget across these four buckets live.

## DoWhy refutation

The best estimator (by Qini) is wrapped in DoWhy's four-step pipeline:

1. **Model** the causal graph (treatment, outcome, common causes).
2. **Identify** a backdoor-adjusted estimand.
3. **Estimate** the effect (`backdoor.linear_regression` by default).
4. **Refute** with all four standard refuters: `placebo_treatment`, `random_common_cause`, `data_subset`, `add_unobserved_common_cause`.

Refutation runs on a **1M-row stratified-by-treatment subsample** (the full 13.9M is intractable on a 16 GB laptop). The sample-size choice is documented in code (`run_dowhy(sample_n=1_000_000)`) and in [`docs/REFUTATION.md`](docs/REFUTATION.md).

## Hardware and memory budget

| Item | Value |
|---|---|
| Target hardware | RTX 4080 laptop, 16 GB system RAM, Windows or WSL2 |
| GPU | Not used. CPU-only, by design. |
| Python | 3.11+ |
| Env manager | uv + pyproject.toml + src/ layout |
| Lint/type | ruff + mypy |
| Tests | pytest, TDD for math-bearing modules, smoke tests for estimators |
| CI | GitHub Actions, ubuntu-latest only |

Memory tactics:

- float32 features, uint8 treatment/visit/conversion.
- pyarrow batch iteration with `batch_size=500_000`.
- `lgb.Dataset(..., free_raw_data=True, max_bin=63)`.
- Documented fallback to a 5M-row stratified-by-treatment sample if peak RSS ever exceeds 12 GB.

## Reproducibility

- All splits derive from a single seed (`SEED=42` in `src/upliftbench/config.py`).
- The held-out test-row hash is recorded in each model's metadata JSON; every estimator on the leaderboard was scored on the same rows.
- Library versions are persisted in metadata at training time.
- CI runs `ruff check`, `ruff format --check`, `mypy src`, `pytest -q`, and an em-dash grep check on every push.

```bash
uv run pre-commit run --all-files
uv run mypy src
uv run pytest --cov=upliftbench --cov-report=term-missing
```

## Documentation

| Doc | What it covers |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Module map, dependency graph, shared-API contracts |
| [`docs/DATA.md`](docs/DATA.md) | Criteo Uplift v2 schema, download path, memory tactics |
| [`docs/ESTIMATORS.md`](docs/ESTIMATORS.md) | Theory + implementation notes for each of the five learners |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Qini and AUUC math, normalization, anti-patterns |
| [`docs/REFUTATION.md`](docs/REFUTATION.md) | DoWhy 4-step pipeline, each refuter's intent |
| [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md) | Dev setup, TDD policy, pre-commit, CI |
| [`blog/post.md`](blog/post.md) | ~900-word technical writeup (the deliverable) |

## Bundled skills

`.claude/skills/` ships five Claude Code skills so any session in this repo auto-loads them:

- `writing-plans` (plan format)
- `executing-plans` (per-task workflow)
- `test-driven-development` (red-green-refactor + anti-patterns)
- `dispatching-parallel-agents` (when to fan out)
- `karpathy-guidelines` (think before coding, simplicity, surgical edits)

## Repo conventions

- **No em dashes** anywhere in committed text. A pre-commit hook enforces this. Use commas, parentheses, or semicolons.
- **No `git add -A`**. Stage specific paths so secrets and large files cannot slip in.
- **Git LFS** only for `artifacts/{scored_sample.parquet, leaderboard.parquet, dowhy_refutation.json}`. Trained models are gitignored; reproduce with `make train-all`.
- **Branch policy**: develop on `claude/causal-inference-uplift-x0DZw`. Do not push to `main` without explicit user request.

## Citation

If you use this benchmark, please cite the Criteo dataset:

```bibtex
@inproceedings{Diemert2018,
  author    = {Diemert Eustache, Betlei Artem and Renaudin, Christophe and Massih-Reza Amini},
  title     = {A Large Scale Benchmark for Uplift Modeling},
  booktitle = {Proceedings of the AdKDD and TargetAd Workshop, KDD},
  year      = {2018}
}
```

## License

MIT (see [`LICENSE`](LICENSE)).

The Criteo Uplift Prediction Dataset v2 is provided by Criteo AI Lab under their dataset license; this repo does not redistribute the data, only fetches it on demand.
