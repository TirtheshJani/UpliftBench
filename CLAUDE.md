# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project intent

UpliftBench is a portfolio project benchmarking causal-inference and uplift-modeling techniques on the **Criteo Uplift Prediction Dataset v2** (~13.9M rows, RCT-style randomized treatment assignment, from Criteo AI Lab). Development happens on feature branches (e.g. `claude/<topic>`) merged to `main` via PR.

Scope:

- **DoWhy four-step pipeline**: model, identify, estimate, refute (all four refuters on a 1M-row stratified sample).
- **CausalML meta-learners**: S-learner, T-learner, X-learner with LightGBM bases.
- **EconML**: Double Machine Learning (LinearDML) and DR-learner.
- Evaluation: Qini curves, AUUC, and four-way segmentation (persuadables, sure-things, lost-causes, do-not-disturb).
- CPU-bound LightGBM with chunked dataset streaming. CPU only, not GPU.

## Hardware and stack

| Item | Value |
|---|---|
| Target hardware | RTX 4080 laptop, 16 GB system RAM, Windows or WSL2 |
| Python | 3.11+ |
| Env manager | uv with `pyproject.toml`, src/ layout |
| Lint | ruff |
| Types | mypy (`uv run mypy src`) |
| Tests | pytest, TDD for `eval/`, `data/loader`, `segmentation`, `persistence`, `refute/`. Smoke tests for estimators. |
| Pre-commit | ruff format, ruff check, large-file guard, EOF/whitespace fixers |
| CI | GitHub Actions, ubuntu-latest only |
| Demo | Streamlit Community Cloud (slim deps via `[project.optional-dependencies] streamlit`) |

## Common commands

```bash
uv sync --extra dev                                       # install
uv run pre-commit run --all-files                         # lint + hygiene
uv run mypy src                                           # type-check
uv run pytest -q                                          # tests
uv run pytest --cov=upliftbench --cov-report=term-missing # tests + coverage
make data                                                 # download Criteo CSV.gz
make prepare                                              # CSV -> parquet
make train-all                                            # train 5 estimators
make dowhy                                                # refutation on 1M sample
make score                                                # build LFS-tracked scored sample
make eval                                                 # build leaderboard
make app                                                  # streamlit local
```

## Architecture

The repo follows a single-source-of-truth library plus thin entry points. Everything lives under `src/upliftbench/`.

- `data/`: `prepare.py` (CSV.gz to parquet, dtype-optimized), `loader.py` (chunked iterator + `train_test_split_rct`). Download lives in `scripts/download_data.py` (primary URL + HF mirror fallback), not in the library.
- `estimators/`: one module per estimator (`s_learner.py`, `t_learner.py`, `x_learner.py`, `dr_learner.py`, `dml.py`). All implement `BaseUpliftEstimator` (`fit`, `predict_cate`, `predict_baseline`). Registered in `ESTIMATOR_REGISTRY` in `estimators/__init__.py`.
- `eval/`: `qini.py`, `auuc.py`, `topk.py`, `harness.py` (`evaluate_estimator(t, y, cate) -> dict`).
- `segmentation.py`: **shared API** used by the training script, the Kaggle notebook, and the Streamlit app. Functions: `score_and_segment(...)`, `budget_allocation(...)`.
- `refute/dowhy_pipeline.py`: `run_dowhy(df, feature_cols, treatment_col="treatment", outcome_col="visit", sample_n=1_000_000, seed=42, estimator_method="backdoor.linear_regression")` does model, identify, estimate, then four refutations. The trained learners from Phase 2 are evaluated separately; DoWhy uses its own backdoor estimator (`estimator_method`) for the refutation pipeline, not the winning meta-learner.
- `persistence.py`: `save_model`, `load_model` with sibling metadata JSON.
- `plotting.py`: Qini curves, AUUC bars, segment bars.

Entry points:

- `scripts/`: Typer CLIs for download, prepare, train, evaluate_all, run_dowhy, score_sample.
- `streamlit_app/app.py`: import allow-list is `pandas`, `matplotlib`, `streamlit`, `upliftbench.config` (pure constants), and `upliftbench.segmentation`. **No lightgbm/dowhy/causalml/econml imports**, so Streamlit Community Cloud stays under its 1 GB RAM cap.
- `notebooks/`: EDA, per-phase notebooks, comparison, Kaggle end-to-end.

## Data handling

- The Criteo v2 CSV.gz is ~700 MB compressed, ~3 GB uncompressed. Convert to parquet once via `scripts/prepare_data.py`, then iterate parquet batches.
- Treatment assignment is randomized. Do NOT stratify by treatment when splitting (it destroys the RCT structure DoWhy refutation depends on).
- Raw and processed data are gitignored. The repo is cloneable without the dataset; `scripts/download_data.py` fetches it with a primary URL plus HF Datasets mirror fallback.

## Memory budget on 16 GB laptop

- Aggressive dtype downcast (float32 features, uint8 treatment/visit/conversion).
- pyarrow batch iteration with `batch_size=500_000`.
- LightGBM: `lgb.Dataset(..., free_raw_data=True, max_bin=63)`.
- If peak RSS exceeds 12 GB on full data, fall back to a 5M-row stratified-by-treatment sample. Document the fallback in the blog.

## Bundled skills

`.claude/skills/` contains five skills any Claude Code session in this repo will auto-load:

| Skill | Use when |
|---|---|
| `writing-plans` | Authoring or revising any plan in this repo |
| `executing-plans` | Following an approved plan task-by-task |
| `test-driven-development` | Any code touching `eval/`, `segmentation`, `data/loader`, `persistence`, or `refute/` |
| `dispatching-parallel-agents` | Fanning out independent estimator builds or final deliverables (Streamlit + Kaggle + blog) |
| `karpathy-guidelines` | Every coding task (think before coding, simplicity first, surgical changes, goal-driven verification) |

## Repo conventions

- **No em dashes** anywhere in code, docs, commits, or chat output. CI grep check enforces.
- **No `git add -A`**: stage specific paths so secrets and large files cannot slip in.
- **LFS scope**: `.gitattributes` declares LFS rules for `artifacts/scored_sample.parquet`, `artifacts/leaderboard.parquet`, `artifacts/dowhy_refutation.json`, but these files are NOT committed to the repo. Produce them locally with `make score` / `make eval` / `make dowhy`; commit via LFS only if you choose to publish. Trained models are gitignored; reproduce with `make train-all`.
- **Branch policy**: develop on a feature branch (e.g. `claude/<topic>`) and open a PR to `main`. Never push to `main` without explicit user request.
