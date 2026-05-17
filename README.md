# UpliftBench

Causal-inference benchmark on the **Criteo Uplift Prediction Dataset v2** (~13.9M rows, RCT-style randomized treatment, from Criteo AI Lab).

Five estimators: S-learner, T-learner, X-learner, DR-learner, Double Machine Learning. Evaluation with Qini and AUUC. Four-way customer segmentation (persuadables, sure-things, lost-causes, do-not-disturb). DoWhy four-step pipeline with all four standard refuters on a 1M-row stratified sample.

## Deliverables

1. This GitHub repo.
2. Streamlit Community Cloud demo: budget slider over a pre-scored 200k-row sample. The app does no model inference at runtime; it imports only `pandas`, `pyarrow`, `numpy`, `matplotlib`, `streamlit`, and `upliftbench.segmentation` (the heavy ML libs are excluded from the `streamlit` extra).
3. Kaggle notebook (`notebooks/kaggle_end_to_end.ipynb`): self-contained, 2M-row subsample, fits the 9-hour CPU wall.
4. Short technical blog post: `blog/post.md`.

## Quickstart

```bash
uv sync --extra dev                        # install
uv sync --extra dev --extra streamlit      # add the slim Streamlit-only deps
make data                                  # download Criteo CSV.gz
make prepare                               # CSV.gz to parquet, dtype-optimized
make train-all                             # train all 5 estimators
make dowhy                                 # DoWhy 4-step pipeline + 4 refuters
make eval                                  # build artifacts/leaderboard.parquet
make score                                 # build LFS-tracked scored sample
make app                                   # streamlit local
```

## Architecture

Everything under `src/upliftbench/`. Three entry points (training scripts, Kaggle notebook, Streamlit app) share one library so logic is not duplicated.

| Module | Role |
|---|---|
| `data/{download,prepare,loader}.py` | Criteo download with HF mirror fallback; streaming CSV.gz to parquet; chunked batch iterator; RCT-preserving train/test split |
| `estimators/{base,s_learner,t_learner,x_learner,dr_learner,dml}.py` | One class per estimator implementing `BaseUpliftEstimator`. `ESTIMATOR_REGISTRY` in `__init__.py` |
| `eval/{qini,auuc,topk,harness}.py` | Qini curve and coefficient, AUUC, top-K uplift, single `evaluate_estimator` entry point |
| `segmentation.py` | Shared `score_and_segment` and `budget_allocation`. Imported identically by training, notebook, Streamlit. No heavy deps. |
| `refute/dowhy_pipeline.py` | `run_dowhy(...)` does model + identify + estimate + four refuters |
| `persistence.py` | `save_model` / `load_model` with sibling JSON metadata (git sha, runtime, Qini, AUUC, lib versions) |
| `plotting.py` | Qini, AUUC, segment-bar helpers |

Entry points:

- `scripts/{download_data,prepare_data,train,run_dowhy,score_sample,evaluate_all}.py` (Typer CLIs)
- `streamlit_app/app.py`
- `notebooks/01_eda.ipynb` ... `05_comparison.ipynb` and `kaggle_end_to_end.ipynb`

## Hardware and stack

| Item | Value |
|---|---|
| Target hardware | RTX 4080 laptop, 16 GB system RAM, Windows or WSL2 |
| Python | 3.11+ |
| Env | uv + pyproject.toml + src/ layout |
| Lint/type | ruff, mypy |
| Tests | pytest, TDD for `eval/`, `segmentation`, `data/loader`, `persistence`, `refute/`; smoke tests for estimators |
| Pre-commit | ruff format/check, large-file guard, em-dash forbidder |
| CI | GitHub Actions, ubuntu-latest only |
| Demo | Streamlit Community Cloud |

## Bundled skills

`.claude/skills/` contains five skills that any Claude Code session in this repo auto-loads:

- `writing-plans`
- `executing-plans`
- `test-driven-development`
- `dispatching-parallel-agents`
- `karpathy-guidelines`

## Repo conventions

- No em dashes anywhere. A pre-commit hook enforces this.
- No `git add -A`. Stage specific paths so secrets and large files cannot slip in.
- LFS only for `artifacts/scored_sample.parquet`, `artifacts/leaderboard.parquet`, `artifacts/dowhy_refutation.json`. Trained models are gitignored; reproduce with `make train-all`.
- Develop on `claude/causal-inference-uplift-x0DZw`. Do not push to `main` without explicit user request.

## License

MIT. The Criteo Uplift Prediction Dataset v2 is provided by Criteo AI Lab under their dataset license.
