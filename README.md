# UpliftBench

> A causal-inference benchmark on the **Criteo Uplift Prediction Dataset v2**: five estimators, Qini and AUUC, four-way customer segmentation, DoWhy refutation, and a live Streamlit demo.

[![CI](https://github.com/TirtheshJani/UpliftBench/actions/workflows/ci.yml/badge.svg)](https://github.com/TirtheshJani/UpliftBench/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/built%20with-uv-261230.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Table of contents

1. [What this project is](#what-this-project-is)
2. [What I learned](#what-i-learned)
3. [Quickstart](#quickstart)
4. [Repo layout](#repo-layout)
5. [Architecture at a glance](#architecture-at-a-glance)
6. [The five estimators](#the-five-estimators)
7. [Evaluation](#evaluation)
8. [The four-way segmentation](#the-four-way-segmentation)
9. [DoWhy refutation](#dowhy-refutation)
10. [Hardware and memory budget](#hardware-and-memory-budget)
11. [Reproducibility](#reproducibility)
12. [Documentation](#documentation)
13. [Bundled skills](#bundled-skills)
14. [Repo conventions](#repo-conventions)
15. [Citation](#citation)
16. [License](#license)

## What this project is

UpliftBench takes the public **Criteo Uplift Prediction Dataset v2** (~13.9M rows from an RCT-style randomized ad-targeting experiment) and runs five canonical uplift estimators end-to-end on a single 16 GB laptop:

- **S-learner** (one model, treatment as feature)
- **T-learner** (two models, one per arm)
- **X-learner** (Kunzel et al., 2019; implemented from scratch with LightGBM)
- **DR-learner** (EconML, doubly robust, 3-fold cross-fit)
- **Double Machine Learning** (EconML LinearDML)

Each estimator implements the same `BaseUpliftEstimator` protocol so the evaluation harness, the Streamlit app, and the Kaggle notebook all consume them identically. The best estimator (by Qini) is then wrapped in DoWhy's four-step pipeline (model, identify, estimate, refute) and stress-tested with all four standard refuters on a stratified 1M-row sample.

Three artifacts come out: an interactive Streamlit demo where a treatment-budget slider re-allocates targeting across persuadable, sure-thing, lost-cause, and do-not-disturb segments live, a self-contained Kaggle notebook that fits the 9-hour CPU wall on a 2M-row subsample, and a short technical write-up.

## What I learned

### Causal inference is structurally different from prediction

A response model predicts `P(Y=1 | X)`. An uplift model predicts `P(Y=1 | X, T=1) - P(Y=1 | X, T=0)`. The catch: you never observe both for the same row. There is no ground-truth column to validate against per-row, only population-level metrics (Qini, AUUC) that integrate over a ranking. This reframes how you think about validation: instead of "did my predictions match the labels?", the question becomes "does my ranking produce a steeper-than-random cumulative-lift curve when we look at the held-out treated and control arms?".

### Qini math is short but the normalization choice matters

`Q(k) = Y_t(k) - Y_c(k) * (N_t(k) / N_c(k))` is two cumulative sums and a rescaling. The whole curve fits in about 15 lines of numpy. What is **not** short is picking the right denominator for the coefficient. My first implementation divided by `area_optimal - area_random`, where I approximated the optimal curve by sorting on `y*t - y*(1-t)`. That gave a Qini coefficient of -0.25 for a perfect ranker on synthetic data with known positive uplift. The "optimal" curve was actually a near-degenerate shape because of how the placeholder score behaved on early prefixes. Switching to the simpler `2 * (area_model - area_random) / |Q_total|` normalization (random near zero, perfect near one) made the math correct and the tests green. Lesson: TDD against synthetic data with known signal will save you from publishing a benchmark with a sign-flip bug.

### The X-learner's propensity-weighting trick is the interesting part

The X-learner is not "two models" or "four models". The structural insight is: build pseudo-outcomes `D_1 = Y - mu_0(X)` on treated rows and `D_0 = mu_1(X) - Y` on control rows (using the **opposite-arm** response model for each), regress them separately, and then combine the two CATE estimates with the propensity as the weight. The propensity weight is what makes it robust to imbalanced arms: when there are few treated rows, `D_1` is noisy, so `1-g` is small, so `tau_1` gets less weight in the final combination. Under an RCT, `g` is approximately constant; under observational data you would fit a propensity model. That single design choice is the difference between T-learner and X-learner.

### Doubly robust is a strong promise

The DR-learner is doubly robust: consistency of the CATE estimate requires only ONE of (propensity model, outcome regression) to be correctly specified, not both. EconML's implementation does this with 3-fold cross-fitting so the nuisance estimates are out-of-fold when they feed the final stage, avoiding the "training residuals are biased toward zero" trap. Cross-fitting is a small implementation cost (you fit nuisances K times) but the bias guarantee is large. This is also why Double Machine Learning works: residualize both sides, then regress residual on residual.

### DoWhy refutation is the part that separates a benchmark from a press release

The four refuters answer different questions:
- `placebo_treatment`: if you scramble the treatment column, does the estimate go to zero? (sanity)
- `random_common_cause`: if you add a meaningless covariate, does the estimate stay put? (robustness to spurious adjustment)
- `data_subset`: is the estimate stable to row subsampling? (no concentration in a few rows)
- `add_unobserved_common_cause`: how strong would a hidden confounder need to be to drive the estimate to zero? (sensitivity)

An ATE point estimate without these refutations is a number, not a finding. The 1M-row sample size for refutation is a pragmatic compromise; the math would be more precise on the full data, but the laptop wall-time is the binding constraint, not the statistical precision.

### Memory engineering is half the work on a single laptop

13.9M rows fits in 16 GB only if you treat dtype as a first-class design decision. Float32 features, uint8 treatment / visit / conversion, parquet with zstd, pyarrow batch iteration at 500k rows, and `lgb.Dataset(free_raw_data=True, max_bin=63)` keep peak RSS under 6 GB during training. Skip any one of those and you can blow past 12 GB and start swapping. The first time I tried with default pandas dtypes, the to_pandas() call alone took 4 GB just for the feature matrix.

### The segmentation step is where the model meets the budget decision

A CATE estimate alone is not actionable. To decide who to treat, you need both the predicted uplift AND the predicted baseline outcome: someone with high baseline propensity is going to convert anyway, so spending budget on them is waste even if their CATE looks high. The four-way segmentation (persuadables, sure-things, lost-causes, do-not-disturb) is the canonical way to translate `(pred_cate, pred_baseline)` into a targeting policy. The Streamlit slider makes this tangible: at small budgets you're targeting almost pure persuadables, but as the budget grows the marginal added user is increasingly a sure-thing or a do-not-disturb. That diminishing-returns curve is the story.

### The "no inference at runtime" Streamlit pattern

Streamlit Community Cloud caps each app at ~1 GB RAM. LightGBM + DoWhy + CausalML + EconML alone come to several hundred MB of installed code, and loading even one trained model adds more. The right pattern is to precompute a 200k-row scored sample at training time, persist it as a parquet via Git LFS, and have the Streamlit app do only filtering and segmentation math at runtime. The app's import list is intentionally `pandas, pyarrow, numpy, matplotlib, streamlit` plus the shared `segmentation` module. No heavy ML imports anywhere. Cold start drops from "model loading times out" to under three seconds.

### Library version pinning is not optional in causal inference

DoWhy 0.12 still uses `networkx.algorithms.d_separated`, which networkx removed in 3.3. CausalML's X-learner internally uses pygam, which calls `.A` on a scipy sparse matrix, which scipy removed in a recent release. Both broke before I had any code on the page. The fix is networkx pinned `<3.3` in pyproject and a from-scratch X-learner implementation that skips the pygam path. Both decisions are documented in the code so a future reader knows which pins are accidental and which are load-bearing.

### TDD is unreasonably effective on math-bearing code

Seven of the ten test files are TDD against synthetic data with a known signal: per-row uplift `u_i = sigmoid(x_i)`, then verify a perfect ranker beats random by a measurable margin and a random ranker stays within ±0.03 of zero. Those tests caught the Qini sign-flip described above, caught a propensity-weighting direction bug in the X-learner, and caught a dtype regression in the parquet writer. Each test ran red first, then green after the minimal implementation. The cycle felt slow on the first task; by the fifth it was the fastest way to ship the math.

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
make ci         # lint + fmt-check + type + test-cov + em-dash-check (mirrors GHA)
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
│   ├── download_data.py                   # primary URL + HF mirror fallback
│   ├── prepare_data.py
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
├── artifacts/                             # produced locally by make score/eval/dowhy; gitignored unless committed via LFS
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
| [`blog/post.md`](blog/post.md) | Short technical write-up |

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
- **Git LFS** rules in `.gitattributes` are scoped to `artifacts/{scored_sample.parquet, leaderboard.parquet, dowhy_refutation.json}`. These files do NOT ship with the repo; they are produced locally by `make score` / `make eval` / `make dowhy` and only committed via LFS if you choose to publish them. Trained models are gitignored; reproduce with `make train-all`.
- **Branch policy**: develop on a feature branch (e.g. `claude/<topic>`) and open a PR to `main`. Do not push to `main` without explicit user request.

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
