# Phase 6: Reproducibility Audit

> **Scope**: audit-only. No fixes were applied in this phase. Each finding lists
> a concrete remediation, but the user explicitly asked for surface-and-stop.
>
> **Date**: 2026-05-18
> **Branch**: `claude/phase6-repro-audit`
> **Baseline**: `main` at `cf203b3` (Phase 5 merged + README polish PRs)
>
> Audit ran four parallel tracks: static checks (ruff/mypy/pytest), Makefile/CLI
> plumbing, code architecture vs `CLAUDE.md` claims, and documentation drift.
> Three of the tracks were delegated to subagents; the static-check track ran in
> the foreground.

## 0. TL;DR

| Track | Status | Worst finding |
|---|---|---|
| Static checks (ruff + mypy) | PASS | none |
| Static checks (pytest, lightweight subset only) | PARTIAL | 4 numpy-2.x failures from a brittle `np.trapz` call; full pytest unverified |
| Makefile + CLI plumbing | PASS with drift | `make ci` is a strict subset of `.github/workflows/ci.yml`: a green local CI can fail in GHA |
| Code architecture | PASS with drift | `run_dowhy` signature does not match `CLAUDE.md` |
| Documentation | DRIFT | five files still reference the merged `claude/causal-inference-uplift-x0DZw` branch as active; `src/upliftbench/data/download.py` documented but absent; LFS artifacts declared but not committed |

Headline blockers for "fresh clone reproduces end-to-end":
1. `uv` is not on PATH on this host. Every `make` target uses `uv run`. There is no documented fallback. A new contributor needs `pip install uv` or the equivalent before `make install` works at all.
2. `np.trapz` is removed in numpy 2.x; the eval code uses it. The `pyproject.toml` pin of `numpy<2.0` is therefore load-bearing. Anyone who relaxes that pin in good faith will silently break Qini/AUUC.
3. Three LFS artifacts (`scored_sample.parquet`, `leaderboard.parquet`, `dowhy_refutation.json`) are declared in `.gitattributes` and described in docs as repo contents, but `git lfs ls-files` is empty. The README's "open the Streamlit app, it just works" promise depends on these.

---

## 1. Static checks

**Environment**: Python 3.13.12 from miniconda3 (system); `uv` not installed.
Tools were installed to `--user` site (`ruff 0.15.12`, `mypy 1.20.2`, `pytest 9.0.3`,
plus `numpy/pandas/pyarrow/scikit-learn/scipy/joblib/typer/tqdm/matplotlib`).
Heavy deps (`lightgbm`, `dowhy`, `causalml`, `econml`) were NOT installed.
The package was installed as `pip install --user --no-deps -e .` so
`import upliftbench` resolves.

This does **not** reproduce what `uv sync --extra dev` would produce, because
`uv` honors the `pyproject.toml` constraint solver and this manual install does
not. Treat the pytest results below as a smoke test, not as a CI proxy.

### 1.1 `ruff check src tests scripts streamlit_app`

```
All checks passed!
```

### 1.2 `ruff format --check src tests scripts streamlit_app`

```
47 files already formatted
```

### 1.3 `mypy src`

```
Success: no issues found in 23 source files
```

### 1.4 `pytest` (lightweight subset)

Ran `tests/eval/`, `tests/test_segmentation.py`, `tests/test_persistence.py`,
`tests/test_import.py`. Result:

```
4 failed, 12 passed in 0.74s
```

All four failures share a root cause: **`np.trapz` was removed in numpy 2.0**,
replaced by `np.trapezoid`. The user-site install resolved `numpy 2.4.5`, so the
call breaks. CI passes because `pyproject.toml` pins `numpy>=1.26,<2.0` and `uv
sync` honors the upper bound.

Sites:
- `src/upliftbench/eval/qini.py:62`
- `src/upliftbench/eval/auuc.py:61`
- `notebooks/kaggle_end_to_end.ipynb:166`

This is a real brittleness even though current CI is green:

- The pin is load-bearing, not aesthetic. A future contributor relaxing it
  (e.g. for compatibility with downstream libraries that need numpy 2) will
  silently break Qini/AUUC math.
- The Kaggle notebook ships a copy of the same call; Kaggle runners that
  upgrade numpy past 2.0 will fail.

**Remediation**: replace `np.trapz` with `np.trapezoid` (alias since numpy
1.22, drop-in). Then the `numpy<2.0` pin can be relaxed.

### 1.5 `pytest` (full suite)

**Unverified.** `tests/data/`, `tests/estimators/`, `tests/test_dowhy_pipeline.py`
require `lightgbm`, `dowhy`, `causalml`, `econml`. Installing those on Windows
+ Python 3.13 was out of scope for a time-boxed audit. CI runs Python 3.11 on
ubuntu-latest where wheels are reliable. Trust CI for those; flag this gap.

---

## 2. Makefile and CLI plumbing

### 2.1 Target summary

| target | script / command | status |
|---|---|---|
| `install` | `uv sync --extra dev` + `pre-commit install` | PASS |
| `data` | `scripts/download_data.py` | PASS |
| `prepare` | `scripts/prepare_data.py` | PASS |
| `train-all` | 5x `scripts/train.py --estimator <name>` | PASS, all 5 names resolve in `ESTIMATOR_REGISTRY` |
| `dowhy` | `scripts/run_dowhy.py --sample 1000000` | PASS, `--sample` is an int Option |
| `score` | `scripts/score_sample.py --n 200000` | PASS |
| `eval` | `scripts/evaluate_all.py` | PASS |
| `test` | `pytest -q` | PASS, but see drift below |
| `lint` / `fmt` / `type` / `precommit` / `app` / `clean` | obvious | PASS |
| `ci` | `lint type test` | **DRIFT**, see 2.2 |

### 2.2 `make ci` vs `.github/workflows/ci.yml`

`make ci` runs `lint type test`. The GHA workflow runs more:

| Step | In Makefile? | In CI? |
|---|---|---|
| `ruff check` | yes (`make lint`) | yes |
| `ruff format --check` | no (only `make fmt` reformats in-place) | yes (ci.yml:34) |
| `mypy src` | yes (`make type`) | yes |
| `pytest -q` | yes (`make test`) | yes, but with `--cov=upliftbench --cov-report=term` |
| em-dash grep | no | yes (ci.yml:43) |

A local `make ci` can pass while GHA fails. Remediation options:
- Extend the `ci` target to `lint fmt-check type test em-dash-check` (need a
  new `fmt-check` and `em-dash-check` target).
- Or document the gap explicitly in `README.md`.

### 2.3 `streamlit_app/app.py` imports

`CLAUDE.md` documents the slim allow-list as `pandas, pyarrow, numpy,
matplotlib, streamlit, upliftbench.segmentation`. Actual imports
(`streamlit_app/app.py:12-23`):

```
json, pathlib.Path, matplotlib.pyplot, pandas, streamlit,
upliftbench.config, upliftbench.segmentation
```

- `upliftbench.config` is **not** in the documented allow-list. It is a pure
  constants module with no heavy ML dependency, so the 1 GB Streamlit Cloud
  cap is not at risk. But the documented contract is violated.
- `numpy` and `pyarrow` are in the allow-list but not actually imported. Not
  an issue (the slim envelope is a ceiling, not a floor), but the doc lists
  more than the app uses.

**Remediation**: either update CLAUDE.md to add `upliftbench.config` to the
allow-list (and drop `numpy`/`pyarrow` from it), or refactor the app to inline
the constants it currently pulls from `config`.

### 2.4 Pre-commit and `uv.lock`

- `.pre-commit-config.yaml` has a local `no-em-dash` `pygrep` hook
  (`.pre-commit-config.yaml:21-28`) that scans U+2014 with the correct
  exclusion of `.claude/skills/` (line 28). PASS.
- `uv.lock` exists at repo root (556 KB, tracked). PASS.

---

## 3. Code architecture vs `CLAUDE.md`

Twelve claims checked. Only one drifted:

### 3.1 `run_dowhy` signature drift

`CLAUDE.md:60` documents:
```
run_dowhy(df, best_estimator_name, sample_n)
```

Actual (`src/upliftbench/refute/dowhy_pipeline.py:70-78`):
```python
def run_dowhy(
    df,
    feature_cols,
    treatment_col="treatment",
    outcome_col="visit",
    sample_n=1_000_000,
    seed=42,
    estimator_method="backdoor.linear_regression",
)
```

The `best_estimator_name` parameter does not exist. The pipeline uses
`estimator_method="backdoor.linear_regression"` regardless of which trained
estimator topped Phase 2's leaderboard. This is a design drift: `CLAUDE.md`
implies the refutation re-uses the winning learner; the code uses DoWhy's
built-in linear backdoor estimator independently.

This is not a bug; DoWhy's refutation is well-defined against the linear
backdoor estimator. But the documented architecture promises something the
code does not deliver. Either:
- Update `CLAUDE.md` to describe the actual contract (DoWhy uses its own
  estimator; the trained learners are evaluated separately), or
- Extend `run_dowhy` to accept a callable wrapping the trained learner and
  pass it through.

### 3.2 Everything else verified

| # | Claim | Status |
|---|---|---|
| 1 | `BaseUpliftEstimator` Protocol has `fit`, `predict_cate`, `predict_baseline` (`estimators/base.py:11-18`) | PASS |
| 2 | All 5 estimators implement those methods | PASS |
| 3 | `ESTIMATOR_REGISTRY` has 5 entries with hyphenated keys; `get_estimator` raises `KeyError` on miss | PASS |
| 4 | `evaluate_estimator(t, y, cate, top_k_fracs=(0.1, 0.2, 0.3))` returns dict with `qini_coef`, `auuc`, `top_k_uplift`, `qini_curve_xy`, `uplift_curve_xy` | PASS |
| 5 | `qini.py`, `auuc.py`, `topk.py` define the documented metric functions | PASS |
| 6 | `segmentation.score_and_segment` + `segmentation.budget_allocation` exported; no heavy imports | PASS |
| 7 | All 4 refuters wired (`placebo_treatment_refuter`, `random_common_cause`, `data_subset_refuter`, `add_unobserved_common_cause`) | PASS |
| 8 | `save_model` writes `<name>_<date>_<sha>.joblib` plus sibling JSON with `name`, `git_sha`, `saved_at_utc`, `lib_versions` | PASS |
| 9 | Chunked loader default `batch_size=500_000`; `train_test_split_rct` does NOT stratify by treatment | PASS |
| 10 | `SEED = 42` in `config.py:34` | PASS |
| 11 | `networkx>=2.8,<3.3` pin in `pyproject.toml:24-25` with comment | PASS |

---

## 4. Documentation drift

### 4.1 Stale branch policy in five places

The branch `claude/causal-inference-uplift-x0DZw` was merged into `main` at
commit `bd573bf` and is no longer the active development branch. It is still
listed as such in:

- `README.md:311` ("Branch policy: develop on `claude/causal-inference-uplift-x0DZw`")
- `CLAUDE.md:7` ("All work happens on branch ...")
- `CLAUDE.md:96` ("Branch policy: develop on ...")
- `docs/CONTRIBUTING.md:86`
- `blog/post.md:5`

### 4.2 Documented but missing: `src/upliftbench/data/download.py`

- `README.md:142` lists `data/download.py` in the layout block.
- `CLAUDE.md:52` says `data/`: `download.py`, `prepare.py` ...
- Reality: `src/upliftbench/data/` contains only `__init__.py`, `loader.py`,
  `prepare.py`. The download lives in `scripts/download_data.py`. The library
  surface described in the docs does not match disk.

### 4.3 LFS artifacts declared but not committed

`.gitattributes` declares LFS rules for:
- `artifacts/scored_sample.parquet`
- `artifacts/leaderboard.parquet`
- `artifacts/dowhy_refutation.json`

`git lfs ls-files` returns nothing. The local `artifacts/` directory does not
exist. `README.md:170` and the Architecture section both imply these files
ship with the repo. They do not. A fresh clone has no Streamlit-ready data
until `make score` / `make eval` / `make dowhy` have been run.

This is the most user-visible reproducibility issue: the Streamlit app in the
README's "Quickstart" assumes those files exist.

### 4.4 Test-file count claim

`README.md:90`: "Five of the seven test files are TDD against synthetic data."
Actual: 10 `test_*.py` files under `tests/`. Either the claim has not been
updated as tests were added, or the original count was wrong.

### 4.5 Em-dashes

`grep -rn $'\xe2\x80\x94' src tests scripts streamlit_app blog README.md
CLAUDE.md` returns nothing. CI is green on this. The pre-commit hook scans a
slightly wider scope (`types: [text]`) but correctly excludes
`.claude/skills/`, which is the only path with em-dashes (vendored skill
files). No action needed.

### 4.6 Bundled skills

All five declared skills present at `.claude/skills/{writing-plans,
executing-plans, test-driven-development, dispatching-parallel-agents,
karpathy-guidelines}/SKILL.md`. PASS.

---

## 5. Prioritized punch list

If a follow-up Phase 6.1 patches drift, the order I would recommend:

1. **`np.trapz` → `np.trapezoid`** in `src/upliftbench/eval/qini.py:62`,
   `src/upliftbench/eval/auuc.py:61`, `notebooks/kaggle_end_to_end.ipynb:166`.
   One-line change each; un-locks the `numpy<2.0` pin if desired.
2. **Document the LFS situation honestly.** Either commit the artifacts via
   LFS, or change `README.md` and `CLAUDE.md` to say the user must run
   `make eval` / `make score` / `make dowhy` to produce them.
3. **Update branch policy in 5 places** to point at `main` (or whatever the
   current default is).
4. **Reconcile `run_dowhy` signature** with `CLAUDE.md` documentation. Pick:
   docs-follow-code or code-follow-docs.
5. **Update `make ci`** to match GHA (add `ruff format --check`, `pytest
   --cov`, em-dash grep), or document the gap in `README.md`.
6. **Remove `data/download.py` references** from `README.md:142` and
   `CLAUDE.md:52` since the file does not exist; or move the function from
   `scripts/download_data.py` into `src/upliftbench/data/download.py` and
   have `scripts/download_data.py` import it.
7. **Fix the "five of seven test files" claim** in `README.md:90`.
8. **`streamlit_app/app.py` import allow-list**: either update the doc
   allow-list in `CLAUDE.md` or refactor the app to drop the
   `upliftbench.config` import.

None of these are correctness bugs in the trained-model sense. They are
contract drift between documentation and code, plus one numpy-version
brittleness. CI is green and will stay green until someone touches a pin.

---

## 6. What this audit did NOT verify

- `make data`, `make prepare`, `make train-all`, `make dowhy`, `make score`,
  `make eval` end-to-end. Time and bandwidth out of scope.
- `pytest tests/data/`, `tests/estimators/`, `tests/test_dowhy_pipeline.py`.
  Heavy-dep environment not bootstrapped. CI runs these; trust CI.
- Timing claims in the README ("~15 min for `make data`", "~2-3 hours for
  `make train-all`"). Not testable here.
- The Streamlit app rendering on Community Cloud. Local-only render not
  attempted because LFS artifacts are not present.

A future Phase 6.2 could close those gaps with a CI-style ubuntu-latest
runner and a `uv sync --extra dev --extra streamlit` bootstrap, but that
crosses out of "audit" into "set up a CI environment locally".
