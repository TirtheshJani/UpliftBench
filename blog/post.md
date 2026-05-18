# UpliftBench: five causal-inference estimators on 13.9M ad-targeting decisions

Causal inference is the fastest-growing required skill in 2026 senior ML/DS job descriptions. Snap, Booking, Wayfair, Uber, DoorDash, and Spotify's Experimentation Platform team all want the same thing: someone who can model a treatment effect, defend it under refutation, and turn a heterogeneous treatment effect estimate into a budget decision. This post benchmarks five canonical uplift estimators on the **Criteo Uplift Prediction Dataset v2** (13.9M RCT-style randomized ad-targeting decisions, free from Criteo AI Lab) using DoWhy, EconML, and CausalML, evaluates with Qini and AUUC, segments the population into the four canonical customer types, and ships a live Streamlit demo where you can move a treatment-budget slider and watch the persuadable segment update.

Repo: https://github.com/TirtheshJani/UpliftBench (development on feature branches, merged to `main` via PR).

## The five estimators

All five share a base learner (LightGBM, CPU-only, chunked) and a common interface (`fit`, `predict_cate`, `predict_baseline`) so they plug into the same evaluation harness and the same Streamlit app.

1. **S-learner** trains one model with treatment as a feature, then estimates CATE(x) as `mu(x, T=1) - mu(x, T=0)`. Simplest baseline. Tends to underestimate effects when treatment is a weak feature.
2. **T-learner** trains two models, one per arm. CATE(x) is `mu_1(x) - mu_0(x)`. Lets each arm fit its own response function but doubles model count and is biased when arms have different sample sizes.
3. **X-learner** (Kunzel et al., 2019) cross-fits residuals. It imputes individual treatment effects, then trains two CATE regressors and propensity-weights them. Implemented from scratch with LightGBM in this repo to avoid CausalML's pygam dependency conflict.
4. **DR-learner** is doubly robust: it combines a propensity model and an outcome regression so that consistency of the CATE estimate requires only ONE of the two to be correctly specified. EconML's `DRLearner` with 3-fold cross-fitting, LightGBM nuisances.
5. **Double Machine Learning** (Chernozhukov et al., 2018) residualizes both outcome and treatment against the covariates, then regresses one residual on the other. EconML's `LinearDML` with LightGBM nuisances.

## Evaluation: Qini and AUUC

The Qini curve plots cumulative incremental positive outcomes when you target the top-k fraction of the population sorted by predicted CATE:

```
Q(k) = Y_treated(k) - Y_control(k) * (N_treated(k) / N_control(k))
```

The Qini coefficient is `2 * (area_model - area_random) / |Q_total|`, so a random ranker is near zero, a strong ranker is positive, and a worse-than-random ranker is negative. AUUC follows the same shape but with rate-of-response normalized per k. Implementations are TDD'd against synthetic data where the true uplift is known analytically, so a perfect ranker beats random with high confidence.

## The four customer segments

Once we have `pred_cate(x)` and `pred_baseline(x)` (the control-arm response model), we split the population into four quadrants:

- **persuadables**: high uplift, low baseline. Treating them works.
- **sure_things**: low uplift, high baseline. They convert anyway; treating wastes budget.
- **lost_causes**: low uplift, low baseline. Treatment will not help.
- **do_not_disturb**: positive uplift sign but high baseline; classical uplift literature flags these as actively negative ROI.

The Streamlit demo shows you how the four-way mix shifts as you change the treatment budget percentage. Most useful insight: at very small budgets the targeted population is almost all persuadables, but as the budget grows the marginal user added is increasingly a sure-thing or a do-not-disturb. You can see the diminishing returns interactively.

## DoWhy four-step pipeline

For the strongest estimator (DR-learner in my run on the full dataset) we wrap the whole pipeline in DoWhy:

1. **Model**: a `CausalModel` with `treatment`, `visit` (outcome), and the 12 features as common causes.
2. **Identify**: `identify_effect(proceed_when_unidentifiable=True)`, backdoor adjustment.
3. **Estimate**: `estimate_effect(method_name="backdoor.linear_regression")` for a point ATE.
4. **Refute**: all four standard refuters:
   - `placebo_treatment_refuter`: replace the real treatment with a random one; the new estimate should be near zero, p-value > 0.05.
   - `random_common_cause`: add a random covariate; the new estimate should be within a few percent of the original.
   - `data_subset_refuter`: re-run on a random subset; the estimate should be stable.
   - `add_unobserved_common_cause`: simulate a hidden confounder of varying strength; the new estimate quantifies how robust the conclusion is.

DoWhy refutation on 13.9M rows is intractable on a laptop, so we run all four refuters on a **1M-row sample stratified by treatment** to preserve the RCT ratio. The repo documents this choice in code (`run_dowhy(sample_n=1_000_000)`) and in the readme.

## Hardware and memory budget

The work runs on an RTX 4080 laptop with **16 GB system RAM** (the project is CPU-only, no GPU). The Criteo CSV.gz is ~700 MB compressed and decompresses to ~3 GB. The whole pipeline:

1. Stream the CSV directly to parquet via pyarrow with float32 features and uint8 treatment/visit/conversion. One-shot, peak RSS under 4 GB.
2. Iterate parquet batches at 500k rows. LightGBM `Dataset(free_raw_data=True, max_bin=63)` keeps peak training RSS under 6 GB on full data.
3. If full-data training ever exceeds 12 GB, the plan is to fall back to a 5M-row stratified-by-treatment sample. The plan documents this; in my runs the full data fits.

## Deliverables

The repo gives you four artifacts:

1. **GitHub repo**: this codebase. Reproducible with `uv sync && make data && make prepare && make train-all && make dowhy && make eval && make score && make app`.
2. **Streamlit Community Cloud demo**: budget slider over a pre-scored 200k-row sample (the app does no inference; it only filters and segments). The slim Streamlit extra excludes lightgbm/dowhy/causalml/econml so the Cloud install stays under the 1 GB cap.
3. **Kaggle notebook**: `notebooks/kaggle_end_to_end.ipynb` reproduces the five-estimator comparison on a 2M-row subsample (fits Kaggle's 9-hour CPU wall).
4. **This blog post**.

## Repository conventions

- **uv + pyproject + src/ layout**. No `requirements.txt`.
- **TDD** for the eval harness, segmentation, data loaders, and persistence (where math correctness is verifiable on synthetic data with known signal). Smoke tests for the estimators themselves.
- **Pre-commit + GitHub Actions CI on ubuntu-latest only**, ruff format + ruff check + mypy + pytest, plus a grep check that forbids em dashes in any committed file.
- **Five bundled skills** under `.claude/skills/` so any future Claude Code session in this repo auto-loads writing-plans, executing-plans, TDD, parallel-agent dispatch, and Karpathy guidelines.

## What to look at first

If you have five minutes, open the Streamlit demo and move the budget slider. The persuadable count rises slowly and then plateaus as you target more of the population; the do-not-disturb count rises mostly linearly. That single chart is what causal inference buys you over a vanilla response model.

If you have an hour, read `src/upliftbench/eval/qini.py` (forty lines, TDD'd) and `src/upliftbench/segmentation.py` (the shared API). Those two files plus the registry in `src/upliftbench/estimators/__init__.py` are the entire conceptual core.
