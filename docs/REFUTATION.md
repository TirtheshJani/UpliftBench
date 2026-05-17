# DoWhy refutation

`src/upliftbench/refute/dowhy_pipeline.py::run_dowhy` runs the four standard refuters against the strongest estimator (best by Qini, recorded in `artifacts/best_estimator.txt`). Refutation is the "step 4" of DoWhy's four-step pipeline:

1. **Model** the causal graph: treatment, outcome, common causes.
2. **Identify** an estimand via backdoor adjustment.
3. **Estimate** the effect (default: `backdoor.linear_regression`).
4. **Refute** the estimate.

A refuter perturbs an assumption and asks "did the estimate move in the expected direction?". Four standard ones, all run by `run_dowhy`:

## `placebo_treatment_refuter`

Replace the real treatment column with a freshly randomized one and re-estimate. If the original estimate was real, the placebo estimate should be near zero. A p-value > 0.05 says we cannot reject "placebo estimate equals zero", which is what we want.

## `random_common_cause`

Append a random covariate to the data and re-estimate. If we have the right backdoor set, the new estimate should be within a few percent of the original. Big movement here indicates the model is sensitive to including spurious covariates, which is a problem.

## `data_subset_refuter`

Re-run on a random row subset. The estimate should be stable to subsampling (under an RCT, it is).

## `add_unobserved_common_cause`

Simulate a hidden confounder of varying strength and ask how strong it would have to be to drive the observed effect to zero. The further away that threshold is from realistic confounder strengths, the more robust the conclusion. This is the most cited refuter in the DoWhy literature.

## Sample-size choice

DoWhy's refutation methods bootstrap-resample and re-fit; on 13.9M rows that is intractable on a laptop CPU. `run_dowhy` therefore takes a **stratified-by-treatment subsample of 1M rows** by default. The stratification preserves the treatment ratio (and hence the RCT property).

Larger samples would be more precise but the marginal gain on 1M rows is small for an RCT this size; the practical bottleneck is laptop wall time.

## What `run_dowhy` returns

```python
{
  "estimand":          str,        # the DoWhy estimand expression
  "estimate":          float,      # the point ATE
  "refutations": {
    "placebo_treatment":        {"new_estimate": ..., "p_value": ..., "test_name": ...},
    "random_common_cause":      {"new_estimate": ..., "p_value": ..., "test_name": ...},
    "data_subset":              {"new_estimate": ..., "p_value": ..., "test_name": ...},
    "unobserved_common_cause":  {"new_estimate": ..., "p_value": ..., "test_name": ...},
  },
  "n":                 int,        # rows in the subsample
  "estimator_method":  str,
}
```

`scripts/run_dowhy.py` writes this dict to `artifacts/dowhy_refutation.json`. The Streamlit app reads it for the refutation table; `notebooks/04_dowhy_refutation.ipynb` and the blog post render it.

## Library pin

DoWhy 0.12 still uses the `networkx.algorithms.d_separated` symbol that was removed in networkx 3.3. `pyproject.toml` therefore pins `networkx>=2.8,<3.3`. If you upgrade DoWhy past a version that fixes this, drop the pin.
