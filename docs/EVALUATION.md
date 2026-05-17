# Evaluation

UpliftBench reports three quantities for every estimator: Qini coefficient, AUUC, and top-K uplift. All three are implemented from scratch in `src/upliftbench/eval/` and TDD'd against synthetic data with known signal.

## Qini curve and coefficient

Sort the population in descending order by predicted CATE. For each prefix of size `k`, define:

```
N_t(k) = number of treated in the top k
N_c(k) = number of control in the top k
Y_t(k) = sum of positive outcomes among treated in the top k
Y_c(k) = sum of positive outcomes among control in the top k

Q(k) = Y_t(k) - Y_c(k) * (N_t(k) / N_c(k))
```

`Q(k)` is the incremental positive outcomes attributable to treatment in the top `k`, with the control side rescaled to match the treated count. Plot `(k / N, Q(k))` and you get the Qini curve. A perfect ranker climbs steeply early and then plateaus at `Q(N)`. A random ranker is the straight line from `(0, 0)` to `(1, Q(N))`.

The Qini **coefficient** in this repo is:

```
qini_coef = 2 * (area_model - area_random) / |Q(N)|
```

where `area_model = ∫ Q(k) dk` and `area_random = Q(N) / 2` (the triangle). This normalization means a random ranker is near zero, a strong ranker is positive, and a worse-than-random ranker is negative. Some libraries divide by `(area_optimal - area_random)` instead, but the empirical "optimal" curve under heterogeneous binary outcomes is ill-defined; this normalization avoids that ambiguity.

See `src/upliftbench/eval/qini.py`.

## AUUC

The Area Under the Uplift Curve uses the rate difference instead of the count difference:

```
U(k) = (Y_t(k) / N_t(k) - Y_c(k) / N_c(k)) * k
```

i.e. the average rate gap times the prefix size. Plot `(k / N, U(k))` and you have the uplift curve. AUUC follows the same `2 * (area_model - area_random) / |U(N)|` normalization as Qini.

See `src/upliftbench/eval/auuc.py`.

## Top-K uplift

The empirical ATE in the top-K fraction:

```
top_k_uplift(t, y, cate, k=0.1) = mean(y[treated ∩ top_k]) - mean(y[control ∩ top_k])
```

Useful as a single business-friendly scalar. `evaluate_estimator` returns top-K at 10%, 20%, 30%.

## Anti-patterns the tests guard against

`tests/eval/test_qini.py` and `tests/eval/test_auuc.py` synthesize data with `u_i = sigmoid(x_i)` known per-row uplift, then assert:

- The curve starts at `(0, 0)` and ends at `Q(N)`.
- A perfect ranker (sorted by `x_i`) strictly beats a random ranker.
- The mean Qini over eight random rankers is within ±0.03 of zero.
- Input validation catches mismatched-length arrays and non-binary treatments.

These TDD checks ran red before any implementation existed, then green after the minimal implementation. If you later refactor the math, those tests are the brake.

## What `evaluate_estimator` returns

`src/upliftbench/eval/harness.py::evaluate_estimator(t, y, cate)` returns a dict with keys:

```python
{
  "qini_coef":      float,
  "auuc":           float,
  "top_k_uplift":   {"top_10": float, "top_20": float, "top_30": float},
  "qini_curve_xy":  (xs: ndarray, ys: ndarray),
  "uplift_curve_xy":(xs: ndarray, ys: ndarray),
}
```

Curve points are returned as `(xs, ys)` tuples so they can be persisted (json / parquet) and replotted without recomputing.
