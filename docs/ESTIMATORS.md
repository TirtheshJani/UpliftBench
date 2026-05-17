# Estimators

All five estimators implement the same `BaseUpliftEstimator` protocol:

```python
class BaseUpliftEstimator(Protocol):
    name: str
    def fit(self, X, T, Y) -> None: ...
    def predict_cate(self, X) -> np.ndarray: ...      # heterogeneous treatment effect
    def predict_baseline(self, X) -> np.ndarray: ...  # control-arm response, needed for segmentation
```

They register themselves into `ESTIMATOR_REGISTRY` so the train CLI can dispatch via a single name string.

## S-learner

`src/upliftbench/estimators/s_learner.py`

One LightGBM classifier `mu(x, T)` trained on the concatenation of features and the treatment column. Then:

- `predict_cate(X) = mu(X, T=1) - mu(X, T=0)`
- `predict_baseline(X) = mu(X, T=0)`

Strengths: cheap, one model. Weakness: when treatment is a weak feature relative to `X`, the model often ignores it, biasing CATE toward zero.

## T-learner

`src/upliftbench/estimators/t_learner.py`

Two independent LightGBM classifiers:

- `mu_1(x)` trained on `(X, Y)` over treated rows only.
- `mu_0(x)` trained on `(X, Y)` over control rows only.
- `predict_cate = mu_1 - mu_0`
- `predict_baseline = mu_0`

Strengths: each arm has full freedom to fit its response. Weakness: when one arm has far fewer rows, that arm's model is undertrained and `mu_1 - mu_0` carries the noise.

## X-learner

`src/upliftbench/estimators/x_learner.py`

Implemented from scratch with LightGBM (the CausalML wrapper hits a scipy/pygam version conflict in this Python environment). Follows Kunzel, Sekhon, Bickel, Yu (2019).

1. Train `mu_0` on control rows, `mu_1` on treated rows.
2. Impute individual treatment effects:
   - On treated rows: `D_1 = Y - mu_0(X)`
   - On control rows: `D_0 = mu_1(X) - Y`
3. Train two CATE regressors: `tau_1` on `(X_treated, D_1)`, `tau_0` on `(X_control, D_0)`.
4. Combine: `CATE(x) = g(x) * tau_0(x) + (1 - g(x)) * tau_1(x)`.

Under the RCT, the propensity `g(x)` is approximately constant; we use the empirical treatment rate.

Strengths: cross-fits residuals, less biased than naive T-learner on imbalanced arms. Weakness: more models, more places to overfit.

## DR-learner

`src/upliftbench/estimators/dr_learner.py`

EconML's `DRLearner` wrapped to expose our protocol. **Doubly robust**: consistency of the CATE estimate requires only ONE of the propensity model OR the outcome regression to be correctly specified, not both. 3-fold cross-fitting; LightGBM nuisances.

Note: EconML's `DRLearner` regards the outcome as continuous by default. We pass `LGBMRegressor` for `model_regression` even though `visit` is binary, and we keep a separate `LGBMClassifier` for `predict_baseline` so segmentation has a calibrated probability to work with.

## Double Machine Learning (LinearDML)

`src/upliftbench/estimators/dml.py`

EconML's `LinearDML` (Chernozhukov et al., 2018). Residualizes outcome on covariates and treatment on covariates, then regresses outcome residual on treatment residual. The "Linear" variant fits a linear final model on the residuals; CausalForestDML would be the nonparametric alternative (left out for the laptop CPU budget).

## Hyperparameters

All LightGBM bases share the same dict (`config.LIGHTGBM_PARAMS`):

```python
{
  "objective": "binary",        # (popped by sklearn wrappers, kept for raw lgb.train)
  "learning_rate": 0.05,
  "num_leaves": 63,
  "max_bin": 63,                # keeps memory under control on 13.9M rows
  "min_data_in_leaf": 200,
  "feature_fraction": 0.9,
  "bagging_fraction": 0.9,
  "bagging_freq": 5,
  "n_estimators": 200,
  "n_jobs": -1,
  "verbose": -1,
}
```

Tuning these per-estimator could squeeze more out, but a uniform setting keeps the cross-estimator comparison fair.
