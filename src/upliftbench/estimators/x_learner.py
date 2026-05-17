"""X-learner with LightGBM bases, implemented directly to avoid causalml's pygam pin.

Algorithm (Kunzel et al., 2019):

1. mu_0(x) trained on control rows; mu_1(x) trained on treated rows.
2. Imputed individual treatment effects:
     D_1 = Y - mu_0(x)  on treated rows
     D_0 = mu_1(x) - Y  on control rows
3. tau_1(x) regressed on D_1 from treated rows; tau_0(x) on D_0 from control rows.
4. CATE(x) = g(x) * tau_0(x) + (1 - g(x)) * tau_1(x).
   Under an RCT we use the constant propensity g = P(T=1).
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np

from upliftbench.config import LIGHTGBM_PARAMS


def _classifier_params() -> dict[str, Any]:
    p = dict(LIGHTGBM_PARAMS)
    p.pop("objective", None)
    p.pop("metric", None)
    return p


def _regressor_params() -> dict[str, Any]:
    p = dict(LIGHTGBM_PARAMS)
    p.pop("objective", None)
    p.pop("metric", None)
    return p


class XLearner:
    name = "x-learner"

    def __init__(self) -> None:
        self._mu0: lgb.LGBMClassifier | None = None
        self._mu1: lgb.LGBMClassifier | None = None
        self._tau0: lgb.LGBMRegressor | None = None
        self._tau1: lgb.LGBMRegressor | None = None
        self._propensity: float = 0.5

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> None:
        mask_t = T == 1
        mask_c = ~mask_t
        if mask_t.sum() == 0 or mask_c.sum() == 0:
            raise ValueError("Need both treated and control rows to fit XLearner")
        self._propensity = float(mask_t.mean())

        cls_params = _classifier_params()
        self._mu0 = lgb.LGBMClassifier(**cls_params)
        self._mu0.fit(X[mask_c], Y[mask_c])
        self._mu1 = lgb.LGBMClassifier(**cls_params)
        self._mu1.fit(X[mask_t], Y[mask_t])

        mu0_proba = self._mu0.predict_proba(X[mask_t])[:, 1]
        mu1_proba = self._mu1.predict_proba(X[mask_c])[:, 1]
        d1 = Y[mask_t].astype(np.float64) - mu0_proba
        d0 = mu1_proba - Y[mask_c].astype(np.float64)

        reg_params = _regressor_params()
        self._tau1 = lgb.LGBMRegressor(**reg_params)
        self._tau1.fit(X[mask_t], d1)
        self._tau0 = lgb.LGBMRegressor(**reg_params)
        self._tau0.fit(X[mask_c], d0)

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        if self._tau0 is None or self._tau1 is None:
            raise RuntimeError("XLearner is not fitted")
        t0 = np.asarray(self._tau0.predict(X), dtype=np.float64)
        t1 = np.asarray(self._tau1.predict(X), dtype=np.float64)
        g = self._propensity
        return g * t0 + (1.0 - g) * t1

    def predict_baseline(self, X: np.ndarray) -> np.ndarray:
        if self._mu0 is None:
            raise RuntimeError("XLearner is not fitted")
        return np.asarray(self._mu0.predict_proba(X)[:, 1], dtype=np.float64)
