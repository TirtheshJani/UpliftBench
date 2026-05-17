"""Double Machine Learning via EconML's LinearDML with LightGBM nuisance models."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from econml.dml import LinearDML

from upliftbench.config import LIGHTGBM_PARAMS


def _classifier() -> lgb.LGBMClassifier:
    p = dict(LIGHTGBM_PARAMS)
    p.pop("objective", None)
    p.pop("metric", None)
    return lgb.LGBMClassifier(**p)


def _regressor() -> lgb.LGBMRegressor:
    p = dict(LIGHTGBM_PARAMS)
    p.pop("objective", None)
    p.pop("metric", None)
    return lgb.LGBMRegressor(**p)


class LinearDMLEstimator:
    name = "dml"

    def __init__(self) -> None:
        self._model = LinearDML(
            model_y=_regressor(),
            model_t=_classifier(),
            discrete_treatment=True,
            cv=3,
        )
        self._mu0: lgb.LGBMClassifier | None = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> None:
        self._model.fit(Y=Y.astype(float), T=T.astype(int), X=X)
        mask_c = T == 0
        self._mu0 = _classifier()
        self._mu0.fit(X[mask_c], Y[mask_c])

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        cate = self._model.effect(X)
        return np.asarray(cate, dtype=np.float64).ravel()

    def predict_baseline(self, X: np.ndarray) -> np.ndarray:
        if self._mu0 is None:
            raise RuntimeError("LinearDMLEstimator is not fitted")
        return np.asarray(self._mu0.predict_proba(X)[:, 1], dtype=np.float64)
