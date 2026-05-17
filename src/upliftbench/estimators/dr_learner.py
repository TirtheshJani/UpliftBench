"""DR-learner (doubly robust) backed by EconML."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
from econml.dr import DRLearner as EconmlDRLearner

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


class DRLearner:
    name = "dr-learner"

    def __init__(self) -> None:
        # `model_regression` is the outcome regressor under DR; econml asks for a regressor
        # interface even when the outcome is binary, so we keep an LGBMRegressor here.
        self._model = EconmlDRLearner(
            model_propensity=_classifier(),
            model_regression=_regressor(),
            model_final=_regressor(),
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
            raise RuntimeError("DRLearner is not fitted")
        return np.asarray(self._mu0.predict_proba(X)[:, 1], dtype=np.float64)
