"""T-learner: two LightGBM models, one per treatment arm.

CATE = mu_1(x) - mu_0(x). baseline = mu_0(x).
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np

from upliftbench.config import LIGHTGBM_PARAMS


class TLearner:
    name = "t-learner"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(LIGHTGBM_PARAMS if params is None else params)
        self._n_estimators = int(self.params.pop("n_estimators", 200))  # type: ignore[call-overload]
        self.model_treated_: lgb.Booster | None = None
        self.model_control_: lgb.Booster | None = None

    def _train_one(self, X: np.ndarray, Y: np.ndarray) -> lgb.Booster:
        dataset = lgb.Dataset(X.astype(np.float32), label=Y.astype(np.float32), free_raw_data=True)
        return lgb.train(self.params, dataset, num_boost_round=self._n_estimators)

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> None:
        mask_t = T == 1
        mask_c = ~mask_t
        if mask_t.sum() == 0 or mask_c.sum() == 0:
            raise ValueError("Need both treated and control rows to fit a T-learner")
        self.model_treated_ = self._train_one(X[mask_t], Y[mask_t])
        self.model_control_ = self._train_one(X[mask_c], Y[mask_c])

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        if self.model_treated_ is None or self.model_control_ is None:
            raise RuntimeError("TLearner is not fitted")
        mu1 = np.asarray(self.model_treated_.predict(X), dtype=np.float64)
        mu0 = np.asarray(self.model_control_.predict(X), dtype=np.float64)
        return mu1 - mu0

    def predict_baseline(self, X: np.ndarray) -> np.ndarray:
        if self.model_control_ is None:
            raise RuntimeError("TLearner is not fitted")
        return np.asarray(self.model_control_.predict(X), dtype=np.float64)
