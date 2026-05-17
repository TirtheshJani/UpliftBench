"""S-learner: single LightGBM with treatment as an input feature.

CATE = mu(x, T=1) - mu(x, T=0).
baseline = mu(x, T=0).
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np

from upliftbench.config import LIGHTGBM_PARAMS


class SLearner:
    name = "s-learner"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(LIGHTGBM_PARAMS if params is None else params)
        self._n_estimators = int(self.params.pop("n_estimators", 200))  # type: ignore[call-overload]
        self.model_: lgb.Booster | None = None

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> None:
        X_aug = np.column_stack([X, T.astype(np.float32)])
        dataset = lgb.Dataset(X_aug, label=Y.astype(np.float32), free_raw_data=True)
        self.model_ = lgb.train(self.params, dataset, num_boost_round=self._n_estimators)

    def _predict(self, X: np.ndarray, treatment: int) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("SLearner is not fitted")
        t_col = np.full((len(X), 1), treatment, dtype=np.float32)
        X_aug = np.column_stack([X, t_col])
        return np.asarray(self.model_.predict(X_aug), dtype=np.float64)

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return self._predict(X, 1) - self._predict(X, 0)

    def predict_baseline(self, X: np.ndarray) -> np.ndarray:
        return self._predict(X, 0)
