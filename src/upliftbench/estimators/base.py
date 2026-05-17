"""Base protocol every uplift estimator implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class BaseUpliftEstimator(Protocol):
    name: str

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> None: ...

    def predict_cate(self, X: np.ndarray) -> np.ndarray: ...

    def predict_baseline(self, X: np.ndarray) -> np.ndarray: ...
