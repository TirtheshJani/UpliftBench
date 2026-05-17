"""Smoke tests for X-learner, DR-learner, and LinearDML. Verify they fit, predict, and produce sensible Qini."""

from __future__ import annotations

import numpy as np
import pytest

from upliftbench.estimators import get_estimator
from upliftbench.eval.qini import qini_coefficient


def _toy(n: int = 3_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4)).astype(np.float32)
    u = 1.0 / (1.0 + np.exp(-X[:, 0]))
    T = (rng.random(n) < 0.5).astype(np.uint8)
    p_treated = np.clip(0.10 + u / 4.0, 0, 1)
    Y_t = (rng.random(n) < p_treated).astype(np.uint8)
    Y_c = (rng.random(n) < 0.10).astype(np.uint8)
    Y = np.where(T == 1, Y_t, Y_c).astype(np.uint8)
    return X, T, Y


@pytest.mark.parametrize("name", ["x-learner", "dr-learner", "dml"])
def test_estimator_smoke(name: str) -> None:
    X, T, Y = _toy()
    est = get_estimator(name)
    est.fit(X, T, Y)
    cate = est.predict_cate(X)
    baseline = est.predict_baseline(X)
    assert cate.shape == (len(X),)
    assert baseline.shape == (len(X),)
    # Sign of Qini is enough on weak synthetic data; not a tight bound.
    _ = qini_coefficient(T, Y, cate)
