"""Smoke tests: S-learner and T-learner train and produce a positive Qini on synthetic uplift."""

from __future__ import annotations

import numpy as np

from upliftbench.estimators.s_learner import SLearner
from upliftbench.estimators.t_learner import TLearner
from upliftbench.eval.qini import qini_coefficient


def _toy(n: int = 5_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4)).astype(np.float32)
    u = 1.0 / (1.0 + np.exp(-X[:, 0]))
    T = (rng.random(n) < 0.5).astype(np.uint8)
    p_treated = np.clip(0.10 + u / 4.0, 0, 1)
    Y_t = (rng.random(n) < p_treated).astype(np.uint8)
    Y_c = (rng.random(n) < 0.10).astype(np.uint8)
    Y = np.where(T == 1, Y_t, Y_c).astype(np.uint8)
    return X, T, Y


def test_s_learner_smoke() -> None:
    X, T, Y = _toy()
    est = SLearner({"objective": "binary", "n_estimators": 50, "num_leaves": 15, "verbose": -1})
    est.fit(X, T, Y)
    cate = est.predict_cate(X)
    baseline = est.predict_baseline(X)
    assert cate.shape == (len(X),)
    assert baseline.shape == (len(X),)
    assert qini_coefficient(T, Y, cate) > 0.0


def test_t_learner_smoke() -> None:
    X, T, Y = _toy()
    est = TLearner({"objective": "binary", "n_estimators": 50, "num_leaves": 15, "verbose": -1})
    est.fit(X, T, Y)
    cate = est.predict_cate(X)
    baseline = est.predict_baseline(X)
    assert cate.shape == (len(X),)
    assert baseline.shape == (len(X),)
    assert qini_coefficient(T, Y, cate) > 0.0
