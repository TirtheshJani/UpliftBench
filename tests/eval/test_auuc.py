"""TDD for src/upliftbench/eval/auuc.py."""

from __future__ import annotations

import numpy as np
import pytest

from upliftbench.eval.auuc import auuc, uplift_curve


def test_uplift_curve_endpoints() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    t = (rng.random(n) < 0.5).astype(np.uint8)
    y = (rng.random(n) < 0.1).astype(np.uint8)
    cate = rng.standard_normal(n)
    xs, ys = uplift_curve(t, y, cate)
    assert xs[0] == 0.0
    assert ys[0] == 0.0
    assert xs[-1] == pytest.approx(1.0, abs=1e-9)


def test_auuc_random_near_zero_perfect_positive() -> None:
    rng = np.random.default_rng(1)
    n = 20_000
    x = rng.standard_normal(n).astype(np.float32)
    u = 1.0 / (1.0 + np.exp(-x))
    t = (rng.random(n) < 0.5).astype(np.uint8)
    p_treated = np.clip(0.10 + u / 4.0, 0.0, 1.0)
    y_treated = (rng.random(n) < p_treated).astype(np.uint8)
    y_control = (rng.random(n) < 0.10).astype(np.uint8)
    y = np.where(t == 1, y_treated, y_control).astype(np.uint8)

    perfect = auuc(t, y, x)
    rng2 = np.random.default_rng(99)
    random_scores = [rng2.standard_normal(n) for _ in range(8)]
    random_coefs = [auuc(t, y, s) for s in random_scores]
    assert perfect > 0.05
    assert abs(float(np.mean(random_coefs))) < 0.03


def test_auuc_input_validation() -> None:
    with pytest.raises(ValueError):
        auuc(np.array([0, 1]), np.array([0, 1, 0]), np.array([0.1, 0.2, 0.3]))
