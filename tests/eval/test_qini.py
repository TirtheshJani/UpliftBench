"""TDD for src/upliftbench/eval/qini.py."""

from __future__ import annotations

import numpy as np
import pytest

from upliftbench.eval.qini import qini_coefficient, qini_curve


def _synthetic_uplift(
    n: int = 10_000,
    seed: int = 0,
    p0: float = 0.10,
    treatment_rate: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Toy data with a known per-row uplift signal.

    `u_i = sigmoid(x_i)`. Treated outcome = Bernoulli(p0 + u_i / 4); control = Bernoulli(p0).
    The true per-row uplift is u_i / 4. A perfect ranker sorting by `x_i` achieves the
    optimal Qini curve.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    u = 1.0 / (1.0 + np.exp(-x))
    t = (rng.random(n) < treatment_rate).astype(np.uint8)
    p_treated = np.clip(p0 + u / 4.0, 0.0, 1.0)
    y_treated = (rng.random(n) < p_treated).astype(np.uint8)
    y_control = (rng.random(n) < p0).astype(np.uint8)
    y = np.where(t == 1, y_treated, y_control).astype(np.uint8)
    return t, y, x  # `x` itself is a near-perfect score


def test_qini_curve_shape_and_endpoints() -> None:
    t, y, score = _synthetic_uplift(n=2_000, seed=0)
    xs, ys = qini_curve(t, y, score)
    assert xs[0] == 0.0
    assert ys[0] == 0.0
    assert xs[-1] == pytest.approx(1.0, abs=1e-9)
    # At 100% targeted, the lift equals the overall (treated_rate - control_rate) * n
    treated_resp = float(y[t == 1].sum())
    control_resp = float(y[t == 0].sum())
    n_t = int((t == 1).sum())
    n_c = int((t == 0).sum())
    expected_final = treated_resp - control_resp * (n_t / n_c)
    assert ys[-1] == pytest.approx(expected_final, abs=1e-6)


def test_qini_perfect_ranker_beats_random() -> None:
    t, y, score = _synthetic_uplift(n=20_000, seed=1)
    rng = np.random.default_rng(99)
    random_score = rng.standard_normal(len(t))
    perfect = qini_coefficient(t, y, score)
    random_q = qini_coefficient(t, y, random_score)
    assert perfect > random_q
    assert abs(random_q) < 0.05  # random ranker should be near zero
    assert perfect > 0.05  # the synthetic signal is strong


def test_qini_random_ranker_is_near_zero() -> None:
    t, y, _ = _synthetic_uplift(n=20_000, seed=2)
    rng = np.random.default_rng(3)
    coefs = [qini_coefficient(t, y, rng.standard_normal(len(t))) for _ in range(8)]
    mean_coef = float(np.mean(coefs))
    assert abs(mean_coef) < 0.03


def test_qini_input_validation() -> None:
    with pytest.raises(ValueError):
        qini_coefficient(np.array([0, 1]), np.array([0, 1, 0]), np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError):
        qini_coefficient(np.array([0, 1, 2]), np.array([0, 1, 0]), np.array([0.1, 0.2, 0.3]))
