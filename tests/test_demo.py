"""TDD for src/upliftbench/demo.py (no-download synthetic RCT demo)."""

from __future__ import annotations

import numpy as np

from upliftbench.config import FEATURES
from upliftbench.demo import make_synthetic_rct, run_demo


def test_make_synthetic_rct_shapes_and_known_effect() -> None:
    X, T, Y, tau = make_synthetic_rct(n=4_000, seed=0)
    assert X.shape == (4_000, len(FEATURES))
    assert X.dtype == np.float32
    assert T.dtype == np.uint8 and Y.dtype == np.uint8
    assert set(np.unique(T)) == {0, 1}
    assert set(np.unique(Y)) <= {0, 1}
    assert tau.shape == (4_000,)
    # Heterogeneous effect with both signs, so all four segments are reachable.
    assert tau.min() < 0.0 < tau.max()


def test_make_synthetic_rct_is_seeded() -> None:
    a = make_synthetic_rct(n=500, seed=7)
    b = make_synthetic_rct(n=500, seed=7)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_run_demo_leaderboard_has_oracle_random_and_learner() -> None:
    lb = run_demo(n=6_000, seed=0, estimators=("s-learner",))
    assert list(lb["estimator"]) == ["oracle (true CATE)", "s-learner", "random"]
    assert {"qini_coef", "auuc", "top_10_uplift", "fit_seconds"} <= set(lb.columns)
    by_name = lb.set_index("estimator")
    # The oracle ranks by the true effect, so it must beat a random ranking,
    # and a real learner must recover a positive ranking signal.
    assert by_name.loc["oracle (true CATE)", "qini_coef"] > by_name.loc["random", "qini_coef"]
    assert by_name.loc["s-learner", "qini_coef"] > 0.0
