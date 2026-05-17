"""TDD for src/upliftbench/refute/dowhy_pipeline.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from upliftbench.refute.dowhy_pipeline import REFUTER_NAMES, run_dowhy


def _toy_rct(n: int = 2_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4)).astype(np.float32)
    T = (rng.random(n) < 0.5).astype(np.uint8)
    base = 0.1 + 0.05 * X[:, 0]
    p = np.clip(base + 0.08 * T, 0, 1)
    Y = (rng.random(n) < p).astype(np.uint8)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    df["treatment"] = T
    df["visit"] = Y
    return df


def test_run_dowhy_returns_expected_structure() -> None:
    df = _toy_rct(n=1_500)
    out = run_dowhy(
        df,
        feature_cols=[f"f{i}" for i in range(4)],
        treatment_col="treatment",
        outcome_col="visit",
        sample_n=None,
        seed=42,
    )
    assert "estimand" in out
    assert "estimate" in out
    assert isinstance(out["estimate"], float)
    assert "refutations" in out
    for name in REFUTER_NAMES:
        assert name in out["refutations"], f"missing refuter {name}"
        entry = out["refutations"][name]
        assert "new_estimate" in entry
        assert "p_value" in entry


def test_refuter_names_includes_all_four() -> None:
    assert set(REFUTER_NAMES) == {
        "placebo_treatment",
        "random_common_cause",
        "data_subset",
        "unobserved_common_cause",
    }
