"""TDD for src/upliftbench/segmentation.py and src/upliftbench/eval/harness.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from upliftbench.eval.harness import evaluate_estimator
from upliftbench.segmentation import (
    SEGMENT_DO_NOT_DISTURB,
    SEGMENT_LOST_CAUSE,
    SEGMENT_PERSUADABLE,
    SEGMENT_SURE_THING,
    budget_allocation,
    classify_segments,
)


def test_classify_segments_disjoint_and_complete() -> None:
    n = 1000
    rng = np.random.default_rng(0)
    cate = rng.standard_normal(n)
    baseline = rng.random(n)
    seg = classify_segments(cate, baseline, cate_threshold=0.0, baseline_threshold=0.5)
    assert len(seg) == n
    cats = set(np.unique(seg))
    assert cats.issubset(
        {SEGMENT_PERSUADABLE, SEGMENT_SURE_THING, SEGMENT_LOST_CAUSE, SEGMENT_DO_NOT_DISTURB}
    )
    # No row has a missing label
    assert all(s != "" for s in seg)


def test_classify_segments_rules() -> None:
    cate = np.array([0.1, -0.1, -0.1, 0.1])
    baseline = np.array([0.2, 0.8, 0.2, 0.8])
    seg = classify_segments(cate, baseline, cate_threshold=0.0, baseline_threshold=0.5)
    assert seg[0] == SEGMENT_PERSUADABLE
    assert seg[1] == SEGMENT_SURE_THING
    assert seg[2] == SEGMENT_LOST_CAUSE
    assert seg[3] == SEGMENT_DO_NOT_DISTURB


def test_budget_allocation_top_k() -> None:
    df = pd.DataFrame(
        {
            "pred_cate": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4],
            "segment": [SEGMENT_PERSUADABLE] * 5 + [SEGMENT_DO_NOT_DISTURB] * 5,
        }
    )
    out = budget_allocation(df, budget_pct=0.30)
    assert out["n_targeted"] == 3
    assert out["counts"][SEGMENT_PERSUADABLE] == 3
    assert out["counts"][SEGMENT_DO_NOT_DISTURB] == 0


def test_budget_allocation_zero_and_full() -> None:
    df = pd.DataFrame(
        {
            "pred_cate": [0.5, 0.1, -0.1],
            "segment": [SEGMENT_PERSUADABLE, SEGMENT_PERSUADABLE, SEGMENT_LOST_CAUSE],
        }
    )
    assert budget_allocation(df, 0.0)["n_targeted"] == 0
    full = budget_allocation(df, 1.0)
    assert full["n_targeted"] == 3
    assert full["counts"][SEGMENT_PERSUADABLE] == 2
    assert full["counts"][SEGMENT_LOST_CAUSE] == 1


def test_budget_allocation_rejects_out_of_range() -> None:
    df = pd.DataFrame({"pred_cate": [0.0], "segment": [SEGMENT_PERSUADABLE]})
    with pytest.raises(ValueError):
        budget_allocation(df, -0.1)
    with pytest.raises(ValueError):
        budget_allocation(df, 1.5)


def test_evaluate_estimator_returns_expected_keys() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    t = (rng.random(n) < 0.5).astype(np.uint8)
    y = (rng.random(n) < 0.1).astype(np.uint8)
    cate = rng.standard_normal(n)
    out = evaluate_estimator(t, y, cate)
    for key in ("qini_coef", "auuc", "top_k_uplift", "qini_curve_xy", "uplift_curve_xy"):
        assert key in out, key
    assert isinstance(out["qini_coef"], float)
    assert isinstance(out["auuc"], float)
    xs, ys = out["qini_curve_xy"]
    assert len(xs) == len(ys) == n + 1
