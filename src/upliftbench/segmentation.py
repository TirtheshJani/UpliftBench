"""Shared scoring + four-way customer segmentation.

This is the single source of truth imported by `scripts/score_sample.py`, the Kaggle
notebook, and `streamlit_app/app.py`. The Streamlit app imports ONLY this module from
upliftbench (no estimator imports, no LightGBM/DoWhy/CausalML/EconML) so that the
Streamlit Community Cloud install stays under its memory cap.

Segment rules (per pred_cate and pred_baseline):

| pred_cate | pred_baseline | segment            | meaning                                     |
|-----------|---------------|--------------------|---------------------------------------------|
| > thr_c   | < thr_b       | persuadable        | treatment moves them from non-buyer to buyer |
| <= thr_c  | >= thr_b      | sure_thing         | buys anyway, treatment redundant            |
| <= thr_c  | < thr_b       | lost_cause         | will not buy with or without treatment      |
| > thr_c   | >= thr_b      | do_not_disturb     | would buy, treatment annoys (negative ROI)   |
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import pandas as pd

from upliftbench.config import BASELINE_THRESHOLD, CATE_THRESHOLD

SEGMENT_PERSUADABLE = "persuadable"
SEGMENT_SURE_THING = "sure_thing"
SEGMENT_LOST_CAUSE = "lost_cause"
SEGMENT_DO_NOT_DISTURB = "do_not_disturb"
ALL_SEGMENTS = (
    SEGMENT_PERSUADABLE,
    SEGMENT_SURE_THING,
    SEGMENT_LOST_CAUSE,
    SEGMENT_DO_NOT_DISTURB,
)


class _Scorer(Protocol):
    def predict_cate(self, X: np.ndarray) -> np.ndarray: ...
    def predict_baseline(self, X: np.ndarray) -> np.ndarray: ...


def classify_segments(
    cate: np.ndarray,
    baseline: np.ndarray,
    cate_threshold: float = CATE_THRESHOLD,
    baseline_threshold: float = BASELINE_THRESHOLD,
) -> np.ndarray:
    """Vectorized four-way segmentation. Returns a numpy array of segment labels."""
    cate = np.asarray(cate)
    baseline = np.asarray(baseline)
    if cate.shape != baseline.shape:
        raise ValueError(f"cate {cate.shape} != baseline {baseline.shape}")
    seg = np.empty(cate.shape, dtype=object)
    high_cate = cate > cate_threshold
    high_base = baseline >= baseline_threshold
    seg[high_cate & ~high_base] = SEGMENT_PERSUADABLE
    seg[~high_cate & high_base] = SEGMENT_SURE_THING
    seg[~high_cate & ~high_base] = SEGMENT_LOST_CAUSE
    seg[high_cate & high_base] = SEGMENT_DO_NOT_DISTURB
    return seg


def score_and_segment(
    estimator: _Scorer,
    X: np.ndarray,
    cate_threshold: float = CATE_THRESHOLD,
    baseline_threshold: float = BASELINE_THRESHOLD,
) -> pd.DataFrame:
    """Score `X` with `estimator` and return a DataFrame with pred_cate/baseline/segment."""
    cate = np.asarray(estimator.predict_cate(X))
    baseline = np.asarray(estimator.predict_baseline(X))
    segment = classify_segments(cate, baseline, cate_threshold, baseline_threshold)
    return pd.DataFrame({"pred_cate": cate, "pred_baseline": baseline, "segment": segment})


def budget_allocation(
    scored_df: pd.DataFrame,
    budget_pct: float,
) -> dict[str, Any]:
    """Sort by pred_cate desc, take top `budget_pct * N` rows, return segment counts.

    `scored_df` must have columns `pred_cate` and `segment`.
    """
    if not 0.0 <= budget_pct <= 1.0:
        raise ValueError(f"budget_pct must be in [0, 1], got {budget_pct}")
    n = len(scored_df)
    k = int(round(n * budget_pct))
    if k == 0:
        counts = {s: 0 for s in ALL_SEGMENTS}
        return {"n_targeted": 0, "counts": counts, "indices": np.array([], dtype=np.int64)}
    top_idx = np.argpartition(-scored_df["pred_cate"].to_numpy(), kth=min(k - 1, n - 1))[:k]
    # Sort the selected top-k by pred_cate desc for stable downstream display.
    top_sorted = top_idx[np.argsort(-scored_df["pred_cate"].to_numpy()[top_idx], kind="mergesort")]
    seg_arr = scored_df["segment"].to_numpy()[top_sorted]
    counts = {s: int((seg_arr == s).sum()) for s in ALL_SEGMENTS}
    return {"n_targeted": int(k), "counts": counts, "indices": top_sorted}
