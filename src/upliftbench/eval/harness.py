"""evaluate_estimator: single entry point that returns all scalar + curve metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from upliftbench.eval.auuc import auuc, uplift_curve
from upliftbench.eval.qini import qini_coefficient, qini_curve
from upliftbench.eval.topk import top_k_uplift


def evaluate_estimator(
    t: np.ndarray,
    y: np.ndarray,
    cate: np.ndarray,
    top_k_fracs: tuple[float, ...] = (0.1, 0.2, 0.3),
) -> dict[str, Any]:
    """Return a dict of Qini, AUUC, top-K uplift, and the (x, y) curve points.

    Curve points are returned as `(xs, ys)` tuples so they can be pickled, plotted,
    or persisted to parquet/json without recomputing.
    """
    return {
        "qini_coef": qini_coefficient(t, y, cate),
        "auuc": auuc(t, y, cate),
        "top_k_uplift": {
            f"top_{int(round(k * 100))}": top_k_uplift(t, y, cate, k) for k in top_k_fracs
        },
        "qini_curve_xy": qini_curve(t, y, cate),
        "uplift_curve_xy": uplift_curve(t, y, cate),
    }
