"""Area Under the Uplift Curve.

The uplift curve plots the cumulative incremental response when targeting the top-k
population fraction (sorted by predicted CATE descending):

    U(k) = (Y_t(k) / N_t(k) - Y_c(k) / N_c(k)) * k_count

where k_count is the absolute count of items in the top-k.

AUUC is the area under this curve, with the same `2 * (area_model - area_random) /
|U_total|` normalization so random is near 0 and a strong ranker is positive.
"""

from __future__ import annotations

import numpy as np


def _validate(t: np.ndarray, y: np.ndarray, cate: np.ndarray) -> None:
    if not (len(t) == len(y) == len(cate)):
        raise ValueError("t, y, cate must have the same length")
    if t.ndim != 1 or y.ndim != 1 or cate.ndim != 1:
        raise ValueError("t, y, cate must be 1-D")


def uplift_curve(
    t: np.ndarray,
    y: np.ndarray,
    cate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (population_fractions, cumulative_uplift)."""
    _validate(t, y, cate)
    n = len(t)
    order = np.argsort(-cate, kind="mergesort")
    t_sorted = t[order].astype(np.float64)
    y_sorted = y[order].astype(np.float64)

    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1.0 - t_sorted)
    cum_yt = np.cumsum(y_sorted * t_sorted)
    cum_yc = np.cumsum(y_sorted * (1.0 - t_sorted))
    k = np.arange(1, n + 1, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        rate_t = np.where(cum_t > 0, cum_yt / cum_t, 0.0)
        rate_c = np.where(cum_c > 0, cum_yc / cum_c, 0.0)
    lift = (rate_t - rate_c) * k

    xs = np.concatenate(([0.0], k / n))
    ys = np.concatenate(([0.0], lift))
    return xs, ys


def auuc(t: np.ndarray, y: np.ndarray, cate: np.ndarray) -> float:
    """Normalized AUUC: 2 * (area_model - area_random) / |U_total|."""
    _validate(t, y, cate)
    xs, ys = uplift_curve(t, y, cate)
    u_total = ys[-1]
    if abs(u_total) < 1e-12:
        return 0.0
    area_model = float(np.trapz(ys, xs))
    area_random = u_total / 2.0
    return float(2.0 * (area_model - area_random) / abs(u_total))
