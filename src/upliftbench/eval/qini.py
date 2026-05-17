"""Qini curve and Qini coefficient.

Definitions follow Radcliffe (2007). For a population sorted in descending order by
predicted CATE, the Qini curve at population fraction k is:

    Q(k) = Y_t(k) - Y_c(k) * (N_t(k) / N_c(k))

where Y_t(k), Y_c(k) are cumulative positive outcomes among treated/control in the
top k fraction, and N_t(k), N_c(k) are the cumulative counts.

The Qini coefficient is the normalized area between the model curve and the random
(diagonal) curve, divided by the area between the optimal and random curves.
"""

from __future__ import annotations

import numpy as np


def _validate(t: np.ndarray, y: np.ndarray, cate: np.ndarray) -> None:
    if not (len(t) == len(y) == len(cate)):
        raise ValueError("t, y, cate must have the same length")
    if t.ndim != 1 or y.ndim != 1 or cate.ndim != 1:
        raise ValueError("t, y, cate must be 1-D")
    if set(np.unique(t)).difference({0, 1}):
        raise ValueError("t must contain only 0/1 values")


def qini_curve(
    t: np.ndarray,
    y: np.ndarray,
    cate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (population_fractions, qini_values) for the predicted ranking."""
    _validate(t, y, cate)
    n = len(t)
    # Stable sort matters for reproducibility when many CATEs tie (e.g. at 0).
    order = np.argsort(-cate, kind="mergesort")
    t_sorted = t[order].astype(np.float64)
    y_sorted = y[order].astype(np.float64)

    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1.0 - t_sorted)
    cum_yt = np.cumsum(y_sorted * t_sorted)
    cum_yc = np.cumsum(y_sorted * (1.0 - t_sorted))

    # The ratio cum_t / cum_c rescales the control-side response to the treated
    # count, which is what makes the curve interpretable as "incremental positives".
    # Early prefixes can have zero controls; the guard avoids 0/0 NaN.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cum_c > 0, cum_t / cum_c, 0.0)
    lift = cum_yt - cum_yc * ratio

    # Prepend (0, 0) so the curve passes through the origin; downstream area
    # integration treats this as the boundary point.
    xs = np.concatenate(([0.0], np.arange(1, n + 1) / n))
    ys = np.concatenate(([0.0], lift))
    return xs, ys


def _qini_area(xs: np.ndarray, ys: np.ndarray) -> float:
    return float(np.trapz(ys, xs))


def qini_coefficient(t: np.ndarray, y: np.ndarray, cate: np.ndarray) -> float:
    """Qini coefficient scaled so a random ranker is near 0 and a perfect one is near 1.

    Definition: `2 * (area_model - area_random) / |Q_total|`, where
    `area_random = Q_total / 2` is the triangle under the diagonal from (0,0) to
    (1, Q_total). When `|Q_total|` is effectively zero (no incremental lift in the
    population), the coefficient is defined as 0.
    """
    _validate(t, y, cate)
    xs, ys = qini_curve(t, y, cate)
    q_total = ys[-1]
    # No incremental lift anywhere in the population means the coefficient is
    # mechanically undefined; return 0 rather than NaN so callers can sort/plot.
    if abs(q_total) < 1e-12:
        return 0.0
    area_model = _qini_area(xs, ys)
    area_random = q_total / 2.0
    # Normalizing by |q_total| keeps the coefficient comparable across populations
    # with different treatment rates and baseline outcome rates.
    return float(2.0 * (area_model - area_random) / abs(q_total))
