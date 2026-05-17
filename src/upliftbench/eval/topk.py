"""Top-K uplift: average treatment effect among the top-k fraction by predicted CATE."""

from __future__ import annotations

import numpy as np


def top_k_uplift(t: np.ndarray, y: np.ndarray, cate: np.ndarray, k_frac: float = 0.1) -> float:
    """Empirical ATE within the top `k_frac` of the population ranked by `cate` descending."""
    if not 0.0 < k_frac <= 1.0:
        raise ValueError(f"k_frac must be in (0, 1], got {k_frac}")
    n = len(t)
    k = max(1, int(round(n * k_frac)))
    order = np.argsort(-cate, kind="mergesort")[:k]
    t_top = t[order]
    y_top = y[order]
    n_t = int((t_top == 1).sum())
    n_c = int((t_top == 0).sum())
    if n_t == 0 or n_c == 0:
        return 0.0
    return float(y_top[t_top == 1].mean() - y_top[t_top == 0].mean())
