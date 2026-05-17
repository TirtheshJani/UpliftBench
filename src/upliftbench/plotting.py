"""Reusable plot helpers (matplotlib only).

These functions return a `Figure` so the caller can either show it or write it
to disk. Keeping plotting out of the eval and segmentation modules lets the
Streamlit app import segmentation without pulling matplotlib in via a heavier
path than necessary.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from upliftbench.segmentation import ALL_SEGMENTS


def qini_plot(curves: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> plt.Figure:
    """`curves` maps estimator name to (xs, ys) as returned by `qini_curve`."""
    fig, ax = plt.subplots(figsize=(7, 5))
    q_total = 0.0
    for name, (xs, ys) in curves.items():
        ax.plot(xs, ys, label=name)
        q_total = float(ys[-1])
    ax.plot([0, 1], [0, q_total], color="gray", linestyle="--", label="random")
    ax.set_xlabel("Population fraction")
    ax.set_ylabel("Cumulative uplift")
    ax.set_title("Qini curves")
    ax.legend()
    return fig


def auuc_bar(leaderboard: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    lb = leaderboard.sort_values("auuc", ascending=False)
    ax.bar(lb["estimator"], lb["auuc"], color="steelblue")
    ax.set_ylabel("AUUC")
    ax.set_title("AUUC per estimator")
    ax.tick_params(axis="x", rotation=20)
    return fig


def segment_bar(counts: Mapping[str, int]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2ecc71", "#3498db", "#7f8c8d", "#e74c3c"]
    values = [int(counts.get(s, 0)) for s in ALL_SEGMENTS]
    ax.bar(list(ALL_SEGMENTS), values, color=colors)
    ax.set_ylabel("count")
    ax.set_title("Segment composition")
    ax.tick_params(axis="x", rotation=15)
    return fig
