"""Streamlit demo: budget slider over a pre-scored sample.

Heavy ML libs are intentionally NOT imported here so the Streamlit Community Cloud
install stays under the 1 GB RAM cap. The app only does filtering and segmentation
math on a precomputed parquet produced by `scripts/score_sample.py`.

Run locally:   uv run streamlit run streamlit_app/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from upliftbench.config import (
    DOWHY_REFUTATION_JSON,
    LEADERBOARD_PARQUET,
    SCORED_SAMPLE_PARQUET,
)
from upliftbench.segmentation import (
    ALL_SEGMENTS,
    SEGMENT_PERSUADABLE,
    budget_allocation,
)

st.set_page_config(page_title="UpliftBench", layout="wide", page_icon=":bar_chart:")


@st.cache_data
def load_scored() -> pd.DataFrame:
    if not Path(SCORED_SAMPLE_PARQUET).exists():
        return pd.DataFrame()
    return pd.read_parquet(SCORED_SAMPLE_PARQUET)


@st.cache_data
def load_leaderboard() -> pd.DataFrame:
    if not Path(LEADERBOARD_PARQUET).exists():
        return pd.DataFrame()
    return pd.read_parquet(LEADERBOARD_PARQUET)


@st.cache_data
def load_refutation() -> dict | None:
    if not Path(DOWHY_REFUTATION_JSON).exists():
        return None
    return json.loads(Path(DOWHY_REFUTATION_JSON).read_text())


def estimator_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("pred_cate_") and c != "pred_cate_best"]


def main() -> None:
    st.title("UpliftBench: Treatment-Budget Explorer")
    st.caption("Criteo Uplift v2. Five estimators. Live persuadable segmentation.")

    scored = load_scored()
    leaderboard = load_leaderboard()
    refute = load_refutation()

    if scored.empty:
        st.warning(
            f"`{SCORED_SAMPLE_PARQUET.name}` not found. Run `make data && make prepare "
            "&& make train-all && make eval && make score` to populate artifacts/."
        )
        return

    estimator_cols = estimator_columns(scored)
    est_options = [c.removeprefix("pred_cate_").replace("_", "-") for c in estimator_cols]

    with st.sidebar:
        st.header("Controls")
        est_label = st.selectbox(
            "Score by estimator",
            options=est_options,
            index=0,
            help="Which model's CATE to rank by.",
        )
        budget_pct = st.slider(
            "Treatment budget (% of population)",
            min_value=0,
            max_value=100,
            value=20,
            step=1,
            help="Top-K by predicted uplift. Higher budgets reach more persuadables AND more do-not-disturb.",
        )

    cate_col = f"pred_cate_{est_label.replace('-', '_')}"
    df = scored[[*estimator_cols, "pred_baseline", "segment"]].copy()
    df["pred_cate"] = df[cate_col]

    allocation = budget_allocation(df, budget_pct=budget_pct / 100.0)

    col_a, col_b = st.columns([2, 1], gap="large")

    with col_a:
        st.subheader(f"Top {budget_pct}% targeted ({allocation['n_targeted']:,} users)")
        seg_df = pd.DataFrame(
            {
                "segment": list(ALL_SEGMENTS),
                "count": [allocation["counts"][s] for s in ALL_SEGMENTS],
            }
        )
        fig, ax = plt.subplots(figsize=(7, 3.5))
        colors = ["#2ecc71", "#3498db", "#7f8c8d", "#e74c3c"]
        ax.bar(seg_df["segment"], seg_df["count"], color=colors)
        ax.set_ylabel("count")
        ax.set_title("Segment composition of targeted population")
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig, clear_figure=True)

        st.markdown(
            f"**Persuadables in budget:** "
            f"`{allocation['counts'][SEGMENT_PERSUADABLE]:,}` of `{allocation['n_targeted']:,}`. "
            "Move the slider on the left to see how the segment mix shifts."
        )

    with col_b:
        st.subheader("Leaderboard")
        if leaderboard.empty:
            st.info("No leaderboard yet. Run `make eval`.")
        else:
            st.dataframe(
                leaderboard[["estimator", "qini", "auuc"]].round(4).reset_index(drop=True),
                hide_index=True,
            )

        st.subheader("DoWhy refutation")
        if refute is None:
            st.info("No refutation yet. Run `make dowhy`.")
        else:
            st.metric("Point estimate (ATE on visit)", f"{refute['estimate']:+.4f}")
            ref_rows = []
            for name, info in refute["refutations"].items():
                ref_rows.append(
                    {
                        "refuter": name,
                        "new_estimate": info.get("new_estimate", float("nan")),
                        "p_value": info.get("p_value", float("nan")),
                    }
                )
            st.dataframe(pd.DataFrame(ref_rows).round(4), hide_index=True)

    st.divider()
    st.subheader("Top targeted users (head)")
    if allocation["n_targeted"] > 0:
        top_idx = allocation["indices"]
        preview = (
            scored.iloc[top_idx]
            .head(50)[
                [
                    cate_col,
                    "pred_baseline",
                    "segment",
                    "treatment",
                    "visit",
                ]
            ]
            .rename(columns={cate_col: "pred_cate"})
        )
        st.dataframe(preview, hide_index=True)
    else:
        st.write("Budget is zero. Move the slider to target users.")

    with st.expander("How segments are defined"):
        st.markdown(
            "- **persuadable**: high predicted uplift, low baseline propensity. The model thinks "
            "treatment will convert them.\n"
            "- **sure_thing**: low predicted uplift, high baseline. Treating them wastes budget.\n"
            "- **lost_cause**: low uplift, low baseline. Treatment is unlikely to help.\n"
            "- **do_not_disturb**: high uplift sign but the model also thinks they would convert; "
            "in classical uplift literature this is the segment to avoid."
        )


if __name__ == "__main__":
    main()
