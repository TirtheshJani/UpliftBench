"""DoWhy four-step pipeline: model, identify, estimate, refute.

Runs all four standard refuters on a (stratified) sub-sample because the full
13.9M-row dataset is intractable for refutation on a laptop CPU. The stratification
is by treatment so both arms remain represented.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from dowhy import CausalModel

REFUTER_NAMES = (
    "placebo_treatment",
    "random_common_cause",
    "data_subset",
    "unobserved_common_cause",
)

_REFUTER_TO_METHOD: dict[str, str] = {
    "placebo_treatment": "placebo_treatment_refuter",
    "random_common_cause": "random_common_cause",
    "data_subset": "data_subset_refuter",
    "unobserved_common_cause": "add_unobserved_common_cause",
}


def _stratified_sample(
    df: pd.DataFrame, treatment_col: str, sample_n: int, seed: int
) -> pd.DataFrame:
    """Sample `sample_n` rows preserving treatment ratio."""
    if sample_n >= len(df):
        return df
    rng = np.random.default_rng(seed)
    treated = df.index[df[treatment_col] == 1].to_numpy()
    control = df.index[df[treatment_col] == 0].to_numpy()
    p_t = len(treated) / len(df)
    n_t = int(round(sample_n * p_t))
    n_c = sample_n - n_t
    pick_t = rng.choice(treated, size=min(n_t, len(treated)), replace=False)
    pick_c = rng.choice(control, size=min(n_c, len(control)), replace=False)
    idx = np.sort(np.concatenate([pick_t, pick_c]))
    return df.loc[idx].reset_index(drop=True)


def _refuter_result(refute_result: Any) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    out["new_estimate"] = float(getattr(refute_result, "new_effect", float("nan")))
    p = getattr(refute_result, "refutation_result", None)
    if isinstance(p, dict) and "p_value" in p:
        out["p_value"] = float(p["p_value"])
    else:
        p_attr = getattr(refute_result, "p_value", None)
        out["p_value"] = float(p_attr) if p_attr is not None else float("nan")
    out["test_name"] = type(refute_result).__name__
    return out


def run_dowhy(
    df: pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str = "treatment",
    outcome_col: str = "visit",
    sample_n: int | None = 1_000_000,
    seed: int = 42,
    estimator_method: str = "backdoor.linear_regression",
) -> dict[str, Any]:
    """Run DoWhy model -> identify -> estimate -> refute (all four refuters).

    Returns a dict with keys: estimand (str), estimate (float), refutations (dict).
    """
    if sample_n is not None and sample_n < len(df):
        df = _stratified_sample(df, treatment_col, sample_n, seed)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CausalModel(
            data=df,
            treatment=treatment_col,
            outcome=outcome_col,
            common_causes=list(feature_cols),
        )
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(estimand, method_name=estimator_method)
        point = float(estimate.value)

        refutations: dict[str, dict[str, float | str]] = {}
        for name in REFUTER_NAMES:
            try:
                ref = model.refute_estimate(
                    estimand, estimate, method_name=_REFUTER_TO_METHOD[name]
                )
                refutations[name] = _refuter_result(ref)
            except Exception as exc:  # noqa: BLE001
                refutations[name] = {
                    "new_estimate": float("nan"),
                    "p_value": float("nan"),
                    "test_name": "error",
                    "error": str(exc),
                }

    return {
        "estimand": str(estimand),
        "estimate": point,
        "refutations": refutations,
        "n": int(len(df)),
        "estimator_method": estimator_method,
    }
