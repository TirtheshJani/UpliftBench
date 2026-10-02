"""No-download demo: fit uplift estimators on a synthetic RCT and score them.

The real benchmark needs the ~700 MB Criteo Uplift v2 download. This module lets a
visitor see the same estimators and the same Qini / AUUC / top-K code run end to end
in well under a minute, on a synthetic randomized experiment where the true per-row
treatment effect is known. Because the truth is known, the leaderboard includes an
"oracle" row (ranking by the true CATE) and a "random" row as reference points.

Numbers printed by this demo describe the synthetic data only. They are not Criteo
results.

Usage:
    uv run python -m upliftbench.demo
    uv run python -m upliftbench.demo --n 50000 --estimators s-learner,x-learner
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence

import numpy as np
import pandas as pd

from upliftbench.config import FEATURES, SEED, TEST_FRAC

DEFAULT_ESTIMATORS = ("s-learner", "t-learner", "x-learner", "dr-learner", "dml")


def make_synthetic_rct(
    n: int = 20_000,
    seed: int = SEED,
    treatment_rate: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic RCT shaped like Criteo (12 float32 features, binary T and Y).

    Baseline response: `p0(x) = sigmoid(-1.5 + 0.8 * f1)`.
    True uplift: `tau(x) = 0.05 + 0.10 * tanh(f0)`, mostly positive but negative for
    f0 below about -0.55, so rows that treatment helps and rows it hurts both exist.
    Treatment is randomized independently of X (Bernoulli(treatment_rate)).

    Returns `(X, T, Y, tau)` where `tau` is the true per-row effect on P(Y=1).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(FEATURES))).astype(np.float32)
    p0 = 1.0 / (1.0 + np.exp(-(-1.5 + 0.8 * X[:, 1])))
    tau = 0.05 + 0.10 * np.tanh(X[:, 0].astype(np.float64))
    p1 = np.clip(p0 + tau, 0.0, 1.0)
    T = (rng.random(n) < treatment_rate).astype(np.uint8)
    u = rng.random(n)
    Y = np.where(T == 1, u < p1, u < p0).astype(np.uint8)
    return X, T, Y, p1 - p0  # effective effect after clipping to [0, 1]


def _row(name: str, t: np.ndarray, y: np.ndarray, cate: np.ndarray, secs: float) -> dict:
    from upliftbench.eval.harness import evaluate_estimator

    m = evaluate_estimator(t, y, cate, top_k_fracs=(0.1,))
    return {
        "estimator": name,
        "qini_coef": m["qini_coef"],
        "auuc": m["auuc"],
        "top_10_uplift": m["top_k_uplift"]["top_10"],
        "fit_seconds": secs,
    }


def run_demo(
    n: int = 20_000,
    seed: int = SEED,
    estimators: Sequence[str] = DEFAULT_ESTIMATORS,
) -> pd.DataFrame:
    """Fit each estimator on a train split and evaluate on a held-out split.

    Returns a leaderboard DataFrame: the oracle row first, then estimators sorted by
    Qini coefficient, then the random baseline last.
    """
    from upliftbench.estimators import get_estimator

    X, T, Y, tau = make_synthetic_rct(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    is_test = rng.random(n) < TEST_FRAC
    Xtr, Ttr, Ytr = X[~is_test], T[~is_test], Y[~is_test]
    Xte, Tte, Yte = X[is_test], T[is_test], Y[is_test]

    rows = []
    for name in estimators:
        est = get_estimator(name)
        start = time.perf_counter()
        est.fit(Xtr, Ttr, Ytr)
        secs = time.perf_counter() - start
        rows.append(_row(name, Tte, Yte, est.predict_cate(Xte), secs))
    rows.sort(key=lambda r: r["qini_coef"], reverse=True)

    oracle = _row("oracle (true CATE)", Tte, Yte, tau[is_test], 0.0)
    random_scores = rng.random(int(is_test.sum()))
    rand = _row("random", Tte, Yte, random_scores, 0.0)
    return pd.DataFrame([oracle, *rows, rand])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=20_000, help="synthetic rows (default 20000)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--estimators",
        default=",".join(DEFAULT_ESTIMATORS),
        help="comma-separated registry names (default: all five)",
    )
    args = parser.parse_args(argv)
    names = [s.strip() for s in args.estimators.split(",") if s.strip()]

    print(f"Synthetic RCT: n={args.n}, seed={args.seed}, test_frac={TEST_FRAC}")
    print("Fitting:", ", ".join(names))
    lb = run_demo(n=args.n, seed=args.seed, estimators=names)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 120):
        print(lb.to_string(index=False))
    print(
        "\nQini/AUUC use 2 * (area_model - area_random) / |total lift|: random is near 0,"
        "\nthe oracle is the expected ceiling, and the random row's distance from 0 is the"
        "\nnoise floor at this test size. The true effect is close to linear in f0, which"
        "\nfavors LinearDML, so do not read this ranking as general. DR and DML cross-fitting"
        "\nis unseeded, so those rows vary between runs. Synthetic numbers, not Criteo results."
    )


if __name__ == "__main__":
    main()
