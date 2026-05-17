"""Train a single uplift estimator on the prepared Criteo parquet.

Usage:
    uv run python scripts/train.py --estimator s-learner
    uv run python scripts/train.py --estimator s-learner --sample 100_000
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import psutil
import pyarrow.parquet as pq
import typer

from upliftbench.config import (
    CRITEO_PARQUET,
    FEATURES,
    MODELS_DIR,
    OUTCOME_VISIT,
    SEED,
    TEST_FRAC,
    TEST_SPLIT_JSON,
    TREATMENT_COL,
)
from upliftbench.data.loader import train_test_split_rct
from upliftbench.estimators import get_estimator
from upliftbench.eval.harness import evaluate_estimator
from upliftbench.persistence import save_model

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _load_arrays(
    parquet_path: Path,
    sample: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols = [*FEATURES, TREATMENT_COL, OUTCOME_VISIT]
    table = pq.read_table(str(parquet_path), columns=cols)
    n_full = table.num_rows
    if sample is not None and sample < n_full:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n_full, size=sample, replace=False))
        table = table.take(idx)
    df = table.to_pandas()
    X = df[FEATURES].to_numpy(dtype=np.float32, copy=False)
    T = df[TREATMENT_COL].to_numpy(dtype=np.uint8, copy=False)
    Y = df[OUTCOME_VISIT].to_numpy(dtype=np.uint8, copy=False)
    train_idx, test_idx = train_test_split_rct(n=len(df), test_frac=TEST_FRAC, seed=seed)
    return X, T, Y, train_idx, test_idx


def _record_test_split(test_idx: np.ndarray, n_total: int, seed: int) -> str:
    """Hash the held-out test row IDs so every estimator references the same split.

    The digest is also written to each model's sibling metadata JSON. `evaluate_all`
    then refuses to mix estimators with different hashes onto the same leaderboard,
    which catches the "trained S on 13.9M, trained T on a 5M sample" foot-gun.
    """
    digest = hashlib.sha256(test_idx.tobytes()).hexdigest()[:16]
    if not TEST_SPLIT_JSON.exists():
        TEST_SPLIT_JSON.parent.mkdir(parents=True, exist_ok=True)
        TEST_SPLIT_JSON.write_text(
            json.dumps({"n_total": int(n_total), "seed": int(seed), "test_hash": digest}, indent=2)
        )
    return digest


@app.command()
def main(
    estimator: Annotated[str, typer.Option(help="Registry key: s-learner, t-learner, ...")],
    parquet_path: Annotated[Path, typer.Option(help="Input parquet.")] = CRITEO_PARQUET,
    sample: Annotated[int | None, typer.Option(help="Optional row subsample.")] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = SEED,
    out_dir: Annotated[Path, typer.Option(help="Output directory.")] = MODELS_DIR,
) -> None:
    if not parquet_path.exists():
        typer.echo(f"Missing parquet: {parquet_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading {parquet_path}...")
    X, T, Y, train_idx, test_idx = _load_arrays(parquet_path, sample, seed)
    split_hash = _record_test_split(test_idx, n_total=len(X), seed=seed)
    typer.echo(f"  N total={len(X):,}  N train={len(train_idx):,}  N test={len(test_idx):,}")
    typer.echo(f"  Test-split sha-16: {split_hash}")

    est = get_estimator(estimator)
    proc = psutil.Process()
    rss_before = proc.memory_info().rss / 1e9
    t0 = time.time()
    typer.echo(f"Fitting {estimator}...")
    est.fit(X[train_idx], T[train_idx], Y[train_idx])
    runtime_s = time.time() - t0
    rss_peak = proc.memory_info().rss / 1e9
    typer.echo(
        f"  fit: {runtime_s:.1f}s  peak RSS: {rss_peak:.2f} GB (delta {rss_peak - rss_before:+.2f})"
    )

    typer.echo("Scoring test split...")
    cate_test = est.predict_cate(X[test_idx])
    metrics = evaluate_estimator(T[test_idx], Y[test_idx], cate_test)
    qini = float(metrics["qini_coef"])
    au = float(metrics["auuc"])
    typer.echo(f"  Qini={qini:+.4f}  AUUC={au:+.4f}")

    paths = save_model(
        est,
        name=estimator,
        out_dir=out_dir,
        metadata={
            "runtime_s": runtime_s,
            "rss_peak_gb": rss_peak,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "qini": qini,
            "auuc": au,
            "top_k_uplift": metrics["top_k_uplift"],
            "seed": seed,
            "sample": sample,
            "test_split_hash": split_hash,
        },
    )
    typer.echo(f"Saved: {paths.model}")
    typer.echo(f"Meta:  {paths.metadata}")


if __name__ == "__main__":
    app()
