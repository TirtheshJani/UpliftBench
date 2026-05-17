"""Build a long-form leaderboard from every model in artifacts/models/.

Re-scores each estimator on the held-out test split (derived from the same seed used
during training) so all rows are directly comparable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import typer

from upliftbench.config import (
    ARTIFACTS_DIR,
    BEST_ESTIMATOR_TXT,
    CRITEO_PARQUET,
    FEATURES,
    LEADERBOARD_PARQUET,
    MODELS_DIR,
    OUTCOME_VISIT,
    SEED,
    TEST_FRAC,
    TREATMENT_COL,
)
from upliftbench.data.loader import train_test_split_rct
from upliftbench.eval.harness import evaluate_estimator
from upliftbench.persistence import load_model
from upliftbench.segmentation import classify_segments

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _load_test_data(parquet_path: Path, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols = [*FEATURES, TREATMENT_COL, OUTCOME_VISIT]
    df = pq.read_table(str(parquet_path), columns=cols).to_pandas()
    _, test_idx = train_test_split_rct(n=len(df), test_frac=TEST_FRAC, seed=seed)
    X = df[FEATURES].to_numpy(dtype=np.float32, copy=False)[test_idx]
    T = df[TREATMENT_COL].to_numpy(dtype=np.uint8, copy=False)[test_idx]
    Y = df[OUTCOME_VISIT].to_numpy(dtype=np.uint8, copy=False)[test_idx]
    return X, T, Y


@app.command()
def main(
    models_dir: Annotated[Path, typer.Option(help="Trained model directory.")] = MODELS_DIR,
    parquet_path: Annotated[Path, typer.Option(help="Held-out parquet.")] = CRITEO_PARQUET,
    out: Annotated[Path, typer.Option(help="Output leaderboard parquet.")] = LEADERBOARD_PARQUET,
    seed: Annotated[int, typer.Option(help="Same seed used in training.")] = SEED,
) -> None:
    if not parquet_path.exists():
        typer.echo(f"Missing parquet: {parquet_path}", err=True)
        raise typer.Exit(1)
    model_files = sorted(models_dir.glob("*.joblib"))
    if not model_files:
        typer.echo(f"No models found in {models_dir}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading test split from {parquet_path}...")
    X, T, Y = _load_test_data(parquet_path, seed=seed)
    typer.echo(f"  N test: {len(X):,}")

    rows = []
    for mpath in model_files:
        meta_path = mpath.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        typer.echo(f"Scoring {mpath.name}...")
        est = load_model(mpath)
        cate = est.predict_cate(X)
        baseline = est.predict_baseline(X)
        metrics = evaluate_estimator(T, Y, cate)
        seg = classify_segments(cate, baseline)
        seg_counts = {f"n_{s}": int((seg == s).sum()) for s in np.unique(seg)}
        rows.append(
            {
                "estimator": meta.get("name", mpath.stem),
                "model_file": str(mpath.name),
                "qini": float(metrics["qini_coef"]),
                "auuc": float(metrics["auuc"]),
                "top_10_uplift": float(metrics["top_k_uplift"].get("top_10", 0.0)),
                "top_20_uplift": float(metrics["top_k_uplift"].get("top_20", 0.0)),
                "runtime_s": float(meta.get("runtime_s", float("nan"))),
                "n_train": int(meta.get("n_train", 0)),
                "n_test": int(meta.get("n_test", len(X))),
                "test_split_hash": str(meta.get("test_split_hash", "")),
                **seg_counts,
            }
        )

    df = pd.DataFrame(rows).sort_values("qini", ascending=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    typer.echo(f"Wrote {len(df)} rows to {out}")
    typer.echo(df[["estimator", "qini", "auuc"]].to_string(index=False))

    if not df.empty:
        best = df.iloc[0]["estimator"]
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        BEST_ESTIMATOR_TXT.write_text(f"{best}\n")
        typer.echo(f"Best estimator (by Qini): {best}")


if __name__ == "__main__":
    app()
