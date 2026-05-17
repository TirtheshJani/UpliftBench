"""Score a sample of held-out rows with every trained estimator.

Produces `artifacts/scored_sample.parquet`, a slim file (~10 MB at n=200k) that the
Streamlit app loads instead of running inference. Columns:

    f0..f11 (float32), treatment (uint8), visit (uint8),
    pred_cate_<est> (float32) per estimator,
    pred_baseline (float32) from the best estimator,
    segment (categorical).
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
    BEST_ESTIMATOR_TXT,
    CRITEO_PARQUET,
    FEATURES,
    MODELS_DIR,
    OUTCOME_VISIT,
    SCORED_SAMPLE_PARQUET,
    SEED,
    TEST_FRAC,
    TREATMENT_COL,
)
from upliftbench.data.loader import train_test_split_rct
from upliftbench.persistence import load_model
from upliftbench.segmentation import classify_segments

app = typer.Typer(add_completion=False, no_args_is_help=False)


def _estimator_key(name: str) -> str:
    return name.replace("-", "_")


@app.command()
def main(
    parquet_path: Annotated[Path, typer.Option(help="Input parquet.")] = CRITEO_PARQUET,
    models_dir: Annotated[Path, typer.Option(help="Trained models dir.")] = MODELS_DIR,
    out: Annotated[Path, typer.Option(help="Output parquet.")] = SCORED_SAMPLE_PARQUET,
    n: Annotated[int, typer.Option(help="Rows to sample from the held-out test split.")] = 200_000,
    seed: Annotated[int, typer.Option(help="Random seed used in training.")] = SEED,
) -> None:
    if not parquet_path.exists():
        typer.echo(f"Missing parquet: {parquet_path}", err=True)
        raise typer.Exit(1)
    model_files = sorted(models_dir.glob("*.joblib"))
    if not model_files:
        typer.echo(f"No models found in {models_dir}", err=True)
        raise typer.Exit(1)

    cols = [*FEATURES, TREATMENT_COL, OUTCOME_VISIT]
    df = pq.read_table(str(parquet_path), columns=cols).to_pandas()
    _, test_idx = train_test_split_rct(n=len(df), test_frac=TEST_FRAC, seed=seed)
    rng = np.random.default_rng(seed)
    pick = test_idx if len(test_idx) <= n else rng.choice(test_idx, size=n, replace=False)
    pick = np.sort(pick)
    sample_df = df.iloc[pick].reset_index(drop=True)
    X = sample_df[FEATURES].to_numpy(dtype=np.float32, copy=False)
    typer.echo(f"Sampled {len(sample_df):,} held-out rows.")

    best_name: str | None = None
    if BEST_ESTIMATOR_TXT.exists():
        best_name = BEST_ESTIMATOR_TXT.read_text().strip()
        typer.echo(f"Best estimator (from {BEST_ESTIMATOR_TXT.name}): {best_name}")

    # Score each estimator; track baseline from the best one.
    pred_cates: dict[str, np.ndarray] = {}
    pred_baseline: np.ndarray | None = None
    for mpath in model_files:
        meta_path = mpath.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        est_name = meta.get("name", mpath.stem.split("_")[0])
        typer.echo(f"Scoring {est_name} ({mpath.name})...")
        est = load_model(mpath)
        cate = np.asarray(est.predict_cate(X), dtype=np.float32)
        pred_cates[_estimator_key(est_name)] = cate
        if best_name is None and pred_baseline is None:
            pred_baseline = np.asarray(est.predict_baseline(X), dtype=np.float32)
            best_name = est_name
        if est_name == best_name and pred_baseline is None:
            pred_baseline = np.asarray(est.predict_baseline(X), dtype=np.float32)

    assert pred_baseline is not None and best_name is not None

    # Build the output frame.
    for k, arr in pred_cates.items():
        sample_df[f"pred_cate_{k}"] = arr
    sample_df["pred_baseline"] = pred_baseline
    sample_df["pred_cate_best"] = pred_cates[_estimator_key(best_name)]
    sample_df["segment"] = classify_segments(sample_df["pred_cate_best"].to_numpy(), pred_baseline)
    sample_df["segment"] = sample_df["segment"].astype("category")

    out.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_parquet(out, index=False, compression="zstd")
    size_mb = out.stat().st_size / 1e6
    typer.echo(
        f"Wrote {len(sample_df):,} rows, {len(sample_df.columns)} cols, {size_mb:.1f} MB to {out}"
    )

    counts = pd.Series(sample_df["segment"]).value_counts()
    typer.echo("\nSegment counts:")
    typer.echo(counts.to_string())


if __name__ == "__main__":
    app()
