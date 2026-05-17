"""DoWhy four-step pipeline + four refuters on a stratified-by-treatment sample.

Writes the result to `artifacts/dowhy_refutation.json` so the blog and Streamlit
can render the refutation table without rerunning the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import pyarrow.parquet as pq
import typer

from upliftbench.config import (
    ARTIFACTS_DIR,
    CRITEO_PARQUET,
    DOWHY_REFUTATION_JSON,
    FEATURES,
    OUTCOME_VISIT,
    SEED,
    TREATMENT_COL,
)
from upliftbench.refute.dowhy_pipeline import run_dowhy

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    parquet_path: Annotated[Path, typer.Option(help="Input parquet.")] = CRITEO_PARQUET,
    sample: Annotated[int, typer.Option(help="Stratified row sample size.")] = 1_000_000,
    seed: Annotated[int, typer.Option(help="Random seed.")] = SEED,
    out: Annotated[Path, typer.Option(help="Output JSON.")] = DOWHY_REFUTATION_JSON,
    estimator_method: Annotated[
        str, typer.Option(help="DoWhy method_name string.")
    ] = "backdoor.linear_regression",
) -> None:
    if not parquet_path.exists():
        typer.echo(f"Missing parquet: {parquet_path}", err=True)
        raise typer.Exit(1)
    cols = [*FEATURES, TREATMENT_COL, OUTCOME_VISIT]
    typer.echo(f"Loading {len(cols)} columns from {parquet_path}...")
    df = pq.read_table(str(parquet_path), columns=cols).to_pandas()
    typer.echo(f"  Loaded {len(df):,} rows. Stratified sample n={sample:,}.")

    typer.echo("Running DoWhy (model, identify, estimate, refute x4)...")
    result = run_dowhy(
        df,
        feature_cols=list(FEATURES),
        treatment_col=TREATMENT_COL,
        outcome_col=OUTCOME_VISIT,
        sample_n=sample,
        seed=seed,
        estimator_method=estimator_method,
    )
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    typer.echo(f"\nPoint estimate (ATE on visit): {result['estimate']:+.6f}")
    typer.echo("Refutations:")
    for name, info in result["refutations"].items():
        typer.echo(
            f"  {name:30s} new_estimate={info.get('new_estimate', float('nan')):+.6f}  "
            f"p_value={info.get('p_value', float('nan')):.4f}"
        )
    typer.echo(f"\nWrote {out}")


if __name__ == "__main__":
    app()
