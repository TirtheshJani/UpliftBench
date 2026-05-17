"""CLI wrapper: convert downloaded Criteo CSV.gz to parquet."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import psutil
import typer

from upliftbench.config import CRITEO_CSV, CRITEO_PARQUET, DEFAULT_BATCH_ROWS
from upliftbench.data.prepare import csv_to_parquet

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.command()
def main(
    csv_path: Annotated[Path, typer.Option(help="Input CSV.gz path.")] = CRITEO_CSV,
    parquet_path: Annotated[Path, typer.Option(help="Output parquet path.")] = CRITEO_PARQUET,
    batch_rows: Annotated[
        int, typer.Option(help="Rows per parquet row group.")
    ] = DEFAULT_BATCH_ROWS,
) -> None:
    if not csv_path.exists():
        typer.echo(f"Input not found: {csv_path}", err=True)
        raise typer.Exit(1)
    proc = psutil.Process()
    rss_before = proc.memory_info().rss / 1e9
    t0 = time.time()
    n = csv_to_parquet(csv_path, parquet_path, batch_rows=batch_rows)
    rss_after = proc.memory_info().rss / 1e9
    dt = time.time() - t0
    typer.echo(
        f"Wrote {n:,} rows to {parquet_path} "
        f"in {dt:.1f}s. RSS {rss_before:.2f} GB -> {rss_after:.2f} GB."
    )


if __name__ == "__main__":
    app()
