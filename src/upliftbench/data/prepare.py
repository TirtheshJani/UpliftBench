"""Stream a Criteo CSV.gz to parquet with dtype-downcast columns."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from upliftbench.features import CRITEO_SCHEMA


def csv_to_parquet(
    csv_path: Path,
    parquet_path: Path,
    batch_rows: int = 500_000,
    compression: str = "zstd",
) -> int:
    """Stream `csv_path` (CSV or CSV.gz) into a parquet file with the Criteo schema.

    Returns the total row count written. Memory footprint is bounded by `batch_rows`.
    """
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # `block_size` is in bytes; ~256 bytes per row is a generous upper bound for
    # Criteo's 16 numeric columns. Setting it as a function of `batch_rows` keeps
    # peak memory roughly proportional to the caller's chunk-size choice.
    read_opts = pacsv.ReadOptions(block_size=batch_rows * 256)
    parse_opts = pacsv.ParseOptions(delimiter=",")
    # Explicit column types here are what makes the parquet dtype-optimized.
    # Without them, pyarrow infers int64 / double which would balloon RSS.
    convert_opts = pacsv.ConvertOptions(
        column_types={field.name: field.type for field in CRITEO_SCHEMA},
    )

    n_rows = 0
    with pacsv.open_csv(
        str(csv_path),
        read_options=read_opts,
        parse_options=parse_opts,
        convert_options=convert_opts,
    ) as reader:
        writer: pq.ParquetWriter | None = None
        try:
            for batch in reader:
                table = pa.Table.from_batches([batch], schema=CRITEO_SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(
                        str(parquet_path), CRITEO_SCHEMA, compression=compression
                    )
                writer.write_table(table)
                n_rows += batch.num_rows
        finally:
            if writer is not None:
                writer.close()

    return n_rows
