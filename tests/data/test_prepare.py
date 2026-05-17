"""TDD for src/upliftbench/data/prepare.py: streaming CSV.gz to parquet with dtype downcast."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from upliftbench.config import FEATURES, OUTCOME_CONVERSION, OUTCOME_VISIT, TREATMENT_COL
from upliftbench.data.prepare import csv_to_parquet


def _write_fake_criteo_csv(path: Path, n_rows: int) -> None:
    cols = FEATURES + [TREATMENT_COL, "exposure", OUTCOME_VISIT, OUTCOME_CONVERSION]
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((n_rows, len(FEATURES))).astype(np.float32)
    treatment = rng.integers(0, 2, n_rows, dtype=np.uint8)
    exposure = rng.integers(0, 2, n_rows, dtype=np.uint8)
    visit = rng.integers(0, 2, n_rows, dtype=np.uint8)
    conv = rng.integers(0, 2, n_rows, dtype=np.uint8)
    lines = [",".join(cols)]
    for i in range(n_rows):
        row_vals = [f"{x:.6f}" for x in feats[i]]
        row_vals += [
            str(int(treatment[i])),
            str(int(exposure[i])),
            str(int(visit[i])),
            str(int(conv[i])),
        ]
        lines.append(",".join(row_vals))
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines))


def test_csv_to_parquet_dtypes_and_rowcount(tmp_path: Path) -> None:
    csv_path = tmp_path / "fake.csv.gz"
    parquet_path = tmp_path / "fake.parquet"
    _write_fake_criteo_csv(csv_path, n_rows=100)

    n = csv_to_parquet(csv_path, parquet_path, batch_rows=20)
    assert n == 100

    table = pq.read_table(parquet_path)
    assert table.num_rows == 100

    schema = table.schema
    for f in FEATURES:
        assert str(schema.field(f).type) == "float", f"{f} should be float32"
    for c in (TREATMENT_COL, "exposure", OUTCOME_VISIT, OUTCOME_CONVERSION):
        assert str(schema.field(c).type) == "uint8", f"{c} should be uint8"
