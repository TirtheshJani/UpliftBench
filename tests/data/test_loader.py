"""TDD for src/upliftbench/data/loader.py: chunked iterator and RCT-preserving split."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from upliftbench.config import FEATURES, TREATMENT_COL
from upliftbench.data.loader import load_parquet_iter, train_test_split_rct
from upliftbench.features import CRITEO_SCHEMA


def _write_tiny_parquet(path: Path, n_rows: int, treatment_rate: float = 0.5) -> None:
    rng = np.random.default_rng(0)
    arrays = [pa.array(rng.standard_normal(n_rows).astype(np.float32)) for _ in FEATURES]
    treatment = (rng.random(n_rows) < treatment_rate).astype(np.uint8)
    exposure = treatment.copy()
    visit = rng.integers(0, 2, n_rows, dtype=np.uint8)
    conv = (visit & rng.integers(0, 2, n_rows, dtype=np.uint8)).astype(np.uint8)
    arrays += [pa.array(treatment), pa.array(exposure), pa.array(visit), pa.array(conv)]
    table = pa.Table.from_arrays(arrays, schema=CRITEO_SCHEMA)
    pq.write_table(table, str(path))


def test_load_parquet_iter_batch_sizes_and_count(tmp_path: Path) -> None:
    p = tmp_path / "tiny.parquet"
    _write_tiny_parquet(p, n_rows=350)
    total = 0
    sizes: list[int] = []
    for batch in load_parquet_iter(p, batch_size=100):
        sizes.append(batch.num_rows)
        total += batch.num_rows
    assert total == 350
    assert max(sizes) <= 100


def test_load_parquet_iter_column_subset(tmp_path: Path) -> None:
    p = tmp_path / "tiny.parquet"
    _write_tiny_parquet(p, n_rows=50)
    cols = ["f0", TREATMENT_COL]
    for batch in load_parquet_iter(p, batch_size=50, columns=cols):
        assert batch.schema.names == cols


def test_train_test_split_rct_disjoint_and_preserves_ratio(tmp_path: Path) -> None:
    p = tmp_path / "tiny.parquet"
    _write_tiny_parquet(p, n_rows=5_000, treatment_rate=0.85)  # Criteo's actual ratio
    table = pq.read_table(p)
    treatment = table.column(TREATMENT_COL).to_numpy()

    train_idx, test_idx = train_test_split_rct(n=len(treatment), test_frac=0.2, seed=42)
    assert len(set(train_idx)) + len(set(test_idx)) == len(treatment)
    assert set(train_idx).isdisjoint(set(test_idx))

    train_ratio = float(treatment[train_idx].mean())
    test_ratio = float(treatment[test_idx].mean())
    full_ratio = float(treatment.mean())
    assert abs(train_ratio - full_ratio) < 0.02
    assert abs(test_ratio - full_ratio) < 0.02


def test_train_test_split_rct_deterministic() -> None:
    a_train, a_test = train_test_split_rct(n=1000, seed=42)
    b_train, b_test = train_test_split_rct(n=1000, seed=42)
    np.testing.assert_array_equal(a_train, b_train)
    np.testing.assert_array_equal(a_test, b_test)


def test_train_test_split_rct_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        train_test_split_rct(n=10, test_frac=0.0)
    with pytest.raises(ValueError):
        train_test_split_rct(n=10, test_frac=1.0)
