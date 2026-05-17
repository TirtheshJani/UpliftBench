"""Chunked parquet iterator and RCT-preserving train/test split.

The split is row-index based and uses a fixed seed. We do NOT stratify by treatment;
the dataset is already an RCT, and stratifying would destroy that property.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def load_parquet_iter(
    path: Path,
    batch_size: int = 500_000,
    columns: list[str] | None = None,
) -> Iterator[pa.RecordBatch]:
    """Yield record batches of at most `batch_size` rows from `path`."""
    pf = pq.ParquetFile(str(path))
    yield from pf.iter_batches(batch_size=batch_size, columns=columns)


def train_test_split_rct(
    n: int,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Random index split with no treatment stratification.

    Returns (train_indices, test_indices), each a sorted int64 numpy array.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * test_frac))
    test = np.sort(perm[:n_test])
    train = np.sort(perm[n_test:])
    return train, test
