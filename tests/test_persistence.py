"""TDD for src/upliftbench/persistence.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from upliftbench.persistence import load_model, save_model


class _DummyEstimator:
    name = "dummy"

    def __init__(self, value: float = 0.42) -> None:
        self.value = value

    def fit(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> None:
        return None

    def predict_cate(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self.value, dtype=np.float64)

    def predict_baseline(self, X: np.ndarray) -> np.ndarray:
        return np.zeros(len(X), dtype=np.float64)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    est = _DummyEstimator(value=0.7)
    meta = {
        "runtime_s": 12.3,
        "n_train": 1000,
        "qini": 0.05,
        "auuc": 0.03,
        "seed": 42,
    }
    paths = save_model(est, "dummy", out_dir=tmp_path, metadata=meta)
    assert paths.model.exists()
    assert paths.metadata.exists()

    loaded = load_model(paths.model)
    X = np.zeros((4, 12), dtype=np.float32)
    np.testing.assert_allclose(loaded.predict_cate(X), 0.7)

    parsed = json.loads(paths.metadata.read_text())
    assert parsed["name"] == "dummy"
    assert parsed["runtime_s"] == 12.3
    assert parsed["n_train"] == 1000
    assert "git_sha" in parsed
    assert "lib_versions" in parsed
    assert "lightgbm" in parsed["lib_versions"]


def test_save_model_creates_dir(tmp_path: Path) -> None:
    nested = tmp_path / "sub" / "dir"
    est = _DummyEstimator()
    paths = save_model(est, "n", out_dir=nested, metadata={})
    assert paths.model.parent == nested
