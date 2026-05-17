"""Save/load trained estimator with sibling JSON metadata.

Layout for `save_model(est, "s-learner", out_dir, metadata={...})`:

    out_dir/s-learner_20260517_a1b2c3d.joblib
    out_dir/s-learner_20260517_a1b2c3d.json
"""

from __future__ import annotations

import datetime as dt
import importlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ModelPaths:
    model: Path
    metadata: Path


def _git_sha(default: str = "nogit") -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return default


def _lib_versions() -> dict[str, str]:
    # Pinning library versions into model metadata makes a stale leaderboard
    # comparison detectable: if two models report different LightGBM versions,
    # their numbers may not be directly comparable.
    libs = ("lightgbm", "causalml", "econml", "dowhy", "pyarrow", "scikit-learn", "numpy", "pandas")
    out: dict[str, str] = {}
    for name in libs:
        try:
            mod = importlib.import_module(name.replace("-", "_"))
            out[name] = getattr(mod, "__version__", "?")
        except ImportError:
            out[name] = "missing"
    return out


def save_model(
    estimator: Any,
    name: str,
    out_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> ModelPaths:
    """Persist `estimator` to `<out_dir>/<name>_<date>_<sha>.joblib` plus sibling .json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = _git_sha()
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
    base = out_dir / f"{name}_{stamp}_{sha}"
    model_path = base.with_suffix(".joblib")
    meta_path = base.with_suffix(".json")

    joblib.dump(estimator, model_path)

    meta: dict[str, Any] = {
        "name": name,
        "git_sha": sha,
        "saved_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "lib_versions": _lib_versions(),
    }
    if metadata:
        meta.update(metadata)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return ModelPaths(model=model_path, metadata=meta_path)


def load_model(path: Path) -> Any:
    """Load a joblib-serialized estimator from `path`."""
    return joblib.load(path)
