"""Project-wide constants: paths, feature names, seed, segmentation thresholds."""

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

CRITEO_CSV = RAW_DIR / "criteo-uplift-v2.1.csv.gz"
CRITEO_PARQUET = PROCESSED_DIR / "criteo.parquet"
LEADERBOARD_PARQUET = ARTIFACTS_DIR / "leaderboard.parquet"
SCORED_SAMPLE_PARQUET = ARTIFACTS_DIR / "scored_sample.parquet"
DOWHY_REFUTATION_JSON = ARTIFACTS_DIR / "dowhy_refutation.json"
TEST_SPLIT_JSON = ARTIFACTS_DIR / "test_split.json"
BEST_ESTIMATOR_TXT = ARTIFACTS_DIR / "best_estimator.txt"

CRITEO_URL_PRIMARY = "https://criteo-uplift.s3.amazonaws.com/criteo-uplift-v2.1.csv.gz"
CRITEO_URL_MIRROR = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/criteo-uplift-v2.1.csv.gz"
)
CRITEO_ROW_COUNT = 13_979_592

FEATURES = [f"f{i}" for i in range(12)]
TREATMENT_COL = "treatment"
OUTCOME_VISIT = "visit"
OUTCOME_CONVERSION = "conversion"
EXPOSURE_COL = "exposure"

SEED = 42
TEST_FRAC = 0.2
DEFAULT_BATCH_ROWS = 500_000

LIGHTGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_bin": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "n_estimators": 200,
    "verbose": -1,
    "n_jobs": -1,
}

BASELINE_THRESHOLD = 0.5
CATE_THRESHOLD = 0.0
