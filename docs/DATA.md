# Data

## Criteo Uplift Prediction Dataset v2

The dataset is released by Criteo AI Lab. It contains roughly 13.9M rows from a randomized treatment-assignment ad experiment.

| Field | Type | Meaning |
|---|---|---|
| `f0`...`f11` | float32 | Anonymized features |
| `treatment` | uint8 | 1 = user assigned to the treatment arm (was eligible to see the ad campaign) |
| `exposure` | uint8 | 1 = user actually saw an ad (a function of treatment + downstream filtering) |
| `visit` | uint8 | 1 = user visited the advertiser's site within the attribution window |
| `conversion` | uint8 | 1 = user converted within the attribution window |

The conventional outcome for the public benchmark is `visit`. `conversion` is much sparser (~0.2%) and noisier.

## Treatment ratio

The treatment ratio in the released file is approximately 0.85 (roughly 85% of users are in the treatment arm). This is by design from Criteo's RCT. Two consequences:

- **Do not stratify the train/test split on treatment.** Stratification would change the marginal treatment distribution within each split and erode the RCT property that DoWhy refutation depends on. `data/loader.py::train_test_split_rct` does a pure random index split.
- For training, downstream classifiers see a class-balance skew. We do not re-weight, because the treatment-as-feature signal under an RCT is what we want the learner to see.

## Sizes

- Compressed CSV.gz: ~700 MB
- Uncompressed CSV: ~3 GB
- Parquet (zstd + dtype-optimized): ~480 MB on disk
- In-memory pandas DataFrame (14 columns at uint8/float32): ~1.0 GB
- LightGBM `Dataset(..., free_raw_data=True, max_bin=63)`: peaks around 5-6 GB during full-data training

If peak RSS ever exceeds 12 GB on a 16 GB laptop, fall back to a 5M-row stratified-by-treatment sample (the plan documents this). Hyperparameters were tuned so the full-data path stays well under that.

## Download

`scripts/download_data.py` tries `https://criteo-uplift.s3.amazonaws.com/criteo-uplift-v2.1.csv.gz` first, falls back to the Hugging Face Datasets mirror, and logs a SHA-256 checksum after a successful download. The raw and processed files are gitignored; the repo is cloneable without the dataset.

## Conversion to parquet

`scripts/prepare_data.py` calls `data/prepare.csv_to_parquet`, which uses `pyarrow.csv.open_csv` with an explicit column-type map (see `features.CSV_COLUMN_DTYPES`) and writes the file with `compression="zstd"` and the schema from `features.CRITEO_SCHEMA`. The conversion is a single pass with batched writes; peak memory is bounded by `batch_rows` (default 500k).

## Loading for training

```python
from upliftbench.data.loader import load_parquet_iter, train_test_split_rct

for batch in load_parquet_iter(CRITEO_PARQUET, batch_size=500_000):
    ...

train_idx, test_idx = train_test_split_rct(n=row_count, test_frac=0.2, seed=42)
```

`train_test_split_rct` returns sorted int64 numpy arrays so downstream slicing is cache-friendly.
