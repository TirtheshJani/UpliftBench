"""Feature schema for Criteo Uplift v2."""

import pyarrow as pa

from upliftbench.config import FEATURES, OUTCOME_CONVERSION, OUTCOME_VISIT, TREATMENT_COL

CRITEO_SCHEMA = pa.schema(
    [(f, pa.float32()) for f in FEATURES]
    + [
        (TREATMENT_COL, pa.uint8()),
        ("exposure", pa.uint8()),
        (OUTCOME_VISIT, pa.uint8()),
        (OUTCOME_CONVERSION, pa.uint8()),
    ]
)

CSV_COLUMN_DTYPES: dict[str, str] = {
    **{f: "float32" for f in FEATURES},
    TREATMENT_COL: "uint8",
    "exposure": "uint8",
    OUTCOME_VISIT: "uint8",
    OUTCOME_CONVERSION: "uint8",
}
