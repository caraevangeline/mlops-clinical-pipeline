"""
Data validation for the patient deterioration pipeline.

Clinical context:
    Data quality is a patient safety issue. Corrupt, incomplete, or out-of-range
    clinical measurements can cause the model to produce unreliable predictions
    with high confidence - the worst possible failure mode.

    These checks run before training begins and fail loudly. A failed validation
    stops the pipeline and pages the on-call engineer. Silent propagation of bad
    data is more dangerous than a halted pipeline.

    PHI note: In a real system, this module would also verify that all patient
    identifiers (MRN, name, date of birth, ZIP code) have been stripped or
    tokenised before data enters this pipeline. The public Heart Failure dataset
    contains no PHI - it is fully de-identified at source.
"""

import logging
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

# Expected columns and their approximate dtypes
EXPECTED_SCHEMA: Dict[str, type] = {
    "age": float,
    "anaemia": int,
    "creatinine_phosphokinase": int,
    "diabetes": int,
    "ejection_fraction": int,
    "high_blood_pressure": int,
    "platelets": float,
    "serum_creatinine": float,
    "serum_sodium": int,
    "sex": int,
    "smoking": int,
    "time": int,
    "DEATH_EVENT": int,
}

# Physiologically plausible ranges for clinical measurements
RANGE_CHECKS: Dict[str, tuple] = {
    "age": (0, 120),           # human lifespan bounds
    "ejection_fraction": (0, 100),  # percentage - cannot exceed 100%
    "serum_sodium": (100, 200),     # mmol/L; outside this range is life-threatening or impossible
    "serum_creatinine": (0.0, 30.0),  # mg/dL; >30 is incompatible with survival
    "creatinine_phosphokinase": (0, 10_000),  # U/L
    "platelets": (0, 1_500_000),   # kiloplatelets/mL
    "time": (0, 365),              # days of follow-up (study period was 1 year)
}

NULL_RATE_THRESHOLD = 0.05  # alert if any feature exceeds 5% nulls
MIN_RECORDS = 100           # fail pipeline if dataset has fewer than this many rows


def validate_dataset(df: pd.DataFrame) -> List[str]:
    """
    Run all data quality checks on the input dataset.

    Clinical context:
        This is the first line of defence against data pipeline issues.
        In healthcare AI, a model trained on corrupted data produces confident
        but wrong predictions - potentially more dangerous than no prediction
        at all because clinical staff trust it.

    Args:
        df: Raw clinical dataset loaded from ingest.py.

    Returns:
        List of validation failure messages. Empty list means all checks passed.
    """
    issues: List[str] = []

    issues.extend(_check_schema(df))
    issues.extend(_check_ranges(df))
    issues.extend(_check_null_rates(df))
    issues.extend(_check_volume(df))

    if issues:
        logger.error("Data validation failed with %d issue(s)", len(issues))
        for issue in issues:
            logger.error("  FAIL: %s", issue)
    else:
        logger.info("All data validation checks passed (%d rows)", len(df))

    return issues


def _check_schema(df: pd.DataFrame) -> List[str]:
    """Verify all expected columns are present."""
    missing = set(EXPECTED_SCHEMA.keys()) - set(df.columns)
    if missing:
        return [f"Missing columns: {sorted(missing)}"]
    return []


def _check_ranges(df: pd.DataFrame) -> List[str]:
    """
    Verify clinical measurements fall within physiologically plausible ranges.

    Clinical context:
        Values outside these ranges are almost certainly data entry errors or
        ETL bugs, not real patient measurements. Age > 120 or ejection fraction
        > 100% are physiologically impossible. Catching these prevents the model
        from learning from corrupted examples.
    """
    issues = []
    for col, (low, high) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        out_of_range = df[(df[col] < low) | (df[col] > high)]
        if not out_of_range.empty:
            issues.append(
                f"Column '{col}' has {len(out_of_range)} value(s) outside "
                f"plausible range [{low}, {high}]"
            )
    return issues


def _check_null_rates(df: pd.DataFrame) -> List[str]:
    """
    Alert if any feature exceeds 5% null rate.

    Clinical context:
        High missingness often indicates upstream data pipeline failures rather
        than natural data sparsity. The 5% threshold is conservative - clinical
        models should train on high-quality, largely complete data. Imputing
        away 20% nulls masks a data quality problem rather than solving it.
    """
    issues = []
    null_rates = df.isnull().mean()
    high_null = null_rates[null_rates > NULL_RATE_THRESHOLD]
    for col, rate in high_null.items():
        issues.append(
            f"Column '{col}' has {rate:.1%} null rate "
            f"(threshold: {NULL_RATE_THRESHOLD:.0%})"
        )
    return issues


def _check_volume(df: pd.DataFrame) -> List[str]:
    """
    Fail if fewer than MIN_RECORDS records are present.

    Clinical context:
        Training on too few samples risks high-variance models that generalise
        poorly to new patients. The 100-record floor is a minimum safety net,
        not a quality target - the full heart failure dataset has 299 records.
    """
    if len(df) < MIN_RECORDS:
        return [
            f"Dataset has only {len(df)} records "
            f"(minimum required: {MIN_RECORDS})"
        ]
    return []
