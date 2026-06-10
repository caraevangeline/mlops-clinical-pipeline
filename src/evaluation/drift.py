"""
Data drift detection for the patient deterioration pipeline.

Clinical context:
    Hospital patient populations shift over time. Seasonal illness patterns,
    changes in admission criteria, evolving treatment protocols, and changes
    to the electronic health record system all alter the distribution of
    clinical measurements the model receives.

    A model trained on last year's data may silently underperform on this
    year's patients even without any bugs. Drift detection is the early
    warning system.

    This module uses the Kolmogorov-Smirnov (KS) two-sample test — a
    non-parametric, distribution-free test appropriate for clinical data
    which is rarely normally distributed.
"""

import json
import logging
import os
from typing import Dict

import mlflow
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

DRIFT_P_VALUE_THRESHOLD = 0.05  # KS test significance level
REFERENCE_STATS_FILE = os.getenv(
    "REFERENCE_STATS_PATH", "/opt/airflow/artifacts/reference_stats.json"
)

FEATURE_COLUMNS = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
    "time",
]


def detect_drift(
    current_df: pd.DataFrame,
    reference_df: pd.DataFrame = None,
    mlflow_uri: str = "http://localhost:5000",
) -> Dict[str, bool]:
    """
    Compare current data feature distributions against a reference using the KS test.

    Clinical context:
        p-value < 0.05 indicates the distributions are statistically different —
        the model may be encountering population shift. Drift in serum_creatinine
        or ejection_fraction is especially concerning as these are the strongest
        mortality predictors.

        Drift does not automatically block deployment, but is logged to MLflow
        and should trigger clinical review before the next model promotion.

    Args:
        current_df: New batch of patient records.
        reference_df: Training reference dataset. If None, loads from saved file.
        mlflow_uri: MLflow tracking server URI.

    Returns:
        Dict mapping feature name → True (drifted) / False (stable).
    """
    mlflow.set_tracking_uri(mlflow_uri)

    if reference_df is None:
        reference_df = _load_reference_stats()

    drift_results: Dict[str, bool] = {}
    drift_metrics: Dict[str, float] = {}

    for col in FEATURE_COLUMNS:
        if col not in current_df.columns or col not in reference_df.columns:
            continue

        ks_stat, p_value = stats.ks_2samp(
            reference_df[col].dropna().values,
            current_df[col].dropna().values,
        )

        drifted = p_value < DRIFT_P_VALUE_THRESHOLD
        drift_results[col] = drifted
        drift_metrics[f"drift_ks_{col}"] = float(ks_stat)
        drift_metrics[f"drift_pvalue_{col}"] = float(p_value)

        if drifted:
            logger.warning(
                "DRIFT DETECTED: feature='%s', ks_stat=%.4f, p_value=%.4f",
                col,
                ks_stat,
                p_value,
            )

    drifted_features = [f for f, d in drift_results.items() if d]
    drift_metrics["n_drifted_features"] = float(len(drifted_features))
    drift_metrics["drift_detected"] = float(len(drifted_features) > 0)

    with mlflow.start_run(run_name="drift_detection"):
        mlflow.log_metrics(drift_metrics)
        if drifted_features:
            mlflow.set_tag("drifted_features", ",".join(drifted_features))
            mlflow.set_tag("drift_alert", "true")

    logger.info(
        "Drift check: %d/%d features drifted — %s",
        len(drifted_features),
        len(FEATURE_COLUMNS),
        drifted_features if drifted_features else "none",
    )

    return drift_results


def save_reference_stats(df: pd.DataFrame) -> None:
    """
    Persist training data as the reference baseline for future drift comparisons.

    This should be called once during the initial training run and whenever the
    model is retrained on a substantially different data vintage.
    """
    os.makedirs(os.path.dirname(REFERENCE_STATS_FILE), exist_ok=True)

    stats_dict = {
        col: df[col].tolist() for col in FEATURE_COLUMNS if col in df.columns
    }

    with open(REFERENCE_STATS_FILE, "w") as f:
        json.dump(stats_dict, f)

    logger.info("Reference stats saved to %s", REFERENCE_STATS_FILE)


def _load_reference_stats() -> pd.DataFrame:
    """Load the saved reference distribution."""
    if not os.path.exists(REFERENCE_STATS_FILE):
        raise FileNotFoundError(
            f"Reference stats not found at '{REFERENCE_STATS_FILE}'. "
            "Call save_reference_stats() during initial model training."
        )
    with open(REFERENCE_STATS_FILE) as f:
        return pd.DataFrame(json.load(f))
