"""
Feature engineering for the patient deterioration pipeline.

Clinical context:
    Prepares clinical measurements for model training. StandardScaler normalises
    features so no single measurement dominates due to scale differences —
    creatinine phosphokinase (range ~23–7861 U/L) would otherwise swamp
    ejection fraction (range ~14–80%) in distance-based models.

    All 12 clinical features are retained: each has documented relevance to
    heart failure outcomes in the literature (ejection fraction, serum creatinine,
    and follow-up time are the strongest predictors per Chicco & Jurman 2020).
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

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
TARGET_COLUMN = "DEATH_EVENT"


def engineer_features(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the feature matrix and target vector from the raw clinical dataset.

    Args:
        df: Raw clinical DataFrame from ingest.py.
        scaler: Pre-fitted StandardScaler for inference-time transforms.
                If None, a new scaler is created.
        fit_scaler: Whether to fit the scaler on this data. Set to False when
                    applying training-time scaling to validation or test data —
                    never fit a scaler on test data (data leakage).

    Returns:
        Tuple (X, y) where X is the scaled feature matrix and y is the binary
        target vector (0=survived, 1=died within follow-up period).
    """
    df = df.copy()

    # Median imputation for any residual nulls — conservative choice that
    # doesn't extrapolate beyond observed values
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].median())

    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    logger.info(
        "Feature matrix: shape=%s, class balance=%d alive / %d died (%.1f%% mortality)",
        X.shape,
        (y == 0).sum(),
        (y == 1).sum(),
        (y == 1).mean() * 100,
    )

    return X, y
