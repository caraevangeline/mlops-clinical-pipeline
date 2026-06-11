"""
Data ingestion for the patient deterioration pipeline.

Clinical context:
    Loads the Heart Failure Clinical Records dataset (Chicco & Jurman, 2020,
    BMC Medical Informatics and Decision Making). 299 patients, 13 clinical
    features, binary outcome: 30-day all-cause mortality.

    In production, this module would connect to a FHIR-compliant clinical data
    warehouse, apply de-identification, and maintain a full provenance chain
    before returning data to the training pipeline.
"""

import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DATASET_URL = "https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records"


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the heart failure clinical records dataset from a CSV file.

    Clinical context:
        This public dataset (299 patients, 13 features) is used to demonstrate
        MLOps patterns. In production this function would:
        1. Pull from a FHIR-compliant clinical data warehouse
        2. Verify dataset provenance and hash against the known training manifest
        3. Log data access to the PHI audit trail (who accessed what, when)
        4. Strip any remaining patient identifiers before returning

    Args:
        path: Filesystem path to heart_failure_clinical_records.csv.

    Returns:
        DataFrame with clinical features and DEATH_EVENT target column.

    Raises:
        FileNotFoundError: If the dataset file does not exist at path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            f"Download from: {DATASET_URL}\n"
            "Place the CSV at the path above, then re-run."
        )

    df = pd.read_csv(path)

    logger.info("Dataset loaded: %d rows, %d columns", len(df), len(df.columns))
    logger.info("Columns: %s", list(df.columns))
    logger.info(
        "Target distribution - alive: %d, died: %d (%.1f%% mortality)",
        (df["DEATH_EVENT"] == 0).sum(),
        (df["DEATH_EVENT"] == 1).sum(),
        df["DEATH_EVENT"].mean() * 100,
    )
    logger.info("Descriptive statistics:\n%s", df.describe().to_string())

    return df
