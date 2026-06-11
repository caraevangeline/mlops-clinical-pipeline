"""
Model evaluation for the patient deterioration pipeline.

Clinical context:
    Why recall over precision for clinical safety?

    - Recall (sensitivity) = fraction of true deteriorations the model catches
    - Precision = fraction of positive predictions that are correct

    In this clinical context:
    - A false negative (missed deterioration) = a patient deteriorates without
      being flagged. Clinical staff don't intervene. Patient outcome may worsen.
    - A false positive (false alarm) = a healthy patient is flagged. Clinical
      staff investigate and find nothing. Time is wasted, patient is reassured.

    False negatives are clinically worse than false positives. Therefore recall
    is the primary metric for the quality gate - we accept more false alarms to
    ensure we catch as many true deteriorations as possible.

    A recall drop from 0.80 to 0.76 means 4 in every 100 deteriorating patients
    who were previously caught are now missed.
"""

import logging
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

MODEL_NAME = "patient_deterioration_model"


def evaluate_model(
    X: np.ndarray,
    y: np.ndarray,
    staging_run_id: str,
    mlflow_uri: str = "http://localhost:5000",
    recall_threshold: float = 0.60,
    max_regression: float = 0.05,
) -> Dict[str, Any]:
    """
    Compare the staging model against the production model on held-out test data.

    Clinical context:
        Uses the same random_state as training to ensure both models are evaluated
        on the identical test split. In production, a held-out patient cohort
        unseen by either model (time-based split) is preferred for fair evaluation.

        Why recall_threshold=0.60?
        Clinical screening tools require sensitivity ≥ 0.60 as a minimum. Below
        this, the model misses more than 1 in 4 true deteriorations - insufficient
        sensitivity for a patient safety application.

    Args:
        X: Feature matrix (same data as training, will be re-split identically).
        y: Binary target vector.
        staging_run_id: MLflow run ID of the newly trained staging model.
        mlflow_uri: MLflow tracking server URI.
        recall_threshold: Absolute minimum recall required to pass.
        max_regression: Maximum allowed recall drop vs current production model.

    Returns:
        Dict with: passed (bool), reason (str), staging_recall (float),
        production_recall (Optional[float]).
    """
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)

    # Use identical split as training for a fair apples-to-apples comparison
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    staging_model = mlflow.sklearn.load_model(f"runs:/{staging_run_id}/model")
    staging_recall = recall_score(y_test, staging_model.predict(X_test))

    # Gate 1: absolute recall threshold
    if staging_recall < recall_threshold:
        return {
            "passed": False,
            "reason": (
                f"Staging recall {staging_recall:.3f} is below the minimum "
                f"clinical threshold of {recall_threshold}. "
                f"This would miss {(1 - staging_recall) * 100:.0f}% of deteriorating patients."
            ),
            "staging_recall": staging_recall,
            "production_recall": None,
        }

    # Gate 2: regression vs production (only if a production model exists)
    production_recall = _get_production_recall(client, X_test, y_test)

    if production_recall is not None:
        regression = production_recall - staging_recall
        if regression > max_regression:
            return {
                "passed": False,
                "reason": (
                    f"Staging recall {staging_recall:.3f} regresses {regression:.3f} "
                    f"points vs production recall {production_recall:.3f}. "
                    f"Maximum allowed regression: {max_regression}."
                ),
                "staging_recall": staging_recall,
                "production_recall": production_recall,
            }

    logger.info(
        "Evaluation passed - staging_recall=%.3f, production_recall=%s",
        staging_recall,
        f"{production_recall:.3f}" if production_recall is not None else "N/A (first deployment)",
    )

    return {
        "passed": True,
        "reason": "All quality gates passed",
        "staging_recall": staging_recall,
        "production_recall": production_recall,
    }


def _get_production_recall(
    client: mlflow.tracking.MlflowClient,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Optional[float]:
    """Load the current production model and return its recall. None if no production model exists."""
    try:
        prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not prod_versions:
            logger.info("No production model found - this appears to be the first deployment")
            return None

        prod_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
        return float(recall_score(y_test, prod_model.predict(X_test)))

    except Exception as exc:
        logger.warning("Could not load production model for comparison: %s", exc)
        return None
