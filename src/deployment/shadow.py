"""
Shadow deployment for the patient deterioration model.

Clinical context:
    Why shadow before canary?

    In a canary deployment, a small percentage of real users receive predictions
    from the new model. For a general consumer app, a 1% canary is low-risk.
    For a clinical safety system, even 1% of patients receiving worse predictions
    is unacceptable — it means real clinical decisions are made on an unvalidated model.

    Shadow deployment eliminates this risk: the candidate model runs on all real
    patient data but its predictions are never exposed to clinicians or acted upon.
    Only the production model's predictions are surfaced. The shadow model runs
    silently alongside, logging its predictions for offline comparison.

    Only after shadow agreement rate is acceptable (≥ 90%) do we proceed to
    full promotion. This is the pattern used by safety-critical systems where
    even a brief exposure of users to degraded performance is unacceptable.
"""

import logging
from datetime import datetime, timezone
from typing import Dict

import mlflow
import mlflow.sklearn
import numpy as np

logger = logging.getLogger(__name__)

MODEL_NAME = "patient_deterioration_model"


def run_shadow_deployment(
    X: np.ndarray,
    mlflow_uri: str = "http://localhost:5000",
    agreement_threshold: float = 0.90,
) -> Dict[str, float]:
    """
    Run production and staging models on the same input; compare predictions.

    Clinical context:
        Agreement rate measures how often the candidate model would have made the
        same clinical decision as the production model. < 90% agreement suggests
        the candidate behaves substantially differently and warrants investigation
        before full promotion — even if its aggregate recall is acceptable.

        We also track whether the staging model catches more deteriorations than
        production (false negatives in production that staging catches). A staging
        model that catches more deteriorations while maintaining high agreement
        is a positive signal for promotion.

    Args:
        X: Feature matrix representing a patient batch.
        mlflow_uri: MLflow tracking server URI.
        agreement_threshold: Minimum prediction agreement rate to log as passing.

    Returns:
        Dict with agreement_rate, n_disagreements, n_patients, shadow_passed.
    """
    mlflow.set_tracking_uri(mlflow_uri)

    prod_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    staging_model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Staging")

    prod_predictions = prod_model.predict(X)
    staging_predictions = staging_model.predict(X)

    agreement = float(np.mean(prod_predictions == staging_predictions))
    n_disagreements = int(np.sum(prod_predictions != staging_predictions))

    # Among disagreements, how often does staging flag a patient that prod misses?
    # This is the most clinically relevant disagreement type.
    disagreement_mask = prod_predictions != staging_predictions
    staging_catches_more = int(
        np.sum(
            (prod_predictions[disagreement_mask] == 0)
            & (staging_predictions[disagreement_mask] == 1)
        )
    )

    result = {
        "agreement_rate": agreement,
        "n_disagreements": float(n_disagreements),
        "n_patients": float(len(X)),
        "staging_catches_more_deteriorations": float(staging_catches_more),
        "shadow_passed": agreement >= agreement_threshold,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with mlflow.start_run(run_name="shadow_deployment"):
        mlflow.log_metrics(
            {k: v for k, v in result.items() if isinstance(v, (int, float))}
        )
        mlflow.set_tag("shadow_passed", str(result["shadow_passed"]))
        mlflow.set_tag("agreement_threshold", str(agreement_threshold))

    logger.info(
        "Shadow deployment: agreement=%.3f (%d/%d patients agree), "
        "staging_catches_more=%d, passed=%s",
        agreement,
        len(X) - n_disagreements,
        len(X),
        staging_catches_more,
        result["shadow_passed"],
    )

    if not result["shadow_passed"]:
        logger.warning(
            "Shadow agreement %.3f below threshold %.3f — "
            "investigate before promoting to production",
            agreement,
            agreement_threshold,
        )

    return result
