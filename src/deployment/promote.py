"""
Model promotion from Staging to Production in the MLflow model registry.

Clinical context:
    Production promotion is a controlled operation. The previous production model
    is archived - never deleted - to enable instant rollback without retraining.
    This is a hard requirement for clinical AI systems: you must always be able to
    restore the previous model within minutes of a detected degradation.

    In a production deployment, this function is called only after:
    1. Automated quality gates pass (evaluate_model)
    2. Shadow deployment shows acceptable agreement rate (shadow.py)
    3. A human has reviewed metrics in MLflow (CD workflow manual approval)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import mlflow

logger = logging.getLogger(__name__)

MODEL_NAME = "patient_deterioration_model"


def promote_to_production(
    run_id: Optional[str],
    mlflow_uri: str = "http://localhost:5000",
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Transition the validated model from Staging to Production.

    Archives the current production version before promoting, preserving it
    for instant rollback via src/deployment/rollback.py.

    Args:
        run_id: MLflow run ID of the model being promoted (used for logging only).
        mlflow_uri: MLflow tracking server URI.
        metrics: Evaluation metrics dict to attach as model version tags.

    Returns:
        The version number of the newly promoted production model.

    Raises:
        ValueError: If no model is in Staging stage.
    """
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)

    staging_versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
    if not staging_versions:
        raise ValueError(
            f"No model in Staging for '{MODEL_NAME}'. "
            "Run the training pipeline first."
        )

    staging_version = staging_versions[0]
    promotion_time = datetime.now(timezone.utc).isoformat()

    # Archive current production before promoting - preserves rollback capability
    current_prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if current_prod:
        prev_version = current_prod[0].version
        client.transition_model_version_stage(
            name=MODEL_NAME, version=prev_version, stage="Archived"
        )
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=prev_version,
            key="archived_at",
            value=promotion_time,
        )
        logger.info("Archived previous production version %s", prev_version)

    client.transition_model_version_stage(
        name=MODEL_NAME, version=staging_version.version, stage="Production"
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=staging_version.version,
        key="promoted_at",
        value=promotion_time,
    )
    if metrics:
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=staging_version.version,
            key="promotion_recall",
            value=str(round(metrics.get("staging_recall", 0), 4)),
        )

    logger.info(
        "PROMOTED: %s v%s → Production at %s",
        MODEL_NAME,
        staging_version.version,
        promotion_time,
    )

    return staging_version.version
