"""
Model rollback for the patient deterioration pipeline.

Clinical context:
    When production model recall drops below the clinical threshold, the first
    response is always immediate rollback — patient safety before root cause
    analysis. This module restores the previous production model within minutes.

    The rollback is possible because promote.py always archives (never deletes)
    the previous production model. This preservation is a hard architectural
    requirement for clinical AI systems.

    See runbooks/model_degradation_response.md for the full incident response
    procedure, which uses this module as its first step.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import mlflow

logger = logging.getLogger(__name__)

MODEL_NAME = "patient_deterioration_model"


def rollback_to_previous(
    mlflow_uri: str = "http://localhost:5000",
    reason: str = "Unspecified",
) -> str:
    """
    Roll back to the most recently archived production model version.

    Clinical context:
        Rollback should be the FIRST action in any model degradation incident.
        Restore the safety net before investigating the root cause. Every minute
        a degraded model is in production is a minute clinical staff may be acting
        on incorrect risk scores.

    Args:
        mlflow_uri: MLflow tracking server URI.
        reason: Human-readable reason for rollback (logged to model version tags
                for the audit trail — this appears in model governance reports).

    Returns:
        Version number of the restored model.

    Raises:
        ValueError: If no archived version exists to roll back to.
    """
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)

    archived_versions = client.get_latest_versions(MODEL_NAME, stages=["Archived"])
    if not archived_versions:
        raise ValueError(
            f"No archived version of '{MODEL_NAME}' available for rollback. "
            "Cannot restore a previous model — deploy from scratch."
        )

    # Restore the highest-numbered archived version (most recent previous production)
    restore_version = max(archived_versions, key=lambda v: int(v.version))
    rollback_time = datetime.now(timezone.utc).isoformat()

    # Archive the failed production model before restoring
    current_prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if current_prod:
        failed_version = current_prod[0].version
        client.transition_model_version_stage(
            name=MODEL_NAME, version=failed_version, stage="Archived"
        )
        client.set_model_version_tag(
            name=MODEL_NAME, version=failed_version, key="rolled_back_at", value=rollback_time
        )
        client.set_model_version_tag(
            name=MODEL_NAME, version=failed_version, key="rollback_reason", value=reason
        )
        logger.warning(
            "Archived failed production version %s. Reason: %s",
            failed_version,
            reason,
        )

    client.transition_model_version_stage(
        name=MODEL_NAME, version=restore_version.version, stage="Production"
    )
    client.set_model_version_tag(
        name=MODEL_NAME,
        version=restore_version.version,
        key="restored_at",
        value=rollback_time,
    )
    client.set_model_version_tag(
        name=MODEL_NAME,
        version=restore_version.version,
        key="restored_reason",
        value=reason,
    )

    logger.info(
        "ROLLBACK COMPLETE: %s v%s restored to Production at %s",
        MODEL_NAME,
        restore_version.version,
        rollback_time,
    )

    return restore_version.version
