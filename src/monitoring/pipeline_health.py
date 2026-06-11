"""
Pipeline health monitoring - dead man's switch implementation.

Clinical context:
    A silent pipeline failure is the highest-risk failure mode in healthcare AI.
    If the training or prediction pipeline fails without raising an alert,
    clinical staff continue using a stale or degraded model with no awareness
    that anything is wrong. The safety net is missing but no alarm has sounded.

    The dead man's switch pattern inverts the failure mode: the system must
    actively signal health at each stage. If the expected signal is absent,
    that absence itself becomes the alert. This ensures a failing pipeline
    cannot fail silently.

    See runbooks/silent_failure_response.md for incident response procedures.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

HEARTBEAT_FILE = os.getenv(
    "HEARTBEAT_FILE", "/opt/airflow/artifacts/pipeline_heartbeat.json"
)
MAX_HEARTBEAT_AGE_MINUTES = 15
PREDICTION_FRESHNESS_HOURS = 24


def write_heartbeat(stage: str) -> None:
    """
    Write a heartbeat timestamp for the current pipeline stage.

    Clinical context:
        Called at the start of each pipeline task so the monitoring check has
        granular visibility into which stage is active. If a task crashes after
        writing its heartbeat but before the next task begins, the monitoring
        check will alert with the correct failed stage identified.

    Args:
        stage: Pipeline stage name (e.g., 'validate_data', 'train_model').
    """
    os.makedirs(os.path.dirname(HEARTBEAT_FILE), exist_ok=True)

    heartbeat = {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }

    with open(HEARTBEAT_FILE, "w") as f:
        json.dump(heartbeat, f)

    logger.info("Heartbeat written - stage='%s'", stage)


def check_pipeline_health(
    max_age_minutes: int = MAX_HEARTBEAT_AGE_MINUTES,
) -> Dict:
    """
    Verify the pipeline has signalled health within the expected window.

    Clinical context:
        This function should run on an independent schedule (e.g., every 5 minutes
        via a monitoring DAG) separate from the main training pipeline. If the main
        pipeline is healthy, heartbeats will always be fresh. If this check fires
        an alert, the pipeline has silently failed.

        The 15-minute threshold is generous - the pipeline should complete each
        stage in under 5 minutes for the heart failure dataset. Adjust for longer
        training jobs.

    Args:
        max_age_minutes: Maximum acceptable heartbeat age before alerting.

    Returns:
        Dict with 'healthy' (bool), 'last_stage', 'age_minutes', 'alert'.
    """
    if not os.path.exists(HEARTBEAT_FILE):
        return {
            "healthy": False,
            "last_stage": None,
            "age_minutes": None,
            "alert": (
                "DEAD MAN SWITCH TRIGGERED: No heartbeat file found. "
                "Pipeline has never completed a stage, or the file was deleted."
            ),
        }

    with open(HEARTBEAT_FILE) as f:
        heartbeat = json.load(f)

    last_ts = datetime.fromisoformat(heartbeat["timestamp"])
    now = datetime.now(timezone.utc)
    age_minutes = (now - last_ts).total_seconds() / 60

    if age_minutes > max_age_minutes:
        return {
            "healthy": False,
            "last_stage": heartbeat["stage"],
            "age_minutes": round(age_minutes, 1),
            "alert": (
                f"DEAD MAN SWITCH TRIGGERED: Last heartbeat was {age_minutes:.1f} minutes ago "
                f"(threshold: {max_age_minutes}m) at stage '{heartbeat['stage']}'. "
                "Pipeline may have failed silently."
            ),
        }

    return {
        "healthy": True,
        "last_stage": heartbeat["stage"],
        "age_minutes": round(age_minutes, 1),
        "alert": None,
    }


def check_prediction_freshness(
    predictions_file: Optional[str] = None,
    max_age_hours: int = PREDICTION_FRESHNESS_HOURS,
) -> Dict:
    """
    Verify predictions were generated within the expected time window.

    Clinical context:
        Stale predictions are a patient safety risk. If the prediction pipeline
        ran yesterday but not today, clinicians are reviewing outdated risk scores
        without knowing it. This check ensures clinicians always have a fresh
        risk assessment, not a stale one that may no longer reflect the patient's
        current condition.

    Args:
        predictions_file: Path to the timestamp file written after each prediction batch.
        max_age_hours: Maximum acceptable age of the last prediction run.

    Returns:
        Dict with 'fresh' (bool), 'age_hours', 'alert'.
    """
    if predictions_file is None:
        predictions_file = os.getenv(
            "PREDICTIONS_TIMESTAMP_FILE",
            "/opt/airflow/artifacts/last_prediction_run.txt",
        )

    if not os.path.exists(predictions_file):
        return {
            "fresh": False,
            "age_hours": None,
            "alert": (
                f"Predictions timestamp file not found at '{predictions_file}'. "
                "Prediction pipeline may never have run."
            ),
        }

    mtime = os.path.getmtime(predictions_file)
    file_dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
    age_hours = (datetime.now(timezone.utc) - file_dt).total_seconds() / 3600

    if age_hours > max_age_hours:
        return {
            "fresh": False,
            "age_hours": round(age_hours, 1),
            "alert": (
                f"Predictions are {age_hours:.1f} hours old (threshold: {max_age_hours}h). "
                "Clinicians may be viewing stale risk scores."
            ),
        }

    return {"fresh": True, "age_hours": round(age_hours, 1), "alert": None}
