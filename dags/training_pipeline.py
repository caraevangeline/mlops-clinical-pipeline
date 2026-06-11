"""
Airflow DAG: Patient Deterioration Risk - Weekly Training Pipeline

Clinical context:
    This DAG trains and validates a model that predicts 30-day all-cause mortality
    risk for heart failure patients. It runs weekly to incorporate new patient data
    and detect distribution shift in clinical measurements before it affects
    prediction quality.

    Reliability is non-negotiable: a silent pipeline failure means clinical staff
    lose their safety net without knowing it. Every task writes a heartbeat signal
    so the monitoring system can detect and alert on silent failures.

Pipeline stages:
    validate_data   → Train quality gate on incoming data
    train_model     → Train + log all artifacts to MLflow
    evaluate_model  → Clinical quality gate (recall ≥ 0.6, no regression vs prod)
    promote_model   → Transition validated model to Production in MLflow registry

Recall is used as the primary metric throughout - see evaluate.py for rationale.
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow")

DATA_PATH = os.getenv("DATA_PATH", "/opt/airflow/data/heart_failure_clinical_records.csv")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
RECALL_THRESHOLD = 0.60
MAX_RECALL_REGRESSION = 0.05

default_args = {
    "owner": "circadia-mlops",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def task_validate_data(**context) -> None:
    """
    Stage 1: Data Quality Gate

    Clinical context:
        Validates the incoming patient dataset before any model training begins.
        In healthcare AI, garbage-in propagates to confident-but-wrong predictions.
        This task fails fast to prevent silent degradation downstream.

        In production, this stage would also strip or verify that all PHI
        (Protected Health Information - MRN, name, date of birth) has been
        de-identified before data enters the training pipeline.

    Raises:
        ValueError: If any data quality check fails - the pipeline halts here
                    and the on-call engineer is paged.
    """
    from src.data.ingest import load_dataset
    from src.data.validate import validate_dataset
    from src.monitoring.pipeline_health import write_heartbeat

    write_heartbeat("validate_data")

    df = load_dataset(DATA_PATH)
    issues = validate_dataset(df)

    if issues:
        raise ValueError(
            f"Data validation failed with {len(issues)} issue(s): {issues}"
        )

    context["ti"].xcom_push(key="row_count", value=len(df))


def task_train_model(**context) -> None:
    """
    Stage 2: Model Training

    Clinical context:
        Trains a RandomForestClassifier on 12 clinical features to predict
        30-day all-cause mortality. Uses class_weight='balanced' to upweight
        the minority (death) class - ensuring the model prioritises catching
        true deteriorations over overall accuracy.

        All experiments, parameters, and artifacts are logged to MLflow for
        full reproducibility and compliance audit trail.
    """
    from src.data.ingest import load_dataset
    from src.features.engineer import engineer_features
    from src.monitoring.pipeline_health import write_heartbeat
    from src.training.train import train_model

    write_heartbeat("train_model")

    df = load_dataset(DATA_PATH)
    X, y = engineer_features(df)
    run_id, model_version = train_model(X, y, mlflow_uri=MLFLOW_TRACKING_URI)

    context["ti"].xcom_push(key="run_id", value=run_id)
    context["ti"].xcom_push(key="model_version", value=model_version)


def task_evaluate_model(**context) -> None:
    """
    Stage 3: Clinical Quality Gate

    Clinical context:
        Compares the newly trained model against the current production model.
        Primary metric: recall, not accuracy.

        Why recall over accuracy?
        - The dataset is class-imbalanced (~68% survived, ~32% died)
        - A model predicting "no risk" for everyone achieves 68% accuracy but
          catches zero deteriorating patients (0% recall)
        - For clinical safety, a false alarm is preferable to a missed deterioration

        Gate criteria:
        - Recall must be ≥ 0.6 (minimum clinical threshold)
        - Recall must not regress more than 5% vs the current production model

        If either criterion fails, this task raises an exception, the pipeline
        halts, and the new model is NOT promoted. The previous production model
        continues serving clinical predictions.
    """
    from src.data.ingest import load_dataset
    from src.evaluation.evaluate import evaluate_model
    from src.features.engineer import engineer_features
    from src.monitoring.pipeline_health import write_heartbeat

    write_heartbeat("evaluate_model")

    run_id = context["ti"].xcom_pull(key="run_id", task_ids="train_model")

    df = load_dataset(DATA_PATH)
    X, y = engineer_features(df)

    result = evaluate_model(
        X=X,
        y=y,
        staging_run_id=run_id,
        mlflow_uri=MLFLOW_TRACKING_URI,
        recall_threshold=RECALL_THRESHOLD,
        max_regression=MAX_RECALL_REGRESSION,
    )

    if not result["passed"]:
        raise ValueError(
            f"Clinical quality gate FAILED: {result['reason']} | "
            f"staging_recall={result['staging_recall']:.3f} | "
            f"production_recall={result.get('production_recall') or 'N/A'}"
        )

    context["ti"].xcom_push(key="evaluation_result", value=result)


def task_promote_model(**context) -> None:
    """
    Stage 4: Production Promotion

    Clinical context:
        Only reached if all quality gates pass. Transitions the validated model
        from MLflow Staging to Production and archives the previous production
        version - preserving it for instant rollback if degradation is detected.

        In a production environment, this task would run after shadow deployment
        (src/deployment/shadow.py) confirms acceptable agreement rate with the
        current production model on live traffic.
    """
    from src.deployment.promote import promote_to_production
    from src.monitoring.pipeline_health import write_heartbeat

    write_heartbeat("promote_model")

    run_id = context["ti"].xcom_pull(key="run_id", task_ids="train_model")
    evaluation_result = context["ti"].xcom_pull(
        key="evaluation_result", task_ids="evaluate_model"
    )

    promote_to_production(
        run_id=run_id,
        mlflow_uri=MLFLOW_TRACKING_URI,
        metrics=evaluation_result,
    )


with DAG(
    dag_id="patient_deterioration_training_pipeline",
    default_args=default_args,
    description=(
        "Weekly retraining pipeline for the patient deterioration risk model. "
        "Recall ≥ 0.6 is the primary quality gate - see evaluate.py for clinical rationale."
    ),
    schedule_interval="@weekly",
    start_date=datetime(2026, 6, 10),
    catchup=False,
    tags=["mlops", "clinical", "heart-failure", "patient-safety"],
) as dag:

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=task_validate_data,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=task_evaluate_model,
    )

    promote = PythonOperator(
        task_id="promote_model",
        python_callable=task_promote_model,
    )

    # Linear dependency chain - each stage must pass before the next begins
    validate >> train >> evaluate >> promote
