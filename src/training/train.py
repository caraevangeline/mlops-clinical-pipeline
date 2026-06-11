"""
Model training for the patient deterioration pipeline.

Clinical context:
    Trains a RandomForestClassifier tuned for recall on the minority (death) class.
    All runs are fully logged to MLflow - every hyperparameter, metric, and artifact
    is stored to support audit trails and model governance requirements.

    class_weight='balanced' upweights samples from the minority class during
    training, directly optimising the model to be more sensitive to true
    deteriorations at the acceptable cost of more false alarms.
"""

import logging
import tempfile
from typing import Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

MODEL_NAME = "patient_deterioration_model"
EXPERIMENT_NAME = "heart_failure_prediction"

# class_weight='balanced' is the key clinical safety choice - it tells sklearn
# to weight each class inversely proportional to its frequency. This means
# errors on the minority (death) class are penalised more during training.
HYPERPARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    mlflow_uri: str = "http://localhost:5000",
    test_size: float = 0.2,
) -> Tuple[str, int]:
    """
    Train a RandomForestClassifier and register it in the MLflow model registry.

    Clinical context:
        Uses a fixed random_state for reproducibility - every training run on
        the same data must produce the same model for regulatory auditability.
        The test split is stratified to ensure the minority (death) class is
        represented proportionally in both train and test sets.

    Args:
        X: Scaled feature matrix from engineer_features().
        y: Binary target vector (0=survived, 1=died).
        mlflow_uri: MLflow tracking server URI.
        test_size: Fraction of data held out for evaluation metrics.

    Returns:
        Tuple of (run_id, model_version) for use by downstream pipeline tasks.
    """
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    with mlflow.start_run() as run:
        mlflow.sklearn.autolog(log_model_signatures=True, log_input_examples=True)

        model = RandomForestClassifier(**HYPERPARAMS)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_params(HYPERPARAMS)
        mlflow.log_metrics(
            {
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "roc_auc": roc_auc,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "positive_rate_train": float(y_train.mean()),
            }
        )

        _log_feature_importances(model)

        model_uri = mlflow.sklearn.log_model(model, artifact_path="model").model_uri
        result = mlflow.register_model(model_uri, MODEL_NAME)

        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=result.version,
            stage="Staging",
        )

        logger.info(
            "Training complete - recall=%.3f, precision=%.3f, f1=%.3f, roc_auc=%.3f",
            recall,
            precision,
            f1,
            roc_auc,
        )
        logger.info(
            "Registered as %s v%s → Staging", MODEL_NAME, result.version
        )
        logger.info(
            "Classification report:\n%s",
            classification_report(y_test, y_pred, target_names=["alive", "died"]),
        )

        return run.info.run_id, int(result.version)


def _log_feature_importances(model: RandomForestClassifier) -> None:
    """Save feature importances as a CSV artifact for clinical interpretability audits."""
    from src.features.engineer import FEATURE_COLUMNS

    importances = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="feature_importances_"
    ) as f:
        importances.to_csv(f, index=False)
        mlflow.log_artifact(f.name, artifact_path="feature_importances")

    logger.info("Feature importances:\n%s", importances.to_string(index=False))
