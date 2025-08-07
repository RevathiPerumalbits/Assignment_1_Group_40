import logging
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(model, model_name, X_train, y_train, X_test, y_test):
    """Train a single model and log to MLflow."""
    logger.info(
        f"Training {model_name} with MLflow tracking URI: {mlflow.get_tracking_uri()}"
    )
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        input_example = X_train.iloc[:1]
        mlflow.sklearn.log_model(
            model, artifact_path=model_name, input_example=input_example
        )
        print(f"{model_name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

        return model, accuracy, run.info.run_id


def main():
    """Train multiple models and return their details."""
    # Set MLflow tracking and artifact URIs
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    # mlflow.set_artifact_location("file://mlruns")
    # Load processed data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
    experiment_name = "iris_classification"
    # Set MLflow tracking
    mlflow.set_experiment("iris_classification")
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            mlflow.create_experiment(experiment, artifact_location="file://mlruns")
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        logger.error(f"Error setting up experiment: {e}")
        raise
    # Train models
    models = [
        (LogisticRegression(max_iter=200), "LogisticRegression"),
        (RandomForestClassifier(n_estimators=100, random_state=42), "RandomForest"),
    ]

    trained_models = []
    for model, name in models:
        trained_model, accuracy, run_id = train_model(
            model, name, X_train, y_train, X_test, y_test
        )
        trained_models.append((trained_model, name, accuracy, run_id))

    return trained_models, X_train


if __name__ == "__main__":
    # Ensure MLflow tracking and artifact URIs are set for local testing
    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlruns.db")
    trained_models, X_train = main()
