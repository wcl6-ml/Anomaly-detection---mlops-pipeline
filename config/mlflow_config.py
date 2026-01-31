import mlflow
import os

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_ARTIFACT_ROOT = "./mlruns/artifacts"

def setup_mlflow():
    """Initialize MLflow with SQLite backend."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    return MLFLOW_TRACKING_URI