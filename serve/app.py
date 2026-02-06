"""FastAPI application for serving fraud detection model."""
import sys
from pathlib import Path
import os

# Add project root to path and change working directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import mlflow.pyfunc
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from prometheus_fastapi_instrumentator import Instrumentator
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import yaml

from src.drift.detector import DriftDetector

# Define the local path where Docker will have the model
MODEL_PATH = Path(__file__).parent / "model/artifacts"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time anomaly detection for fraud prevention",
    version="1.0.0"
)

# For Grafana monitoring
prediction_counter = Counter('model_predictions_total', 'Total predictions made')
prediction_latency = Histogram('model_inference_seconds', 'Model inference latency')
anomaly_score_gauge = Gauge('model_anomaly_score', 'Latest average anomaly score')
feature_null_rate = Gauge('feature_null_rate', 'Null rate in input features', ['feature_index'])
error_counter = Counter('prediction_errors_total', 'Total prediction errors', ['error_type'])

# Define Gauge for Prometheus
MODEL_DRIFT_GAUGE = Gauge(
    "model_drift_psi", 
    "Population Stability Index for data drift",
    ["feature"]
)

# Initialize Instrumentator but DON'T expose yet
instrumentator = Instrumentator().instrument(app)

# Create the Prometheus ASGI app
metrics_app = make_asgi_app()

# Mount it to the /metrics path
app.mount("/metrics", metrics_app)

# Global model variable
model = None
model_metadata = {}

# For drift detection
drift_detector = None  

class PredictionRequest(BaseModel):
    """Request schema for predictions."""
    features: List[List[float]] = Field(
        ..., 
        description="List of feature vectors (each should have 29 features)",
        example=[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
    )


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    model_config = {"protected_namespaces": ()}  
    predictions: List[int]
    anomaly_scores: List[float]
    model_version: str
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    model_config = {"protected_namespaces": ()}
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


@app.on_event("startup")
async def load_model():
    """Load model from local folder on startup."""
    global model, model_metadata, drift_detector
    
    try:
        logger.info(f"Looking for model in: {MODEL_PATH}")
        
        # 1. Check if the directory exists
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model directory not found at {MODEL_PATH}")

        # 2. Load the model using pyfunc from the LOCAL path
        # MLflow knows how to read its own exported directory
        model = mlflow.pyfunc.load_model(str(MODEL_PATH))
        
        # 3. Extract metadata from the MLmodel file (created by MLflow during export)
        # This allows us to still know the run_id and version without the DB!
        mlmodel_file = MODEL_PATH / "MLmodel"
        if mlmodel_file.exists():
            with open(mlmodel_file, "r") as f:
                config = yaml.safe_load(f)
                model_metadata = {
                    "run_id": config.get("run_id", "unknown"),
                    "version": "baked-in", # Since it's inside the Docker image
                    "utc_time_created": config.get("utc_time_created", "unknown")
                }
        
        model_metadata["startup_time"] = datetime.now()
        logger.info(f"Successfully loaded model from {MODEL_PATH}")

        # Load reference data for drift detection
        ref_path = project_root / "data/processed/reference.csv"
        if ref_path.exists():
            reference_df = pd.read_csv(ref_path)
            drift_detector = DriftDetector(reference_df)
            logger.info("Drift detector initialized with reference data.")
        else:
            logger.warning(f"Reference data not found at {ref_path}. Drift detection disabled.")
        
        # set up the dynamic threshold
        ref_path = project_root / "data/processed/reference.csv"
        if ref_path.exists():
            reference_df = pd.read_csv(ref_path)
            
            # 1. Calculate "System Noise" (Self-PSI)
            # Split reference data to see what 'natural' variance looks like
            mid = len(reference_df) // 2
            ref_a = reference_df.iloc[:mid]
            ref_b = reference_df.iloc[mid:]
            
            temp_detector = DriftDetector(ref_a)
            # Calculate PSI of one half vs the other
            self_drift_results = temp_detector.detect_drift(ref_b)
            self_psi = self_drift_results['overall_psi']
            
            # 2. Set dynamic threshold: 0.2 (Standard) + Self-PSI (Noise)
            # This accounts for the 'split issue' you mentioned
            dynamic_threshold = 0.16 + self_psi
            
            drift_detector = DriftDetector(reference_df, threshold_psi=dynamic_threshold)
            
            logger.info(f"Drift detector initialized. Base Noise: {self_psi:.4f}")
            logger.info(f"Dynamic Alert Threshold set to: {dynamic_threshold:.4f}")
    except Exception as e:
        logger.error(f"Critical error loading model: {e}")
        # In a real production system, you might want the container to crash 
        # (exit 1) if the model fails to load so the orchestrator (K8s) restarts it.
        model = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "status": "running" if model is not None else "model_not_loaded",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with model validation."""
    if model is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            model_version="none",
            uptime_seconds=0.0
        )
    
    # Calculate uptime
    uptime = (datetime.now() - model_metadata.get("startup_time", datetime.now())).total_seconds()
    
    # Test inference on dummy data
    try:
        # Adjust number of features based on your model
        dummy_input = pd.DataFrame([[0.0] * 29])
        _ = model.predict(dummy_input)
        status = "healthy"
    except Exception as e:
        logger.error(f"Health check inference failed: {e}")
        status = "degraded"
    
    return HealthResponse(
        status=status,
        model_loaded=True,
        model_version=str(model_metadata.get("version", "unknown")),
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        error_counter.labels(error_type='model_not_loaded').inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(request.features)
        
        # Ensure we have column names for the drift detector if it expects them
        # If your detector uses index-based slicing, this is fine
        if drift_detector is not None:
            drift_feature_count = len(drift_detector.feature_names)
            drift_df = df.iloc[:, :drift_feature_count]
            # Rename columns to match what the detector was initialized with
            drift_df.columns = drift_detector.feature_names
            
            drift_results = drift_detector.detect_drift(drift_df)
            MODEL_DRIFT_GAUGE.labels(feature="overall").set(float(drift_results['overall_psi']))

        # Track null rates - ENSURE float conversion for Prometheus
        for i in range(len(df.columns)):
            null_count = int(df.iloc[:, i].isna().sum())
            rate = float(null_count / len(df))
            feature_null_rate.labels(feature_index=str(i)).set(rate)
        
        # Inference
        start_time = datetime.now()
        predictions = model.predict(df)
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # Convert results to standard Python types to satisfy Pydantic/JSON
        if hasattr(predictions, 'tolist'):
            pred_list = [int(x) for x in predictions.tolist()]
        else:
            pred_list = [int(x) for x in predictions]
        
        # isolation forest anomaly scores (convert to float)
        anomaly_scores = [abs(float(p)) for p in pred_list]
        
        # Binary preds (ensure native int)
        if len(anomaly_scores) > 1:
            threshold = float(np.percentile(anomaly_scores, 90))
        else:
            threshold = 0.0
            
        binary_preds = [1 if float(score) > threshold else 0 for score in anomaly_scores]
        
        # Update metrics with native floats
        prediction_counter.inc(len(binary_preds))
        anomaly_score_gauge.set(float(np.mean(anomaly_scores)))
        
        return PredictionResponse(
            predictions=binary_preds,
            anomaly_scores=anomaly_scores,
            model_version=str(model_metadata.get("version", "unknown")),
            inference_time_ms=float(inference_time * 1000)
        )
        
@app.get("/model-info")
async def model_info():
    """Get information about the currently loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "fraud-detector",
        "version": str(model_metadata.get("version", "unknown")),
        "run_id": str(model_metadata.get("run_id", "unknown")),
        "loaded_at": model_metadata.get("startup_time").isoformat() if model_metadata.get("startup_time") else "unknown",
        "stage": "Production"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server from __main__")
    uvicorn.run(app, host="0.0.0.0", port=8000)